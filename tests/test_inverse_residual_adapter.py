import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_inverse_residual_adapter import (
    build_parser as build_eval_parser,
    load_controller,
)
from audit_inverse_residual_adapter import (
    build_parser as build_audit_parser,
    run as run_adapter_audit,
)
from evaluation_controls import (
    expected_clipped_uniform_action,
    prepare_evaluation_command,
)
from inverse_residual_core import (
    InverseResidualAdapter,
    InverseResidualConfig,
    InverseResidualController,
    ResampledAntitheticChannel,
    adapter_objective,
    baseline_commands_from_desired,
    expected_clipped_uniform_action_torch,
    exact_inverse_baseline_commands,
    module_state_sha256,
    value_estimator_metadata,
)
from td3bc_core import TD3BCAgent, TD3BCConfig
from train_inverse_residual_adapter import (
    build_observation_split,
    build_parser as build_train_parser,
    calibrate_value_scale,
    load_frozen_physical_agent,
    load_observations,
    prepare_fresh_output_dir,
    require_new_output_file,
    resolve_channel_calibration,
    run as run_adapter_training,
)


def make_base_agent(action_pairing="executed_executed"):
    config = TD3BCConfig(
        observation_dim=2,
        action_dim=1,
        hidden_dim=8,
        depth=1,
        action_pairing=action_pairing,
    )
    return TD3BCAgent(
        config,
        torch.device("cpu"),
        np.asarray([0.5, -0.25], dtype=np.float32),
        np.asarray([2.0, 0.5], dtype=np.float32),
    )


def make_config(**overrides):
    values = dict(
        observation_dim=2,
        action_dim=1,
        hidden_dim=8,
        depth=1,
        execution_noise_beta=0.4,
        execution_noise_samples=4,
        execution_noise_seed=71,
        delta_max=0.25,
        alpha=1.0,
        residual_penalty=0.5,
    )
    values.update(overrides)
    return InverseResidualConfig(**values)


def test_adapter_is_exactly_zero_initialized_and_delta_zero_is_identity():
    normalized = torch.randn(7, 2)
    baseline = torch.linspace(-0.8, 0.8, 7)[:, None]
    adapter = InverseResidualAdapter(make_config())
    command, proposed, applied = adapter.compose_command(normalized, baseline)
    torch.testing.assert_close(command, baseline, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(proposed) == 0
    assert torch.count_nonzero(applied) == 0

    zero_adapter = InverseResidualAdapter(make_config(delta_max=0.0))
    with torch.no_grad():
        for parameter in zero_adapter.parameters():
            parameter.fill_(10.0)
    zero_command, zero_proposed, _ = zero_adapter.compose_command(
        normalized, baseline
    )
    torch.testing.assert_close(zero_command, baseline, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(zero_proposed) == 0


def test_beta_zero_inverse_is_desired_action_and_k_independent():
    desired = torch.tensor([[-1.0], [-0.4], [0.0], [0.75], [1.0]])
    for sample_count in (1, 4):
        config = make_config(
            execution_noise_beta=0.0,
            execution_noise_samples=sample_count,
        )
        baseline, audit = exact_inverse_baseline_commands(desired, config)
        torch.testing.assert_close(baseline, desired, rtol=0.0, atol=0.0)
        assert audit["command_transform_saturation_count"] == 0
        channel = ResampledAntitheticChannel(config, torch.device("cpu"))
        noise = channel.sample(5, dtype=torch.float32, device=torch.device("cpu"))
        assert noise.shape == (5, sample_count, 1)
        assert torch.count_nonzero(noise) == 0


@pytest.mark.parametrize("beta", (0.0, 0.05, 0.4, 1.25, 2.5))
def test_torch_exact_clipped_uniform_mean_matches_numpy_random_and_boundaries(beta):
    rng = np.random.default_rng(20260914)
    random_commands = rng.uniform(-3.0, 3.0, size=(97, 2))
    boundary_commands = np.asarray(
        [
            [-3.0, -3.0],
            [-1.0 - beta, -0.5 - beta],
            [-1.0, -0.5],
            [-1.0 + beta, -0.5 + beta],
            [0.0, 0.125],
            [1.0 - beta, 0.75 - beta],
            [1.0, 0.75],
            [1.0 + beta, 0.75 + beta],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    commands = np.concatenate((random_commands, boundary_commands), axis=0)
    low = np.asarray([-1.0, -0.5], dtype=np.float64)
    high = np.asarray([1.0, 0.75], dtype=np.float64)
    expected = expected_clipped_uniform_action(commands, low, high, beta)
    command_tensor = torch.tensor(commands, dtype=torch.float64, requires_grad=True)
    actual = expected_clipped_uniform_action_torch(
        command_tensor,
        torch.tensor(low, dtype=torch.float64),
        torch.tensor(high, dtype=torch.float64),
        beta,
    )
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-12, atol=1e-12)
    actual.sum().backward()
    assert command_tensor.grad is not None
    assert torch.isfinite(command_tensor.grad).all()


def test_identity_baseline_is_direct_actor_control_not_channel_inverse():
    desired = torch.tensor([[-0.6], [0.0], [0.6]])
    identity = make_config(
        execution_noise_beta=1.0, baseline_transform="identity"
    )
    inverse = make_config(
        execution_noise_beta=1.0, baseline_transform="inverse"
    )
    direct, direct_audit = baseline_commands_from_desired(desired, identity)
    compensated, _ = baseline_commands_from_desired(desired, inverse)
    torch.testing.assert_close(direct, desired, rtol=0.0, atol=0.0)
    assert direct_audit["command_transform_saturation_count"] == 0
    assert not torch.equal(direct, compensated)


def test_channel_samples_are_per_state_antithetic_bounded_and_reproducible():
    config = make_config(execution_noise_beta=0.6, execution_noise_samples=4)
    first = ResampledAntitheticChannel(config, torch.device("cpu"))
    repeated = ResampledAntitheticChannel(config, torch.device("cpu"))
    global_state = torch.random.get_rng_state().clone()
    noise = first.sample(3, dtype=torch.float32, device=torch.device("cpu"))
    same = repeated.sample(3, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(noise, same, rtol=0.0, atol=0.0)
    torch.testing.assert_close(noise[:, :2], -noise[:, 2:])
    assert torch.all(noise.abs() <= 0.6)
    assert not torch.equal(noise[0], noise[1])
    assert not torch.equal(
        noise,
        first.sample(3, dtype=torch.float32, device=torch.device("cpu")),
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_channel_canonicalizes_implicit_cuda_device():
    """``cuda`` and a tensor's concrete ``cuda:0`` must denote one device."""

    config = make_config()
    channel = ResampledAntitheticChannel(config, torch.device("cuda"))
    commands = torch.zeros(3, config.action_dim, device="cuda")
    noise = channel.sample(
        commands.shape[0], dtype=commands.dtype, device=commands.device
    )
    assert noise.device == commands.device


def test_channel_checkpoint_restores_exact_next_draw_without_consuming_on_save():
    config = make_config()
    source = ResampledAntitheticChannel(config, torch.device("cpu"))
    source.sample(2, dtype=torch.float32, device=torch.device("cpu"))
    state_before = source.generator.get_state().clone()
    checkpoint = source.checkpoint()
    source.metadata()
    torch.testing.assert_close(source.generator.get_state(), state_before)
    expected = source.sample(2, dtype=torch.float32, device=torch.device("cpu"))
    restored = ResampledAntitheticChannel(config, torch.device("cpu"))
    restored.restore(checkpoint)
    actual = restored.sample(2, dtype=torch.float32, device=torch.device("cpu"))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_positive_beta_rejects_odd_k_but_beta_zero_accepts_k1():
    with pytest.raises(ValueError, match="K=2m"):
        make_config(execution_noise_beta=0.2, execution_noise_samples=3)
    make_config(execution_noise_beta=0.0, execution_noise_samples=1)


class AffineActionCritic(nn.Module):
    def __init__(self, scale=1.0, offset=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor(float(offset)), requires_grad=False)

    def q1_only(self, observations, actions):
        return self.scale * (actions[:, 0] + 0.2 * observations[:, 0]) + self.offset


class QuadraticActionCritic(nn.Module):
    def __init__(self, optimum=0.35):
        super().__init__()
        self.optimum = float(optimum)

    def q1_only(self, observations, actions):
        del observations
        return -(actions[:, 0] - self.optimum).square()


class RecordingQuadraticActionCritic(QuadraticActionCritic):
    def __init__(self, optimum=0.35):
        super().__init__(optimum)
        self.action_inputs = []

    def q1_only(self, observations, actions):
        self.action_inputs.append(actions.detach().clone())
        return super().q1_only(observations, actions)


def test_paired_objective_zero_at_initialization_and_only_adapter_gets_gradient():
    torch.manual_seed(5)
    config = make_config(residual_penalty=0.0)
    adapter = InverseResidualAdapter(config)
    critic = AffineActionCritic()
    normalized = torch.tensor([[1.0, 0.2], [-0.5, 1.0], [0.7, -1.0]])
    baseline = torch.zeros(3, 1)
    channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    critic_before = module_state_sha256({"critic": critic})
    loss, metrics = adapter_objective(
        adapter,
        critic,
        normalized,
        baseline,
        channel,
        torch.tensor(1.0),
    )
    assert metrics["normalized_value_gain"].item() == 0.0
    loss.backward()
    assert critic.scale.grad is None and critic.offset.grad is None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in adapter.parameters()
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    optimizer.step()
    assert module_state_sha256({"critic": critic}) == critic_before


def test_frozen_value_scale_makes_positive_q_rescaling_and_shift_equivalent():
    torch.manual_seed(19)
    config = make_config(residual_penalty=0.3)
    first_adapter = InverseResidualAdapter(config)
    second_adapter = InverseResidualAdapter(config)
    second_adapter.load_state_dict(first_adapter.state_dict())
    observations = torch.tensor([[1.0, 0.1], [-0.4, 0.2], [0.8, -0.7]])
    baseline = torch.zeros(3, 1)
    first_channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    second_channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    first_loss, _ = adapter_objective(
        first_adapter,
        AffineActionCritic(scale=1.0, offset=0.0),
        observations,
        baseline,
        first_channel,
        torch.tensor(2.0),
    )
    second_loss, _ = adapter_objective(
        second_adapter,
        AffineActionCritic(scale=3.0, offset=17.0),
        observations,
        baseline,
        second_channel,
        torch.tensor(6.0),
    )
    first_loss.backward()
    second_loss.backward()
    torch.testing.assert_close(first_loss, second_loss)
    for first_parameter, second_parameter in zip(
        first_adapter.parameters(), second_adapter.parameters()
    ):
        torch.testing.assert_close(first_parameter.grad, second_parameter.grad)


def test_nonlinear_clipped_channel_objective_and_gradient_match_manual_expectation():
    """Exercise E[Q(clip(u+eps))], including the clipping/Jensen distinction."""

    config = make_config(
        execution_noise_beta=0.6,
        execution_noise_samples=4,
        delta_max=0.25,
        residual_penalty=0.0,
    )
    adapter = InverseResidualAdapter(config)
    final = adapter.network[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(0.2)
    observations = torch.tensor(
        [[0.1, -0.2], [0.4, 0.7], [-0.3, 0.5]], dtype=torch.float32
    )
    baseline = torch.tensor([[0.8], [-0.2], [0.1]], dtype=torch.float32)
    value_scale = torch.tensor(1.7)
    critic = QuadraticActionCritic(optimum=0.35)

    reference_channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    noise = reference_channel.sample(
        baseline.shape[0], dtype=baseline.dtype, device=baseline.device
    )
    delta = config.delta_max * np.tanh(0.2)
    adapted_commands = baseline.numpy() + delta
    adapted_actions = np.clip(
        adapted_commands[:, None, :] + noise.numpy(), -1.0, 1.0
    )
    baseline_actions = np.clip(
        baseline.numpy()[:, None, :] + noise.numpy(), -1.0, 1.0
    )
    adapted_q = -np.square(adapted_actions[..., 0] - critic.optimum)
    baseline_q = -np.square(baseline_actions[..., 0] - critic.optimum)
    expected_gain = float((adapted_q.mean() - baseline_q.mean()) / value_scale.item())
    expected_loss = -expected_gain

    channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    loss, metrics = adapter_objective(
        adapter,
        critic,
        observations,
        baseline,
        channel,
        value_scale,
    )
    assert float(metrics["normalized_value_gain"]) == pytest.approx(
        expected_gain, abs=1e-6
    )
    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)

    # A plug-in Q(E[a]) is deliberately different for this nonlinear critic.
    plugin_q = -np.square(adapted_actions[..., 0].mean(axis=1) - critic.optimum)
    assert abs(float(plugin_q.mean() - adapted_q.mean())) > 1e-3

    loss.backward()
    requested = adapted_commands[:, None, :] + noise.numpy()
    unclipped = ((requested > -1.0) & (requested < 1.0))[..., 0]
    dq_dcommand = -2.0 * (adapted_actions[..., 0] - critic.optimum) * unclipped
    dcommand_dbias = config.delta_max * (1.0 - np.tanh(0.2) ** 2)
    expected_bias_gradient = -float(dq_dcommand.mean()) * dcommand_dbias
    expected_bias_gradient /= value_scale.item()
    assert float(final.bias.grad[0]) == pytest.approx(
        expected_bias_gradient, abs=1e-6
    )


def test_q1_at_channel_mean_repeats_exact_mean_rows_and_consumes_no_channel_rng():
    config = make_config(
        value_estimator="q1_at_channel_mean",
        execution_noise_beta=0.6,
        execution_noise_samples=4,
        residual_penalty=0.0,
    )
    adapter = InverseResidualAdapter(config)
    with torch.no_grad():
        adapter.network[-1].bias.fill_(0.2)
    observations = torch.tensor([[0.1, -0.2], [0.4, 0.7], [-0.3, 0.5]])
    baseline = torch.tensor([[0.8], [-0.2], [0.1]])
    critic = RecordingQuadraticActionCritic()
    channel = ResampledAntitheticChannel(config, torch.device("cpu"))
    generator_before = channel.generator.get_state().clone()

    loss, _ = adapter_objective(
        adapter,
        critic,
        observations,
        baseline,
        channel,
        torch.tensor(1.0),
    )
    assert len(critic.action_inputs) == 2
    assert all(rows.shape == (12, 1) for rows in critic.action_inputs)
    command, _, _ = adapter.compose_command(observations, baseline)
    expected_adapted = expected_clipped_uniform_action_torch(
        command, -1.0, 1.0, config.execution_noise_beta
    )
    expected_baseline = expected_clipped_uniform_action_torch(
        baseline, -1.0, 1.0, config.execution_noise_beta
    )
    for recorded, expected in zip(
        critic.action_inputs, (expected_adapted, expected_baseline)
    ):
        reshaped = recorded.reshape(3, config.execution_noise_samples, 1)
        torch.testing.assert_close(
            reshaped,
            expected[:, None, :].repeat(1, config.execution_noise_samples, 1),
        )
    assert channel.draw_calls == 0
    assert channel.half_vectors_drawn == 0
    torch.testing.assert_close(channel.generator.get_state(), generator_before)
    metadata = value_estimator_metadata(config)
    assert metadata["channel_rng_consumed"] is False
    assert metadata["deterministic_channel_mean_repeated_for_equal_q1_rows"] is True
    loss.backward()
    assert adapter.network[-1].bias.grad is not None


def test_value_scale_metadata_names_declared_identity_baseline_and_estimator():
    torch.manual_seed(31)
    base = make_base_agent()
    config = make_config(
        baseline_transform="identity",
        value_estimator="q1_at_channel_mean",
    )
    normalized = torch.randn(9, 2)
    commands = torch.linspace(-0.8, 0.8, 9)[:, None]
    _, audit = calibrate_value_scale(base, normalized, commands, config, 7, 101)
    assert audit["method"] == (
        "frozen_std_of_declared_baseline_value_estimator_q1"
    )
    assert audit["baseline_transform"] == "identity"
    assert audit["value_estimator"]["name"] == "q1_at_channel_mean"
    assert audit["channel_seed"] is None
    assert audit["channel_sample_calls"] == 0
    assert audit["q1_input_rows"] == 7 * config.execution_noise_samples


def test_initial_controller_command_matches_uniform_mean_inverse_evaluation_exactly():
    torch.manual_seed(23)
    base = make_base_agent()
    config = make_config(execution_noise_beta=1.0)
    adapter = InverseResidualAdapter(config)
    controller = InverseResidualController(base, adapter)
    observation = np.asarray([0.2, -0.4], dtype=np.float32)
    low = np.asarray([-1.0], dtype=np.float32)
    high = np.asarray([1.0], dtype=np.float32)
    raw = base.act(observation, low, high)
    expected, _ = prepare_evaluation_command(
        raw,
        low,
        high,
        command_transform="uniform_mean_inverse",
        actuator_noise_beta=1.0,
    )
    actual, details = controller.command(observation, low, high)
    np.testing.assert_array_equal(actual, expected)
    assert details["proposed_residual"] == [0.0]


def test_base_checkpoint_loader_restores_all_four_networks_bitwise_and_freezes(tmp_path):
    torch.manual_seed(29)
    original = make_base_agent()
    path = tmp_path / "base.pt"
    torch.save({"step": 17, "agent": original.checkpoint()}, path)
    loaded, _, step = load_frozen_physical_agent(path, torch.device("cpu"))
    assert step == 17
    for name in ("actor", "critic", "actor_target", "critic_target"):
        expected = getattr(original, name).state_dict()
        actual = getattr(loaded, name).state_dict()
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
        assert all(not parameter.requires_grad for parameter in getattr(loaded, name).parameters())

    invalid = make_base_agent(action_pairing="executed_commanded")
    invalid_path = tmp_path / "ambiguous_actor.pt"
    torch.save({"agent": invalid.checkpoint()}, invalid_path)
    with pytest.raises(ValueError, match="executed/executed base"):
        load_frozen_physical_agent(invalid_path, torch.device("cpu"))


def test_episode_level_observation_split_is_disjoint_deterministic_and_hashed(tmp_path):
    dataset = tmp_path / "episodic_flat.hdf5"
    observations = np.arange(24, dtype=np.float32).reshape(12, 2)
    terminals = np.zeros(12, dtype=np.bool_)
    timeouts = np.zeros(12, dtype=np.bool_)
    terminals[[2, 6]] = True
    timeouts[11] = True
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = observations
        handle["terminals"] = terminals
        handle["timeouts"] = timeouts

    first = build_observation_split(dataset, None, 0.3, 1234)
    second = build_observation_split(dataset, None, 0.3, 1234)
    first_observations, train_indices, audit_indices, metadata = first
    np.testing.assert_array_equal(first_observations, observations)
    np.testing.assert_array_equal(train_indices, second[1])
    np.testing.assert_array_equal(audit_indices, second[2])
    assert metadata == second[3]
    assert metadata["method"] == "whole_episode_permutation_to_state_count"
    assert metadata["boundary_source"] == "terminals+timeouts"
    assert np.intersect1d(train_indices, audit_indices).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train_indices, audit_indices))), np.arange(12)
    )
    for episode_indices in (np.arange(0, 3), np.arange(3, 7), np.arange(7, 12)):
        in_audit = np.isin(episode_indices, audit_indices)
        assert bool(in_audit.all()) or bool((~in_audit).all())
    assert len(metadata["train_indices_sha256"]) == 64
    assert len(metadata["audit_indices_sha256"]) == 64


def test_observation_split_explicitly_falls_back_to_fixed_indices(tmp_path):
    dataset = tmp_path / "flat_without_boundaries.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = np.arange(20, dtype=np.float32).reshape(10, 2)
    _, train_indices, audit_indices, metadata = build_observation_split(
        dataset, None, 0.2, 99
    )
    assert metadata["method"] == "fixed_index_permutation_no_episode_boundaries"
    assert train_indices.size == 8
    assert audit_indices.size == 2


def test_training_and_value_scale_use_only_recorded_train_partition(tmp_path):
    dataset = tmp_path / "training_states.hdf5"
    observations = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(12, 2)
    terminals = np.zeros(12, dtype=np.bool_)
    terminals[[2, 6, 11]] = True
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = observations
        handle["terminals"] = terminals
    base = make_base_agent()
    base_checkpoint = tmp_path / "base.pt"
    torch.save({"step": 0, "agent": base.checkpoint()}, base_checkpoint)
    output_dir = tmp_path / "adapter_run"
    args = build_train_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--base-checkpoint",
            str(base_checkpoint),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--updates",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "8",
            "--depth",
            "1",
            "--execution-noise-beta",
            "0.4",
            "--execution-noise-samples",
            "2",
            "--audit-fraction",
            "0.3",
            "--split-seed",
            "1234",
            "--scale-calibration-observations",
            "100",
            "--log-period",
            "1",
            "--checkpoint-period",
            "1",
        ]
    )
    summary = run_adapter_training(args)
    config_payload = json.loads((output_dir / "config.json").read_text("utf-8"))
    split = config_payload["observation_split"]
    train_count = split["train_observation_count"]
    assert split["audit_observation_count"] > 0
    assert config_payload["baseline_precomputation"]["observation_count"] == train_count
    assert config_payload["value_scale_calibration"]["observation_count"] == train_count
    assert summary["baseline_precomputation"]["observation_count"] == train_count

    audit_output = tmp_path / "heldout_audit.json"
    audit_args = build_audit_parser().parse_args(
        [
            "--adapter-checkpoint",
            str(output_dir / "latest.pt"),
            "--dataset",
            str(dataset),
            "--output",
            str(audit_output),
            "--device",
            "cpu",
            "--audit-noise-samples",
            "2",
            "--audit-action-noise-beta",
            "1.25",
            "--batch-size",
            "4",
        ]
    )
    audit_result = run_adapter_audit(audit_args)
    assert audit_result["dataset"]["selection"] == (
        "exact_recorded_held_out_partition"
    )
    assert audit_result["dataset"]["selected_observations"] == split[
        "audit_observation_count"
    ]
    assert audit_result["dataset"]["selected_indices_sha256"] == split[
        "audit_indices_sha256"
    ]
    assert "adapted_minus_baseline_q2" in audit_result["metrics"]
    assert "median" in audit_result["metrics"]["adapted_minus_baseline_q2"]
    assert "baseline_absolute_twin_disagreement" in audit_result["metrics"]
    assert "adapted_absolute_twin_disagreement" in audit_result["metrics"]
    assert audit_result["audit_channel"]["model_or_calibration_beta"] == pytest.approx(0.4)
    assert audit_result["audit_channel"]["audit_action_noise_beta"] == pytest.approx(1.25)
    assert audit_result["audit_channel"]["beta_mismatch"] is True
    assert set(audit_result["implementation"]) == {
        "audit_sha256",
        "evaluate_sha256",
        "core_sha256",
        "train_sha256",
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        run_adapter_audit(audit_args)
    with pytest.raises(FileExistsError, match="nonempty"):
        run_adapter_training(args)


def _tiny_training_cli(dataset, base_checkpoint, output_dir, updates, *extra):
    return build_train_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--base-checkpoint",
            str(base_checkpoint),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--train-seed",
            "37",
            "--updates",
            str(updates),
            "--batch-size",
            "4",
            "--hidden-dim",
            "8",
            "--depth",
            "1",
            "--learning-rate",
            "0.001",
            "--execution-noise-beta",
            "0.4",
            "--execution-noise-samples",
            "2",
            "--execution-noise-seed",
            "71",
            "--audit-fraction",
            "0.3",
            "--split-seed",
            "1234",
            "--scale-calibration-observations",
            "8",
            "--log-period",
            "1",
            "--checkpoint-period",
            "2",
            "--torch-threads",
            "1",
            *extra,
        ]
    )


def _make_tiny_training_inputs(tmp_path):
    dataset = tmp_path / "resume_states.hdf5"
    observations = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2)
    terminals = np.zeros(16, dtype=np.bool_)
    terminals[[3, 7, 11, 15]] = True
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = observations
        handle["terminals"] = terminals
    torch.manual_seed(911)
    base = make_base_agent()
    base_checkpoint = tmp_path / "resume_base.pt"
    torch.save({"step": 13, "agent": base.checkpoint()}, base_checkpoint)
    return dataset, base_checkpoint


def _assert_nested_exact(first, second):
    assert type(first) is type(second)
    if torch.is_tensor(first):
        assert torch.equal(first.cpu(), second.cpu())
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for first_item, second_item in zip(first, second):
            _assert_nested_exact(first_item, second_item)
    else:
        assert first == second


@pytest.mark.parametrize(
    "value_estimator", ("sampled_expected_q1", "q1_at_channel_mean")
)
def test_exact_resume_matches_continuous_training_and_preserves_progress(
    tmp_path, value_estimator
):
    dataset, base_checkpoint = _make_tiny_training_inputs(tmp_path)
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"

    run_adapter_training(
        _tiny_training_cli(
            dataset,
            base_checkpoint,
            continuous_dir,
            4,
            "--value-estimator",
            value_estimator,
        )
    )
    run_adapter_training(
        _tiny_training_cli(
            dataset,
            base_checkpoint,
            resumed_dir,
            2,
            "--value-estimator",
            value_estimator,
        )
    )
    progress_prefix = (resumed_dir / "progress.jsonl").read_bytes()
    run_adapter_training(
        _tiny_training_cli(
            dataset,
            base_checkpoint,
            resumed_dir,
            4,
            "--resume",
            "--value-estimator",
            value_estimator,
        )
    )

    continuous = torch.load(continuous_dir / "latest.pt", map_location="cpu")
    resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu")
    assert continuous["step"] == resumed["step"] == 4
    assert resumed["target_total_updates"] == 4
    assert resumed["resume_count"] == 1
    assert resumed["exact_training_resume_implemented"] is True
    assert resumed["adapter_config"]["value_estimator"] == value_estimator
    assert resumed["resume_signature"]["adapter"]["value_estimator"] == (
        value_estimator
    )
    assert resumed["resume_signature"]["optimization"]["value_estimator"] == (
        value_estimator
    )
    expected_channel_calls = 4 if value_estimator == "sampled_expected_q1" else 0
    assert resumed["channel"]["draw_calls"] == expected_channel_calls
    for field in (
        "adapter",
        "optimizer",
        "channel",
        "index_generator_state",
        "global_rng_state",
    ):
        _assert_nested_exact(continuous[field], resumed[field])

    resumed_progress = (resumed_dir / "progress.jsonl").read_bytes()
    assert resumed_progress.startswith(progress_prefix)
    events = [
        json.loads(line)
        for line in resumed_progress.decode("utf-8").splitlines()
        if line.strip()
    ]
    assert [event["step"] for event in events if event["event"] == "train"] == [
        1,
        2,
        3,
        4,
    ]
    resume_events = [event for event in events if event["event"] == "resume"]
    assert len(resume_events) == 1
    assert resume_events[0]["from_step"] == 2
    assert resume_events[0]["target_total_updates"] == 4


@pytest.mark.parametrize(
    "mismatch_arguments",
    (
        ("--residual-penalty", "0.75"),
        ("--value-estimator", "q1_at_channel_mean"),
    ),
)
def test_resume_rejects_mismatched_config_without_mutating_run(
    tmp_path, mismatch_arguments
):
    dataset, base_checkpoint = _make_tiny_training_inputs(tmp_path)
    output_dir = tmp_path / ("mismatch_" + mismatch_arguments[0].lstrip("-"))
    run_adapter_training(_tiny_training_cli(dataset, base_checkpoint, output_dir, 2))
    progress_before = (output_dir / "progress.jsonl").read_bytes()
    checkpoint_before = (output_dir / "latest.pt").read_bytes()
    config_before = (output_dir / "config.json").read_bytes()

    args = _tiny_training_cli(
        dataset,
        base_checkpoint,
        output_dir,
        4,
        "--resume",
        *mismatch_arguments,
    )
    with pytest.raises(ValueError, match="resume signature mismatch"):
        run_adapter_training(args)
    assert (output_dir / "progress.jsonl").read_bytes() == progress_before
    assert (output_dir / "latest.pt").read_bytes() == checkpoint_before
    assert (output_dir / "config.json").read_bytes() == config_before


def test_evaluator_loads_legacy_checkpoint_without_value_estimator(tmp_path):
    dataset, base_checkpoint = _make_tiny_training_inputs(tmp_path)
    output_dir = tmp_path / "modern"
    run_adapter_training(_tiny_training_cli(dataset, base_checkpoint, output_dir, 1))
    payload = torch.load(output_dir / "latest.pt", map_location="cpu")
    payload["format"] = "inverse_residual_adapter_v1"
    payload["adapter_config"].pop("value_estimator")
    legacy_path = tmp_path / "legacy_pre_split.pt"
    torch.save(payload, legacy_path)
    controller, _ = load_controller(legacy_path, torch.device("cpu"))
    assert controller.adapter.config.value_estimator == "sampled_expected_q1"


def _valid_calibration_payload():
    commands_hash = "1" * 64
    executed_hash = "2" * 64
    input_provenance = {
        "schema": "paired_uniform_clip_channel_v1",
        "channel": "iid_uniform_additive_then_clip",
        "contains_state_or_transition_fields": False,
        "pair_count": 512,
        "action_dim": 1,
        "action_low": -1.0,
        "action_high": 1.0,
        "commands_sha256": commands_hash,
        "executed_sha256": executed_hash,
    }
    return {
        "status": "complete",
        "estimator": "censored_uniform_plus_clip_mle",
        "estimator_uses_provenance_beta": False,
        "pair_archive_path": "/relocatable/pairs.npz",
        "pair_archive_sha256": "3" * 64,
        "pair_count": 512,
        "action_dim": 1,
        "commands_sha256": commands_hash,
        "executed_sha256": executed_hash,
        "action_low": -1.0,
        "action_high": 1.0,
        "estimate": {"beta_mle": 1.249},
        "uncertainty": {"beta_interval": [1.24, 1.25]},
        "input_provenance": input_provenance,
        "calibration_script_sha256": "4" * 64,
    }


def test_channel_calibration_schema_is_strict_and_estimate_is_authoritative(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_valid_calibration_payload()), encoding="utf-8")
    beta, provenance = resolve_channel_calibration(path, None)
    assert beta == pytest.approx(1.249)
    assert provenance["pair_count"] == 512
    assert provenance["source"] == "censored_uniform_plus_clip_pair_calibration"
    with pytest.raises(ValueError, match="authoritative"):
        resolve_channel_calibration(path, 1.25)


@pytest.mark.parametrize(
    "corruption",
    ("status", "estimator", "state_fields", "pair_count", "digest"),
)
def test_channel_calibration_rejects_invalid_or_inconsistent_schema(
    tmp_path, corruption
):
    payload = copy.deepcopy(_valid_calibration_payload())
    if corruption == "status":
        payload["status"] = "dry_run"
    elif corruption == "estimator":
        payload["estimator"] = "endpoint_residual"
    elif corruption == "state_fields":
        payload["input_provenance"]["contains_state_or_transition_fields"] = True
    elif corruption == "pair_count":
        payload["input_provenance"]["pair_count"] = 511
    elif corruption == "digest":
        payload["commands_sha256"] = "not-a-digest"
    path = tmp_path / f"bad_{corruption}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_channel_calibration(path, None)


def test_output_guards_allow_empty_directory_but_reject_prior_artifacts(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    assert prepare_fresh_output_dir(empty) == empty.resolve()
    (empty / "progress.jsonl").write_text("prior", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        prepare_fresh_output_dir(empty)

    result = tmp_path / "evaluation.json"
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        require_new_output_file(result)
    result.unlink()
    stale_temporary = result.with_suffix(result.suffix + ".tmp")
    stale_temporary.write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        require_new_output_file(result)


def test_observation_loader_and_cli_surface(tmp_path):
    dataset = tmp_path / "states.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    observations = load_observations(dataset, max_observations=4)
    np.testing.assert_array_equal(
        observations, np.arange(8, dtype=np.float32).reshape(4, 2)
    )
    train_args = build_train_parser().parse_args(
        [
            "--dataset",
            "data.hdf5",
            "--base-checkpoint",
            "base.pt",
            "--output-dir",
            "output",
            "--execution-noise-beta",
            "1.0",
            "--updates",
            "2000",
            "--alpha",
            "0.4",
            "--residual-penalty",
            "2.0",
            "--delta-max",
            "0.15",
        ]
    )
    assert train_args.updates == 2000
    assert train_args.alpha == pytest.approx(0.4)
    assert train_args.residual_penalty == pytest.approx(2.0)
    assert train_args.delta_max == pytest.approx(0.15)
    assert train_args.baseline_transform == "inverse"
    assert train_args.value_estimator == "sampled_expected_q1"
    assert train_args.audit_fraction == pytest.approx(0.1)
    assert train_args.split_seed == 424242
    direct_args = build_train_parser().parse_args(
        [
            "--dataset",
            "data.hdf5",
            "--base-checkpoint",
            "base.pt",
            "--output-dir",
            "output",
            "--execution-noise-beta",
            "1.0",
            "--baseline-transform",
            "identity",
        ]
    )
    assert direct_args.baseline_transform == "identity"
    mean_args = build_train_parser().parse_args(
        [
            "--dataset",
            "data.hdf5",
            "--base-checkpoint",
            "base.pt",
            "--output-dir",
            "output",
            "--execution-noise-beta",
            "1.0",
            "--value-estimator",
            "q1_at_channel_mean",
        ]
    )
    assert mean_args.value_estimator == "q1_at_channel_mean"
    eval_args = build_eval_parser().parse_args(
        ["--adapter-checkpoint", "adapter.pt", "--output", "result.json"]
    )
    assert eval_args.eval_episodes == 50
    audit_args = build_audit_parser().parse_args(
        [
            "--adapter-checkpoint",
            "adapter.pt",
            "--dataset",
            "data.hdf5",
            "--output",
            "audit.json",
        ]
    )
    assert audit_args.audit_noise_samples == 64
    assert audit_args.audit_action_noise_beta is None
