import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cc_fqe_rpi_core import (  # noqa: E402
    CCFQERPIAgent,
    CCFQERPIConfig,
    audit_policy_k64,
    canonical_torch_device,
    source_support_expansion,
)
from evaluate_cc_fqe_rpi import (  # noqa: E402
    load_agent as load_evaluation_agent,
    resolve_evaluation_beta,
)
from td3bc_core import DeterministicPolicy, TD3BCConfig, TwinCritic  # noqa: E402
from train_cc_fqe_rpi import (  # noqa: E402
    build_parser,
    restore_training_checkpoint,
    run,
    sample_batch,
    sha256_file,
    training_checkpoint,
)


def make_agent(**overrides):
    arguments = dict(
        observation_dim=1,
        action_dim=1,
        base_hidden_dim=2,
        base_depth=1,
        adapter_hidden_dim=2,
        adapter_depth=1,
        max_action=1.0,
        discount=0.9,
        tau=0.1,
        source_beta=0.2,
        target_beta=0.4,
        continuation_mode="target",
        train_channel_samples=4,
        audit_channel_samples=64,
        continuation_noise_seed=11,
        actor_noise_seed=12,
        audit_noise_seed=13,
        fqe_updates=2,
        adapter_updates=2,
        critic_learning_rate=1e-3,
        adapter_learning_rate=1e-3,
    )
    arguments.update(overrides)
    config = CCFQERPIConfig(**arguments)
    torch.manual_seed(7)
    actor = DeterministicPolicy(1, 1, 2, 1, 1.0)
    critic = TwinCritic(1, 1, 2, 1)
    return CCFQERPIAgent(
        config,
        torch.device("cpu"),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        actor.state_dict(),
        critic.state_dict(),
    )


def make_fqe_batch(batch_size=3):
    return {
        "observations": torch.linspace(-0.3, 0.3, batch_size).reshape(-1, 1),
        "executed_actions": torch.linspace(-0.8, 0.7, batch_size).reshape(-1, 1),
        "next_observations": torch.linspace(0.2, 0.5, batch_size).reshape(-1, 1),
        "next_inverse_commands": torch.linspace(-0.1, 0.1, batch_size).reshape(-1, 1),
        "rewards": torch.linspace(0.1, 0.3, batch_size),
        "terminals": torch.zeros(batch_size),
    }


def test_rollout_beta_is_independent_of_calibrated_model_beta():
    assert resolve_evaluation_beta(1.2498949, 1.25) == pytest.approx(1.25)
    assert resolve_evaluation_beta(1.2498949, None) == pytest.approx(1.2498949)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_evaluation_beta(1.0, -0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_implicit_cuda_is_canonicalized_before_channel_device_check():
    resolved = canonical_torch_device("cuda")
    assert resolved.index == torch.cuda.current_device()
    agent = make_agent()
    # Exercise the same defensive comparison without constructing a second
    # full GPU agent in the unit test.
    channel_cls = type(agent.continuation_channel)
    channel = channel_cls(
        action_dim=1,
        sample_count=4,
        beta=0.4,
        max_action=1.0,
        seed=11,
        device=torch.device("cuda"),
    )
    samples = channel.sample(
        2,
        dtype=torch.float32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    assert samples.device == resolved


def set_absolute_action_critic(critic):
    """Configure both depth-one heads as Q(s,a)=abs(a)."""

    with torch.no_grad():
        for head in (critic.q1, critic.q2):
            head[0].weight.zero_()
            head[0].bias.zero_()
            head[0].weight[0, 1] = 1.0
            head[0].weight[1, 1] = -1.0
            head[-1].weight.fill_(1.0)
            head[-1].bias.zero_()


def assert_module_equal(left, right):
    for key, value in left.state_dict().items():
        torch.testing.assert_close(
            value, right.state_dict()[key], rtol=0.0, atol=0.0
        )


def test_default_protocol_is_fixed_20k_5k_k4_k64_and_target_control_switch():
    args = build_parser().parse_args(
        ["--dataset", "data.hdf5", "--base-checkpoint", "base.pt", "--output-dir", "out"]
    )
    assert args.fqe_updates == 20_000
    assert args.adapter_updates == 5_000
    assert args.train_channel_samples == 4
    assert args.audit_channel_samples == 64
    assert args.source_beta == pytest.approx(1.0)
    assert args.target_beta == pytest.approx(1.2498949)
    assert args.continuation_mode == "target"
    target = CCFQERPIConfig(observation_dim=2, action_dim=1)
    source = CCFQERPIConfig(
        observation_dim=2, action_dim=1, continuation_mode="source"
    )
    assert target.continuation_beta == pytest.approx(1.2498949)
    assert source.continuation_beta == pytest.approx(1.0)


def test_target_vs_source_continuation_changes_beta_not_compute():
    target_agent = make_agent(continuation_mode="target")
    source_agent = make_agent(continuation_mode="source")
    set_absolute_action_critic(target_agent.critic_target)
    set_absolute_action_critic(source_agent.critic_target)
    batch = make_fqe_batch(batch_size=4)
    batch["next_inverse_commands"].zero_()
    batch["rewards"].zero_()
    target, _ = target_agent.compute_fqe_target(batch)
    source, _ = source_agent.compute_fqe_target(batch)
    # Identical uniform variates are scaled by beta; clipping is inactive here.
    torch.testing.assert_close(target, 2.0 * source, rtol=1e-6, atol=1e-7)
    assert target_agent.continuation_channel.draw_calls == 1
    assert source_agent.continuation_channel.draw_calls == 1
    assert (
        target_agent.continuation_channel.half_vectors_drawn
        == source_agent.continuation_channel.half_vectors_drawn
        == 8
    )
    assert target_agent.metadata()["hubl_mc_mixing"] is False
    assert target_agent.metadata()["td3_target_policy_smoothing"] is False


def test_fqe_uses_executed_action_and_ignores_hubl_and_command_fields():
    first = make_agent()
    second = make_agent()
    batch = make_fqe_batch()
    first_batch = {**batch, "logged_commands": torch.full((3, 1), -99.0)}
    first_batch.update(
        {"heuristic_next": torch.full((3,), 1e6), "lambdas": torch.ones(3)}
    )
    second_batch = {**batch, "logged_commands": torch.full((3, 1), 99.0)}
    second_batch.update(
        {"heuristic_next": torch.full((3,), -1e6), "lambdas": torch.zeros(3)}
    )
    metrics_first = first.fqe_update(first_batch)
    metrics_second = second.fqe_update(second_batch)
    assert metrics_first == metrics_second
    assert_module_equal(first.critic, second.critic)
    assert metrics_first["executed_action_regression"] == 1.0
    assert metrics_first["hubl_mc_mixing"] == 0.0
    assert metrics_first["td3_target_smoothing"] == 0.0


def test_terminal_target_and_min_of_expectations_are_exact():
    agent = make_agent()
    with torch.no_grad():
        for parameter in agent.critic_target.parameters():
            parameter.zero_()
        agent.critic_target.q1[-1].bias.fill_(2.0)
        agent.critic_target.q2[-1].bias.fill_(5.0)
    batch = make_fqe_batch(batch_size=2)
    batch["rewards"].fill_(1.0)
    batch["terminals"][1] = 1.0
    noise = torch.tensor([[[-0.3], [0.1], [0.3], [-0.1]]] * 2)
    target, diagnostics = agent.compute_fqe_target(batch, noise=noise)
    assert target[0].item() == pytest.approx(1.0 + 0.9 * 2.0)
    assert target[1].item() == pytest.approx(1.0)
    torch.testing.assert_close(
        diagnostics["target_continuation"], torch.full((2,), 2.0)
    )


def test_support_expansion_penalty_detects_target_only_action_region():
    command = torch.tensor([[0.8], [0.0]])
    logged = torch.tensor([[0.8], [0.0]])
    penalty, violation = source_support_expansion(
        command,
        logged,
        source_beta=0.2,
        target_beta=0.4,
        max_action=1.0,
    )
    assert penalty[0] > 0.0 and penalty[1] > 0.0
    torch.testing.assert_close(violation, torch.ones_like(violation))
    zero, no_violation = source_support_expansion(
        command,
        logged,
        source_beta=0.4,
        target_beta=0.2,
        max_action=1.0,
    )
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(no_violation, torch.zeros_like(no_violation))


def test_adapter_common_random_numbers_zero_gain_and_frozen_critic_gradients():
    agent = make_agent(fqe_updates=1, adapter_updates=1)
    agent.fqe_steps = 1
    agent.enter_adapter_stage(1.0)
    batch = {
        "observations": torch.tensor([[-0.2], [0.3]]),
        "inverse_commands": torch.tensor([[0.1], [-0.1]]),
        "logged_commands": torch.zeros(2, 1),
    }
    noise = torch.tensor(
        [[[-0.2], [0.3], [0.2], [-0.3]], [[0.1], [-0.4], [-0.1], [0.4]]]
    )
    loss, diagnostics = agent.adapter_objective(batch, noise=noise)
    # Zero final layer means the adapted and inverse commands are bit-identical;
    # common random numbers therefore make finite-K gain exactly zero.
    torch.testing.assert_close(
        diagnostics["value_gain"], torch.zeros(2), rtol=0.0, atol=0.0
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in agent.critic.parameters())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.adapter.parameters()
    )


def test_training_checkpoint_exactly_restores_batch_channel_and_optimizers():
    tensor_data = {
        "observations": torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 10,
        "executed_actions": torch.linspace(-0.5, 0.5, 8).reshape(-1, 1),
        "next_observations": torch.arange(1, 9, dtype=torch.float32).reshape(-1, 1) / 10,
        "next_inverse_commands": torch.zeros(8, 1),
        "rewards": torch.linspace(0.0, 0.7, 8),
        "terminals": torch.zeros(8),
        "inverse_commands": torch.zeros(8, 1),
        "logged_commands": torch.zeros(8, 1),
    }
    fqe_keys = (
        "observations",
        "executed_actions",
        "next_observations",
        "next_inverse_commands",
        "rewards",
        "terminals",
    )
    adapter_keys = ("observations", "inverse_commands", "logged_commands")
    agent = make_agent()
    fqe_rng = torch.Generator().manual_seed(101)
    adapter_rng = torch.Generator().manual_seed(102)
    agent.fqe_update(
        sample_batch(
            tensor_data,
            fqe_keys,
            batch_size=4,
            generator=fqe_rng,
            device=torch.device("cpu"),
        )
    )
    saved = training_checkpoint(
        agent,
        fqe_batch_generator=fqe_rng,
        adapter_batch_generator=adapter_rng,
        provenance_fingerprint="fixed",
        provenance={"test": True},
    )
    assert saved["provenance"] == {"test": True}
    continued_batch = sample_batch(
        tensor_data,
        fqe_keys,
        batch_size=4,
        generator=fqe_rng,
        device=torch.device("cpu"),
    )
    continued_metrics = agent.fqe_update(continued_batch)

    restored = make_agent()
    restored_fqe_rng = torch.Generator().manual_seed(999)
    restored_adapter_rng = torch.Generator().manual_seed(998)
    restore_training_checkpoint(
        saved,
        restored,
        fqe_batch_generator=restored_fqe_rng,
        adapter_batch_generator=restored_adapter_rng,
        provenance_fingerprint="fixed",
    )
    restored_batch = sample_batch(
        tensor_data,
        fqe_keys,
        batch_size=4,
        generator=restored_fqe_rng,
        device=torch.device("cpu"),
    )
    for key in fqe_keys:
        torch.testing.assert_close(restored_batch[key], continued_batch[key])
    restored_metrics = restored.fqe_update(restored_batch)
    assert restored_metrics == continued_metrics
    assert_module_equal(restored.critic, agent.critic)
    assert_module_equal(restored.critic_target, agent.critic_target)

    agent.enter_adapter_stage(1.0)
    restored.enter_adapter_stage(1.0)
    first_adapter_batch = sample_batch(
        tensor_data,
        adapter_keys,
        batch_size=4,
        generator=adapter_rng,
        device=torch.device("cpu"),
    )
    restored_first_batch = sample_batch(
        tensor_data,
        adapter_keys,
        batch_size=4,
        generator=restored_adapter_rng,
        device=torch.device("cpu"),
    )
    for key in adapter_keys:
        torch.testing.assert_close(first_adapter_batch[key], restored_first_batch[key])
    first_metrics = agent.adapter_update(first_adapter_batch)
    restored_first_metrics = restored.adapter_update(restored_first_batch)
    assert first_metrics == restored_first_metrics
    saved_adapter = training_checkpoint(
        agent,
        fqe_batch_generator=fqe_rng,
        adapter_batch_generator=adapter_rng,
        provenance_fingerprint="fixed",
        provenance={"test": True},
    )
    next_adapter_batch = sample_batch(
        tensor_data,
        adapter_keys,
        batch_size=4,
        generator=adapter_rng,
        device=torch.device("cpu"),
    )
    next_metrics = agent.adapter_update(next_adapter_batch)

    resumed_adapter = make_agent()
    resumed_fqe_rng = torch.Generator()
    resumed_adapter_rng = torch.Generator()
    restore_training_checkpoint(
        saved_adapter,
        resumed_adapter,
        fqe_batch_generator=resumed_fqe_rng,
        adapter_batch_generator=resumed_adapter_rng,
        provenance_fingerprint="fixed",
    )
    resumed_batch = sample_batch(
        tensor_data,
        adapter_keys,
        batch_size=4,
        generator=resumed_adapter_rng,
        device=torch.device("cpu"),
    )
    for key in adapter_keys:
        torch.testing.assert_close(resumed_batch[key], next_adapter_batch[key])
    resumed_metrics = resumed_adapter.adapter_update(resumed_batch)
    assert resumed_metrics == next_metrics
    assert_module_equal(resumed_adapter.adapter, agent.adapter)


def test_k64_audit_is_deterministic_and_does_not_advance_training_rng():
    agent = make_agent(fqe_updates=1, adapter_updates=1, audit_channel_samples=64)
    agent.fqe_steps = 1
    agent.enter_adapter_stage(1.0)
    observations = torch.linspace(-0.5, 0.5, 7).reshape(-1, 1)
    inverse = torch.zeros(7, 1)
    logged = torch.linspace(-0.2, 0.2, 7).reshape(-1, 1)
    continuation_before = agent.continuation_channel.generator.get_state().clone()
    actor_before = agent.actor_channel.generator.get_state().clone()
    first, first_metadata = audit_policy_k64(
        agent, observations, inverse, logged, [6, 1, 4, 0], batch_size=3
    )
    second, second_metadata = audit_policy_k64(
        agent, observations, inverse, logged, [6, 1, 4, 0], batch_size=3
    )
    assert first_metadata["sample_count"] == 64
    assert first_metadata == second_metadata
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    torch.testing.assert_close(
        agent.continuation_channel.generator.get_state(), continuation_before
    )
    torch.testing.assert_close(agent.actor_channel.generator.get_state(), actor_before)


def test_completed_checkpoint_is_standalone_for_evaluation(tmp_path):
    agent = make_agent(fqe_updates=1, adapter_updates=1)
    agent.fqe_steps = 1
    agent.enter_adapter_stage(1.0)
    agent.adapter_steps = 1
    fqe_rng = torch.Generator().manual_seed(31)
    adapter_rng = torch.Generator().manual_seed(32)
    payload = training_checkpoint(
        agent,
        fqe_batch_generator=fqe_rng,
        adapter_batch_generator=adapter_rng,
        provenance_fingerprint="standalone",
        provenance={"base_checkpoint": {"sha256": "test"}},
    )
    path = tmp_path / "complete.pt"
    torch.save(payload, path)
    restored, provenance = load_evaluation_agent(path, torch.device("cpu"))
    assert restored.stage == "complete"
    assert provenance["provenance_fingerprint"] == "standalone"
    command, details = restored.command(np.asarray([0.2], dtype=np.float32), use_residual=False)
    np.testing.assert_array_equal(
        command, np.asarray(details["inverse_baseline_command"], dtype=np.float32)
    )


def write_tiny_dataset(path):
    with h5py.File(path, "w") as handle:
        for episode_id, reward in enumerate((1.0, 2.0)):
            group = handle.create_group(f"episode_{episode_id}")
            group.create_dataset(
                "observations",
                data=np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32),
            )
            group.create_dataset(
                "actions", data=np.asarray([[0.0], [0.1]], dtype=np.float32)
            )
            group.create_dataset(
                "clean_policy_actions",
                data=np.asarray([[0.0], [0.05]], dtype=np.float32),
            )
            group.create_dataset(
                "rewards", data=np.asarray([reward, reward], dtype=np.float32)
            )
            group.create_dataset(
                "terminations", data=np.asarray([False, True], dtype=np.bool_)
            )
            group.create_dataset(
                "truncations", data=np.asarray([False, False], dtype=np.bool_)
            )


def write_tiny_hubl_checkpoint(path, dataset_path):
    torch.manual_seed(17)
    actor = DeterministicPolicy(1, 1, 2, 1, 1.0)
    critic = TwinCritic(1, 1, 2, 1)
    config = TD3BCConfig(
        observation_dim=1,
        action_dim=1,
        hidden_dim=2,
        depth=1,
        action_pairing="executed_executed",
    )
    payload = {
        "config": config.__dict__,
        "total_updates": 5,
        "observation_mean": torch.zeros(1, 1),
        "observation_std": torch.ones(1, 1),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }
    torch.save(
        {
            "step": 5,
            "agent": payload,
            "config": {
                "arguments": {"variant": "hubl_constant"},
                "hubl_variant_used_by_shared_builder": "hubl_constant",
                "dataset": {"sha256": sha256_file(dataset_path)},
            },
        },
        path,
    )


def test_cli_dry_run_validates_hubl_data_inverse_and_provenance(tmp_path):
    dataset_path = tmp_path / "tiny.hdf5"
    checkpoint_path = tmp_path / "base.pt"
    output_dir = tmp_path / "unused"
    write_tiny_dataset(dataset_path)
    write_tiny_hubl_checkpoint(checkpoint_path, dataset_path)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset_path),
            "--base-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--dry-run",
            "--precompute-batch-size",
            "2",
            "--audit-observations",
            "2",
        ]
    )
    result = run(args)
    assert result["status"] == "dry_run"
    assert result["provenance"]["operator"]["critic_current_action"] == (
        "logged_executed_action"
    )
    assert result["provenance"]["operator"]["hubl_mc_mixing"] is False
    assert result["provenance"]["operator"]["td3_target_policy_smoothing"] is False
    assert result["provenance"]["method_config"]["fqe_updates"] == 20_000
    assert not output_dir.exists()
