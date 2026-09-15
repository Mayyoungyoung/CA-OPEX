import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from td3bc_core import (
    ACTION_PAIRINGS,
    TD3BCAgent,
    TD3BCConfig,
    fixed_antithetic_uniform_samples,
)
from train_td3bc import (
    SUPPORTED_VARIANTS,
    build_parser,
    load_action_channels,
    resolve_action_pairing,
    resolve_shared_hubl_variant,
)


def make_agent(**overrides):
    arguments = dict(
        observation_dim=2,
        action_dim=1,
        hidden_dim=8,
        depth=1,
        discount=0.9,
        policy_frequency=2,
    )
    arguments.update(overrides)
    return TD3BCAgent(
        TD3BCConfig(**arguments),
        torch.device("cpu"),
        np.asarray([2.0, -1.0], dtype=np.float32),
        np.asarray([2.0, 0.5], dtype=np.float32),
    )


def make_batch(batch_size=4):
    return {
        "observations": torch.zeros(batch_size, 2),
        "executed_actions": torch.zeros(batch_size, 1),
        "commanded_actions": torch.zeros(batch_size, 1),
        "next_observations": torch.zeros(batch_size, 2),
        "rewards": torch.ones(batch_size),
        "terminals": torch.zeros(batch_size),
        "heuristic_next": torch.zeros(batch_size),
        "lambdas": torch.zeros(batch_size),
    }


def test_state_normalization_is_applied_exactly():
    agent = make_agent()
    values = torch.tensor([[4.0, 0.0], [0.0, -2.0]])
    expected = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    torch.testing.assert_close(agent.normalize_observations(values), expected)


def test_hubl_target_mixes_heuristic_and_double_q():
    agent = make_agent()
    with torch.no_grad():
        for parameter in agent.actor_target.parameters():
            parameter.zero_()
        for parameter in agent.critic_target.parameters():
            parameter.zero_()
        agent.critic_target.q1[-1].bias.fill_(5.0)
        agent.critic_target.q2[-1].bias.fill_(7.0)
    batch = make_batch(batch_size=2)
    batch["rewards"].fill_(2.0)
    batch["heuristic_next"].fill_(10.0)
    batch["lambdas"].fill_(0.25)
    target = agent.compute_critic_target(batch, target_noise=torch.zeros(2, 1))
    # 2 + 0.9 * (0.25*10 + 0.75*min(5, 7)) = 7.625
    torch.testing.assert_close(target, torch.full((2,), 7.625))
    batch["terminals"][0] = 1.0
    target = agent.compute_critic_target(batch, target_noise=torch.zeros(2, 1))
    assert target[0].item() == pytest.approx(2.0)


def test_noise_marginalized_hubl_constant_uses_same_mixed_target_equation():
    agent = make_resampled_agent(seed=13, beta=0.2, sample_count=2, action_dim=1)
    with torch.no_grad():
        for parameter in agent.actor_target.parameters():
            parameter.zero_()
        for parameter in agent.critic_target.parameters():
            parameter.zero_()
        agent.critic_target.q1[-1].bias.fill_(5.0)
        agent.critic_target.q2[-1].bias.fill_(7.0)
    batch = make_batch(batch_size=2)
    batch["rewards"].fill_(2.0)
    batch["heuristic_next"].fill_(10.0)
    batch["lambdas"].fill_(0.25)
    target = agent.compute_critic_target(batch, target_noise=torch.zeros(2, 1))
    torch.testing.assert_close(target, torch.full((2,), 7.625))
    assert agent.execution_noise_draw_calls == {"target": 1, "actor": 0}


def test_actor_and_targets_update_only_on_delayed_step():
    torch.manual_seed(1)
    agent = make_agent(policy_frequency=2)
    batch = make_batch()
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    first = agent.update(batch)
    after_first = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    assert first["actor_updated"] == 0.0
    assert all(torch.equal(left, right) for left, right in zip(before, after_first))
    second = agent.update(batch)
    after_second = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    assert second["actor_updated"] == 1.0
    assert any(not torch.equal(left, right) for left, right in zip(after_first, after_second))
    assert "actor_q_scale" in second and second["actor_q_scale"] > 0.0


def test_checkpoint_contains_normalizer_and_both_target_networks():
    agent = make_agent()
    payload = agent.checkpoint()
    assert payload["observation_mean"].shape == (1, 2)
    assert payload["observation_std"].shape == (1, 2)
    assert "actor_target" in payload and "critic_target" in payload


def test_target_smoothing_respects_noise_and_action_clips():
    agent = make_agent(max_action=1.0, policy_noise=0.2, noise_clip=0.5)
    normalized_next = torch.zeros(3, 2)
    huge_noise = torch.full((3, 1), 100.0)
    actions = agent._target_actions(normalized_next, huge_noise)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)


def test_fixed_antithetic_samples_are_seeded_zero_mean_and_bounded():
    first = fixed_antithetic_uniform_samples(3, 5, beta=0.7, seed=17)
    repeat = fixed_antithetic_uniform_samples(3, 5, beta=0.7, seed=17)
    changed = fixed_antithetic_uniform_samples(3, 5, beta=0.7, seed=18)
    assert first.shape == (5, 3)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, changed)
    np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-7)
    assert np.max(np.abs(first)) <= 0.7


def make_resampled_agent(seed=31, beta=0.4, sample_count=4, action_dim=2):
    return make_agent(
        action_dim=action_dim,
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=beta,
        execution_noise_samples=sample_count,
        execution_noise_seed=seed,
        execution_noise_scheme="resampled_antithetic",
    )


def test_resampled_antithetic_is_per_state_symmetric_bounded_and_reproducible():
    commands = torch.zeros(3, 2)
    first_agent = make_resampled_agent(seed=31)
    repeated_agent = make_resampled_agent(seed=31)
    changed_seed_agent = make_resampled_agent(seed=32)
    first = first_agent.execution_action_samples(commands, stream="actor")
    repeated = repeated_agent.execution_action_samples(commands, stream="actor")
    changed_seed = changed_seed_agent.execution_action_samples(
        commands, stream="actor"
    )
    assert first.shape == (3, 4, 2)
    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
    assert not torch.equal(first, changed_seed)
    torch.testing.assert_close(first[:, :2], -first[:, 2:])
    assert torch.all(first.abs() <= 0.4)
    assert not torch.equal(first[0], first[1])
    second = first_agent.execution_action_samples(commands, stream="actor")
    assert not torch.equal(first, second)


def test_resampled_streams_are_independent_and_do_not_advance_global_rng():
    commands = torch.zeros(2, 2)
    with_actor_call = make_resampled_agent(seed=41)
    target_only = make_resampled_agent(seed=41)
    global_before = torch.random.get_rng_state().clone()
    first_target = with_actor_call.execution_action_samples(
        commands, stream="target"
    )
    torch.testing.assert_close(
        first_target,
        target_only.execution_action_samples(commands, stream="target"),
        rtol=0.0,
        atol=0.0,
    )
    with_actor_call.execution_action_samples(commands, stream="actor")
    second_target = with_actor_call.execution_action_samples(
        commands, stream="target"
    )
    torch.testing.assert_close(
        second_target,
        target_only.execution_action_samples(commands, stream="target"),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_before)


def test_execution_noise_metadata_checkpoint_and_restore_do_not_consume_rng():
    commands = torch.zeros(2, 2)
    source = make_resampled_agent(seed=53)
    source.execution_action_samples(commands, stream="actor")
    source.execution_action_samples(commands, stream="target")
    states_before = {
        stream: generator.get_state().clone()
        for stream, generator in source.execution_noise_generators.items()
    }
    metadata = source.execution_noise_metadata()
    checkpoint = source.checkpoint()
    for stream, generator in source.execution_noise_generators.items():
        torch.testing.assert_close(generator.get_state(), states_before[stream])
    assert metadata["resampled_per_state_and_q_call"] is True
    assert metadata["fixed_action_perturbations"] is None
    assert metadata["execution_noise_draw_calls"] == {"target": 1, "actor": 1}
    assert metadata["trainer_resume_restores_generator_state"] is False
    expected_actor = source.execution_action_samples(commands, stream="actor")
    expected_target = source.execution_action_samples(commands, stream="target")

    restored = make_resampled_agent(seed=53)
    restored.restore_execution_noise_state(checkpoint)
    torch.testing.assert_close(
        restored.execution_action_samples(commands, stream="actor"),
        expected_actor,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored.execution_action_samples(commands, stream="target"),
        expected_target,
        rtol=0.0,
        atol=0.0,
    )


def test_resampled_positive_beta_requires_even_k_but_zero_beta_allows_k1():
    with pytest.raises(ValueError, match="even"):
        make_resampled_agent(beta=0.3, sample_count=3)
    zero = make_resampled_agent(beta=0.0, sample_count=1)
    actions = zero.execution_action_samples(torch.zeros(2, 2), stream="target")
    assert actions.shape == (2, 1, 2)
    assert torch.count_nonzero(actions) == 0


def test_execution_action_quadrature_has_batch_sample_action_shape_and_clips():
    agent = make_agent(
        action_dim=2,
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.8,
        execution_noise_samples=4,
        execution_noise_seed=23,
    )
    commands = torch.tensor([[0.9, -0.9], [-0.2, 0.3]])
    actions = agent.execution_action_samples(commands)
    assert actions.shape == (2, 4, 2)
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)
    metadata = agent.execution_noise_metadata()
    assert metadata["sample_count"] == 4
    assert metadata["target_twin_q_rows_per_transition"] == 4
    assert len(metadata["fixed_action_perturbations"]) == 4


def test_target_smoothing_is_one_command_noise_shared_across_channel_samples():
    agent = make_agent(
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.2,
        execution_noise_samples=4,
    )
    with torch.no_grad():
        for parameter in agent.actor_target.parameters():
            parameter.zero_()
    normalized_next = torch.zeros(2, 2)
    smoothing = torch.tensor([[0.1], [-0.1]])
    commands = agent._target_actions(normalized_next, smoothing)
    physical = agent.execution_action_samples(commands)
    observed_channel_noise = physical - commands[:, None, :]
    expected = agent.execution_noise[None, :, :].expand(2, -1, -1)
    torch.testing.assert_close(observed_channel_noise, expected)


@pytest.mark.parametrize(
    ("scheme", "sample_count"),
    (
        ("fixed_antithetic", 1),
        ("fixed_antithetic", 4),
        ("resampled_antithetic", 1),
        ("resampled_antithetic", 4),
    ),
)
def test_zero_noise_samples_degenerate_to_dual_action_td3bc(scheme, sample_count):
    torch.manual_seed(101)
    baseline = make_agent(action_pairing="executed_commanded")
    marginalized = make_agent(
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.0,
        execution_noise_samples=sample_count,
        execution_noise_scheme=scheme,
    )
    marginalized.actor.load_state_dict(baseline.actor.state_dict())
    marginalized.actor_target.load_state_dict(baseline.actor_target.state_dict())
    marginalized.critic.load_state_dict(baseline.critic.state_dict())
    marginalized.critic_target.load_state_dict(baseline.critic_target.state_dict())
    batch = make_batch(batch_size=5)
    batch["observations"] = torch.randn(5, 2)
    batch["next_observations"] = torch.randn(5, 2)
    target_noise = torch.zeros(5, 1)
    torch.testing.assert_close(
        baseline.compute_critic_target(batch, target_noise),
        marginalized.compute_critic_target(batch, target_noise),
        rtol=0.0,
        atol=0.0,
    )
    normalized = baseline.normalize_observations(batch["observations"])
    baseline_loss = baseline.compute_actor_loss(
        normalized, batch["commanded_actions"]
    )
    marginalized_loss = marginalized.compute_actor_loss(
        normalized, batch["commanded_actions"]
    )
    for left, right in zip(baseline_loss, marginalized_loss):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_target_marginalizes_each_twin_before_taking_min_and_logs_swap_gap():
    agent = make_agent(
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.5,
        execution_noise_samples=2,
    )
    batch = make_batch(batch_size=1)

    def crossed_twin_values(observations, actions):
        assert observations.shape[0] == actions.shape[0] == 2
        return torch.tensor([0.0, 10.0]), torch.tensor([9.0, 1.0])

    agent.critic_target.both = crossed_twin_values
    target = agent.compute_critic_target(batch, target_noise=torch.zeros(1, 1))
    # min(mean([0, 10]), mean([9, 1])) = 5.0. The reversed operators
    # produce 0.5, so this also catches accidental extra pessimism.
    assert target.item() == pytest.approx(1.0 + 0.9 * 5.0)
    assert agent.last_target_twin_reduction_gap == pytest.approx(4.5)

    pessimistic = make_agent(
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.5,
        execution_noise_samples=2,
        execution_noise_twin_reduction="expectation_of_min",
    )
    pessimistic.critic_target.both = crossed_twin_values
    pessimistic_target = pessimistic.compute_critic_target(
        batch, target_noise=torch.zeros(1, 1)
    )
    assert pessimistic_target.item() == pytest.approx(1.0 + 0.9 * 0.5)


def test_noise_marginalized_update_uses_executed_data_and_commanded_bc():
    torch.manual_seed(29)
    agent = make_agent(
        policy_frequency=1,
        action_pairing="executed_commanded",
        marginalize_execution_noise=True,
        execution_noise_beta=0.4,
        execution_noise_samples=4,
    )
    batch = make_batch(batch_size=3)
    batch["executed_actions"].fill_(0.25)
    batch["commanded_actions"].fill_(-0.75)
    data_actions_seen = []
    target_actions_seen = []
    actor_physical_actions_seen = []
    bc_seen = []
    original_data_both = agent.critic.both
    original_target_both = agent.critic_target.both
    original_actor_q1 = agent.critic.q1_only
    original_actor_loss = agent.compute_actor_loss

    def record_data(observations, actions):
        data_actions_seen.append(actions.detach().clone())
        return original_data_both(observations, actions)

    def record_target(observations, actions):
        target_actions_seen.append(actions.detach().clone())
        return original_target_both(observations, actions)

    def record_actor_q1(observations, actions):
        actor_physical_actions_seen.append(actions.detach().clone())
        return original_actor_q1(observations, actions)

    def record_actor_loss(observations, actions):
        bc_seen.append(actions.detach().clone())
        return original_actor_loss(observations, actions)

    agent.critic.both = record_data
    agent.critic_target.both = record_target
    agent.critic.q1_only = record_actor_q1
    agent.compute_actor_loss = record_actor_loss
    metrics = agent.update(batch)

    torch.testing.assert_close(data_actions_seen[-1], torch.full((3, 1), 0.25))
    torch.testing.assert_close(bc_seen[-1], torch.full((3, 1), -0.75))
    assert target_actions_seen[-1].shape == (12, 1)
    assert actor_physical_actions_seen[-1].shape == (12, 1)
    assert metrics["critic_uses_commanded_action"] == 0.0
    assert metrics["bc_uses_commanded_action"] == 1.0
    assert metrics["execution_noise_marginalized"] == 1.0
    assert metrics["execution_noise_sample_count"] == 4.0


@pytest.mark.parametrize(
    ("pairing", "expected_critic", "expected_bc"),
    (
        ("executed_executed", 0.25, 0.25),
        ("commanded_commanded", -0.75, -0.75),
        ("executed_commanded", 0.25, -0.75),
    ),
)
def test_update_routes_explicit_action_channels_to_critic_and_actor(
    pairing, expected_critic, expected_bc
):
    """Spy at the optimizer boundary, not only at a standalone selector."""

    torch.manual_seed(9)
    agent = make_agent(policy_frequency=1, action_pairing=pairing)
    batch = make_batch(batch_size=3)
    batch["executed_actions"].fill_(0.25)
    batch["commanded_actions"].fill_(-0.75)
    critic_seen = []
    bc_seen = []
    original_critic_both = agent.critic.both
    original_actor_loss = agent.compute_actor_loss

    def record_critic(observations, actions):
        critic_seen.append(actions.detach().clone())
        return original_critic_both(observations, actions)

    def record_actor(observations, actions):
        bc_seen.append(actions.detach().clone())
        return original_actor_loss(observations, actions)

    agent.critic.both = record_critic
    agent.compute_actor_loss = record_actor
    metrics = agent.update(batch)

    torch.testing.assert_close(
        critic_seen[-1], torch.full((3, 1), expected_critic)
    )
    torch.testing.assert_close(bc_seen[-1], torch.full((3, 1), expected_bc))
    assert metrics["critic_uses_commanded_action"] == float(
        pairing == "commanded_commanded"
    )
    assert metrics["bc_uses_commanded_action"] == float(
        pairing != "executed_executed"
    )


def test_all_declared_action_pairings_validate_and_missing_channels_fail():
    for pairing in ACTION_PAIRINGS:
        agent = make_agent(action_pairing=pairing)
        critic_action, bc_action = agent.select_action_channels(make_batch())
        assert critic_action.shape == bc_action.shape == (4, 1)
    with pytest.raises(ValueError, match="action_pairing"):
        make_agent(action_pairing="ambiguous")
    agent = make_agent()
    invalid = make_batch()
    del invalid["commanded_actions"]
    with pytest.raises(KeyError, match="explicit executed_actions"):
        agent.update(invalid)


def test_loader_passes_clean_policy_actions_through_exact_timeout_mask(tmp_path):
    path = tmp_path / "dual_actions.hdf5"
    with h5py.File(path, "w") as handle:
        handle["actions"] = np.arange(5, dtype=np.float32)[:, None]
        handle["clean_policy_actions"] = (
            np.arange(5, dtype=np.float32)[:, None] + 10.0
        )
        handle["terminals"] = np.asarray([0, 1, 0, 0, 0], dtype=np.bool_)
        handle["timeouts"] = np.asarray([0, 0, 0, 0, 1], dtype=np.bool_)
    # prepare_dataset keeps true terminal rows and discards timeout rows.
    executed_after_prepare = np.arange(4, dtype=np.float32)[:, None]
    commanded, manifest = load_action_channels(
        path,
        executed_after_prepare,
        action_pairing="executed_commanded",
        max_episodes=None,
    )
    np.testing.assert_array_equal(
        commanded, np.arange(10, 14, dtype=np.float32)[:, None]
    )
    assert manifest["critic_action_source"] == "actions"
    assert manifest["behavior_cloning_action_source"] == "clean_policy_actions"
    assert manifest["commanded_action_source"] == "hdf5:clean_policy_actions"


def test_commanded_pairing_rejects_dataset_without_command_channel(tmp_path):
    path = tmp_path / "executed_only.hdf5"
    with h5py.File(path, "w") as handle:
        handle["actions"] = np.zeros((2, 1), dtype=np.float32)
        handle["terminals"] = np.asarray([0, 1], dtype=np.bool_)
        handle["timeouts"] = np.zeros(2, dtype=np.bool_)
    with pytest.raises(KeyError, match="requires HDF5 field"):
        load_action_channels(
            path,
            np.zeros((2, 1), dtype=np.float32),
            action_pairing="executed_commanded",
            max_episodes=None,
        )


@pytest.mark.parametrize(
    "variant",
    (
        "hubl_horizon_shuffled",
        "hubl_horizon_reverse",
        "hubl_rank_horizon",
        "hubl_action_residual",
        "hubl_action_residual_shuffled",
    ),
)
def test_td3bc_parser_accepts_horizon_controls(variant):
    assert variant in SUPPORTED_VARIANTS
    args = build_parser().parse_args(
        [
            "--dataset",
            "dataset.hdf5",
            "--output-dir",
            "output",
            "--variant",
            variant,
            "--horizon-control-seed",
            "47",
        ]
    )
    assert args.variant == variant
    assert args.horizon_control_seed == 47


def test_noise_marginalized_variant_resolves_channels_and_cli_parameters():
    assert "noise_marginalized_td3bc" in SUPPORTED_VARIANTS
    args = build_parser().parse_args(
        [
            "--dataset",
            "dataset.hdf5",
            "--output-dir",
            "output",
            "--variant",
            "noise_marginalized_td3bc",
            "--execution-noise-beta",
            "0.6",
            "--execution-noise-samples",
            "6",
            "--execution-noise-seed",
            "71",
        ]
    )
    assert args.action_pairing is None
    assert resolve_action_pairing(args.variant, args.action_pairing) == (
        "executed_commanded"
    )
    assert args.execution_noise_beta == pytest.approx(0.6)
    assert args.execution_noise_samples == 6
    assert args.execution_noise_seed == 71
    assert args.execution_noise_scheme == "resampled_antithetic"
    assert args.execution_noise_twin_reduction == "min_of_expectations"
    with pytest.raises(ValueError, match="requires --action-pairing"):
        resolve_action_pairing("noise_marginalized_td3bc", "executed_executed")
    with pytest.raises(ValueError, match="requires action_pairing"):
        make_agent(
            action_pairing="executed_executed",
            marginalize_execution_noise=True,
            execution_noise_beta=0.6,
            execution_noise_samples=6,
        )


def test_noise_marginalized_hubl_constant_has_only_declared_builder_mapping():
    expected = {
        "td3bc": "iql",
        "noise_marginalized_td3bc": "iql",
        "noise_marginalized_hubl_constant": "hubl_constant",
        "hubl_constant": "hubl_constant",
        "hubl_rank": "hubl_rank",
    }
    assert {
        variant: resolve_shared_hubl_variant(variant) for variant in expected
    } == expected
    assert "noise_marginalized_hubl_rank" not in SUPPORTED_VARIANTS
    args = build_parser().parse_args(
        [
            "--dataset",
            "dataset.hdf5",
            "--output-dir",
            "output",
            "--variant",
            "noise_marginalized_hubl_constant",
            "--heuristic-discount",
            "0.35",
        ]
    )
    assert resolve_action_pairing(args.variant, args.action_pairing) == (
        "executed_commanded"
    )
    assert resolve_shared_hubl_variant(args.variant) == "hubl_constant"
    assert args.heuristic_discount == pytest.approx(0.35)
    with pytest.raises(ValueError, match="requires --action-pairing"):
        resolve_action_pairing(
            "noise_marginalized_hubl_constant", "commanded_commanded"
        )
