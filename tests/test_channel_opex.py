import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_channel_opex import (
    CHANNEL_OPEX_DELTA_MAX,
    ChannelOPEXConfig,
    build_parser,
    canonical_device,
    channel_opex_step,
    command_from_observation,
    evaluate_arm,
    make_episode_gradient_generator,
    run,
    sample_antithetic_channel_noise,
    sha256_file,
)
from td3bc_core import TD3BCAgent, TD3BCConfig


class LinearQ1:
    def __init__(self, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def q1_only(self, observations, actions):
        del observations
        return (actions * self.weights.to(actions.device)).sum(dim=-1)


def test_cli_exposes_frozen_channel_opex_controls():
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--channel-calibration",
            "calibration.json",
            "--output",
            "result.json",
            "--step-size",
            "0.02",
            "--rollout-beta",
            "1.25",
            "--k",
            "8",
            "--gradient-noise-seed",
            "59311",
            "--eval-episodes",
            "3",
            "--eval-seed",
            "39311",
            "--eval-noise-seed",
            "49311",
        ]
    )
    assert args.step_size == pytest.approx(0.02)
    assert args.baseline_transform == "inverse"
    assert args.delta_max == pytest.approx(CHANNEL_OPEX_DELTA_MAX)
    assert args.gradient_noise_samples == 8
    assert args.gradient_steps == 1
    assert args.gradient_noise_seed == 59311
    assert args.rollout_action_noise_beta == pytest.approx(1.25)
    assert args.eval_episodes == 3
    assert args.eval_seed == 39311
    assert args.eval_noise_seed == 49311


def test_antithetic_gradient_noise_is_reproducible_and_global_rng_isolated():
    config = ChannelOPEXConfig(
        action_dim=2,
        step_size=0.1,
        channel_beta=0.6,
        gradient_noise_samples=8,
    )
    global_state = torch.random.get_rng_state().clone()
    first = make_episode_gradient_generator("cpu", 701)
    repeated = make_episode_gradient_generator("cpu", 701)
    different = make_episode_gradient_generator("cpu", 702)
    first_noise = sample_antithetic_channel_noise(
        3, config, generator=first, dtype=torch.float32, device="cpu"
    )
    repeated_noise = sample_antithetic_channel_noise(
        3, config, generator=repeated, dtype=torch.float32, device="cpu"
    )
    different_noise = sample_antithetic_channel_noise(
        3, config, generator=different, dtype=torch.float32, device="cpu"
    )
    torch.testing.assert_close(first_noise, repeated_noise, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_noise[:, :4], -first_noise[:, 4:])
    assert not torch.equal(first_noise, different_noise)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_positive_beta_multistep_draws_fresh_antithetic_rows_per_gradient_step():
    class RecordingQ1:
        def __init__(self):
            self.action_rows = []

        def q1_only(self, observations, actions):
            del observations
            self.action_rows.append(actions.detach().clone())
            return actions[:, 0]

    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.0,
        channel_beta=0.2,
        gradient_noise_samples=8,
        gradient_steps=2,
    )
    critic = RecordingQ1()
    generator = make_episode_gradient_generator("cpu", 709)
    state_before = generator.get_state().clone()
    _, details = channel_opex_step(
        critic,
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        generator,
    )
    assert len(critic.action_rows) == 2
    for rows in critic.action_rows:
        torch.testing.assert_close(rows[:4], -rows[4:], rtol=0.0, atol=0.0)
    assert not torch.equal(critic.action_rows[0], critic.action_rows[1])
    assert not torch.equal(state_before, generator.get_state())
    assert details["q1_rows"] == 16
    assert details["q1_gradient_calls"] == 2


def test_beta_zero_k1_objective_does_not_consume_gradient_generator():
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.1,
        channel_beta=0.0,
        gradient_noise_samples=1,
        gradient_steps=1,
        delta_max=2.0,
        baseline_transform="identity",
    )
    generator = make_episode_gradient_generator("cpu", 719)
    state_before = generator.get_state().clone()
    _, details = channel_opex_step(
        LinearQ1([1.0]),
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        generator,
    )
    assert torch.equal(state_before, generator.get_state())
    assert details["q1_rows"] == 1


def test_beta_zero_opex_uses_full_inward_q_gradient_at_action_bound():
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.2,
        channel_beta=0.0,
        gradient_noise_samples=1,
        gradient_steps=1,
        delta_max=2.0,
        baseline_transform="identity",
    )
    command, details = channel_opex_step(
        LinearQ1([-1.0]),
        torch.zeros(1, 2),
        torch.tensor([[1.0]]),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        make_episode_gradient_generator("cpu", 721),
    )
    # Original Eq. 7 gives u <- 1 + 0.2*(-1) = 0.8.  A redundant
    # pre-Q clamp at u=1 would use a 0.5 subgradient and incorrectly give 0.9.
    assert float(command.item()) == pytest.approx(0.8, abs=1e-7)
    assert details["gradient_l2"] == pytest.approx(1.0, abs=1e-7)
    assert details["q1_rows"] == 1


def test_single_opex_step_follows_q_gradient_and_clamps_linf_residual():
    config = ChannelOPEXConfig(
        action_dim=2,
        step_size=1.0,
        channel_beta=0.0,
        gradient_noise_samples=8,
    )
    command, details = channel_opex_step(
        LinearQ1([1.0, 2.0]),
        torch.zeros(1, 3),
        torch.zeros(1, 2),
        torch.full((2,), -1.0),
        torch.full((2,), 1.0),
        config,
        make_episode_gradient_generator("cpu", 17),
    )
    torch.testing.assert_close(
        command, torch.full((1, 2), CHANNEL_OPEX_DELTA_MAX)
    )
    assert details["q1_rows"] == 8
    assert details["gradient_l2"] == pytest.approx(np.sqrt(5.0))
    assert details["trust_region_clipped_value_count"] == 2
    assert max(np.abs(details["applied_residual"])) <= CHANNEL_OPEX_DELTA_MAX


def test_action_bound_clamp_can_make_applied_residual_smaller_than_trust_region():
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=1.0,
        channel_beta=0.0,
        gradient_noise_samples=8,
    )
    command, details = channel_opex_step(
        LinearQ1([1.0]),
        torch.zeros(1, 2),
        torch.tensor([[0.9]]),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        make_episode_gradient_generator("cpu", 23),
    )
    assert float(command.item()) == pytest.approx(1.0)
    assert details["applied_residual"][0] == pytest.approx(0.1)
    assert details["command_at_bound_count"] == 1


def test_two_steps_use_two_k_rows_but_cannot_accumulate_past_fixed_anchor_ball():
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.2,
        channel_beta=0.0,
        gradient_noise_samples=8,
        gradient_steps=2,
    )
    command, details = channel_opex_step(
        LinearQ1([1.0]),
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        make_episode_gradient_generator("cpu", 27),
    )
    assert float(command.item()) == pytest.approx(CHANNEL_OPEX_DELTA_MAX)
    assert details["q1_rows"] == 16
    assert details["q1_gradient_calls"] == 2
    assert details["applied_residual"][0] == pytest.approx(
        CHANNEL_OPEX_DELTA_MAX
    )


def test_positive_beta_gradient_uses_exactly_k_antithetic_q_rows():
    class QuadraticQ1:
        def q1_only(self, observations, actions):
            del observations
            return -(actions[:, 0] - 0.7).square()

    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.1,
        channel_beta=0.2,
        gradient_noise_samples=8,
    )
    command, details = channel_opex_step(
        QuadraticQ1(),
        torch.zeros(1, 2),
        torch.tensor([[0.0]]),
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        make_episode_gradient_generator("cpu", 29),
    )
    # Antithetic eps has exactly zero sample mean and no clipping here, so
    # grad_u E[-(u+eps-0.7)^2] at u=0 is exactly 1.4.
    assert float(command.item()) == pytest.approx(0.14, abs=1e-6)
    assert details["q1_rows"] == 8
    assert details["gradient_l2"] == pytest.approx(1.4, abs=1e-6)


class FixedActionAgent:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(max_action=1.0)
        self.critic = LinearQ1(np.ones_like(self._action))

    def act(self, observation, action_low, action_high):
        del observation
        return np.clip(self._action, action_low, action_high)

    def normalize_observations(self, observations):
        return observations


def test_identity_anchor_is_actor_action_and_inverse_path_remains_compensated():
    agent = FixedActionAgent([0.5])
    common = dict(
        base_agent=agent,
        observation=np.asarray([0.0], dtype=np.float32),
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        adapted=False,
        gradient_generator=None,
    )
    identity, identity_details = command_from_observation(
        config=ChannelOPEXConfig(
            action_dim=1,
            step_size=0.1,
            channel_beta=0.8,
            gradient_noise_samples=8,
            baseline_transform="identity",
        ),
        **common,
    )
    inverse, inverse_details = command_from_observation(
        config=ChannelOPEXConfig(
            action_dim=1,
            step_size=0.1,
            channel_beta=0.8,
            gradient_noise_samples=8,
            baseline_transform="inverse",
        ),
        **common,
    )
    assert identity[0] == pytest.approx(0.5)
    assert inverse[0] > identity[0]
    assert identity_details["baseline_transform"] == "identity"
    assert inverse_details["baseline_transform"] == "inverse"


class TinyActionSpace:
    def __init__(self):
        self.low = np.asarray([-1.0], dtype=np.float32)
        self.high = np.asarray([1.0], dtype=np.float32)


class TinyEnv:
    def __init__(self):
        self.action_space = TinyActionSpace()
        self.spec = SimpleNamespace(max_episode_steps=2)
        self._step = 0
        self._seed = 0

    def reset(self, seed):
        self._step = 0
        self._seed = int(seed)
        observation = np.asarray([seed * 0.001, -0.2], dtype=np.float32)
        return observation, {}

    def step(self, action):
        self._step += 1
        observation = np.asarray(
            [self._seed * 0.001, -0.2 + self._step * 0.01], dtype=np.float32
        )
        reward = float(np.asarray(action)[0])
        return observation, reward, self._step >= 2, False, {}

    def close(self):
        return None


def install_tiny_gym(monkeypatch):
    fake_gym = SimpleNamespace(make=lambda name: TinyEnv())
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)


def make_base_agent():
    config = TD3BCConfig(
        observation_dim=2,
        action_dim=1,
        hidden_dim=8,
        depth=1,
        action_pairing="executed_executed",
    )
    agent = TD3BCAgent(
        config,
        torch.device("cpu"),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )
    for module in (agent.actor, agent.actor_target, agent.critic, agent.critic_target):
        module.requires_grad_(False)
        module.eval()
    return agent


def valid_calibration_payload():
    commands_hash = "1" * 64
    executed_hash = "2" * 64
    provenance = {
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
        "estimate": {"beta_mle": 0.4},
        "uncertainty": {"beta_interval": [0.39, 0.41]},
        "input_provenance": provenance,
        "calibration_script_sha256": "4" * 64,
    }


def test_evaluate_arm_uses_episode_local_seeds_and_exact_q_cost(monkeypatch):
    install_tiny_gym(monkeypatch)
    agent = make_base_agent()
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.01,
        channel_beta=0.4,
        gradient_noise_samples=8,
    )
    kwargs = dict(
        base_agent=agent,
        config=config,
        env_name="Tiny-v0",
        environment_seeds=[11, 12],
        action_noise_seeds=[21, 22],
        gradient_noise_seeds=[31, 32],
        rollout_beta=0.3,
        adapted=True,
        reference_min=0.0,
        reference_max=1.0,
    )
    first = evaluate_arm(**kwargs)
    repeated = evaluate_arm(**kwargs)
    assert first["returns"] == repeated["returns"]
    assert first["environment_seeds"] == [11, 12]
    assert first["action_noise_seeds"] == [21, 22]
    assert first["gradient_noise_seeds"] == [31, 32]
    assert first["q1_rows_by_episode"] == [16, 16]
    assert first["q1_rows_total"] == 32
    assert first["q1_gradient_calls"] == 4
    assert first["gradient_noise_stream_used"] is True
    assert first["gradient_noise_sampling_frequency"] == "per_gradient_step"
    assert first["gradient_noise_draw_calls"] == 4


def test_paired_arms_replay_the_same_actuator_rng_stream(monkeypatch):
    install_tiny_gym(monkeypatch)
    agent = make_base_agent()
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.0,
        channel_beta=0.4,
        gradient_noise_samples=8,
    )
    common = dict(
        base_agent=agent,
        config=config,
        env_name="Tiny-v0",
        environment_seeds=[111, 112],
        action_noise_seeds=[211, 212],
        gradient_noise_seeds=[311, 312],
        rollout_beta=0.3,
        reference_min=0.0,
        reference_max=1.0,
    )
    baseline = evaluate_arm(adapted=False, **common)
    adapted = evaluate_arm(adapted=True, **common)
    # A zero OPEX step makes commands identical, so exact equality establishes
    # that each arm replays the same per-episode actuator-noise stream.
    assert baseline["returns"] == adapted["returns"]
    assert baseline["executed_commanded_action_mse"] == pytest.approx(
        adapted["executed_commanded_action_mse"], rel=0.0, abs=0.0
    )
    assert baseline["action_noise_seeds"] == adapted["action_noise_seeds"]


def test_cpu_run_smoke_writes_paired_provenance_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    install_tiny_gym(monkeypatch)
    torch.manual_seed(811)
    base = make_base_agent()
    base_checkpoint = tmp_path / "base.pt"
    torch.save({"format": "tiny_test", "step": 13, "agent": base.checkpoint()}, base_checkpoint)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(valid_calibration_payload()), encoding="utf-8"
    )
    output = tmp_path / "opex.json"
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            str(base_checkpoint),
            "--channel-calibration",
            str(calibration_path),
            "--model-action-noise-beta",
            "0.4",
            "--output",
            str(output),
            "--env-name",
            "Tiny-v0",
            "--device",
            "cpu",
            "--step-size",
            "0.01",
            "--rollout-action-noise-beta",
            "0.3",
            "--eval-episodes",
            "2",
            "--eval-seed",
            "41",
            "--eval-noise-seed",
            "51",
            "--gradient-noise-seed",
            "61",
        ]
    )
    result = run(args)
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == on_disk["status"] == "complete"
    assert result["method_scope"].startswith("stronger channel-aware extension")
    assert result["raw_schema"] == "channel_opex_v1"
    assert result["base_checkpoint"]["sha256"] == sha256_file(base_checkpoint)
    assert result["calibration"]["calibration_sha256"] == sha256_file(
        calibration_path
    )
    assert result["arms"]["baseline_only"]["q1_rows_total"] == 0
    assert result["arms"]["adapted"]["q1_rows_total"] == 32
    assert result["paired"]["episode_count"] == 2
    assert result["controller"]["gradient_steps_per_action"] == 1
    assert result["controller"]["actor_parameter_updates"] == 0
    assert result["controller"]["critic_parameter_updates"] == 0
    assert result["cost"]["q1_forward_rows"] == 32
    assert result["cost"]["q1_backward_rows"] == 32
    assert result["cost"]["cost_scope"] == (
        "adapted_arm_deployment_controller_only"
    )
    assert result["cost"]["environment_steps"] == 4
    assert result["cost"]["base_actor_rows"] == 4
    assert result["cost"]["baseline_environment_steps"] == 4
    assert result["cost"]["baseline_base_actor_rows"] == 4
    assert result["cost"]["paired_evaluation_environment_steps"] == 8
    assert set(
        (
            "evaluate_sha256",
            "inverse_residual_core_sha256",
            "td3bc_core_sha256",
            "train_td3bc_sha256",
        )
    ).issubset(result["implementation"])

    original_output = tmp_path / "original_structure.json"
    original_args = build_parser().parse_args(
        [
            "--base-checkpoint",
            str(base_checkpoint),
            "--model-action-noise-beta",
            "0",
            "--output",
            str(original_output),
            "--env-name",
            "Tiny-v0",
            "--device",
            "cpu",
            "--baseline-transform",
            "identity",
            "--step-size",
            "0.1",
            "--gradient-steps",
            "1",
            "--k",
            "1",
            "--delta-max",
            "2",
            "--rollout-action-noise-beta",
            "0.3",
            "--eval-episodes",
            "2",
        ]
    )
    original = run(original_args)
    assert original["method_id"] == "original_structure_opex_t1"
    assert original["method_scope"].startswith("original single-step OPEX")
    assert original["calibration"]["calibration_sha256"] is None
    assert original["controller"]["baseline_transform"] == "identity"
    assert original["controller"]["model_beta"] == pytest.approx(0.0)
    assert original["controller"]["K"] == 1
    assert original["controller"]["gradient_steps"] == 1
    assert original["controller"]["delta_max"] == pytest.approx(2.0)
    assert original["arms"]["adapted"]["q1_rows_total"] == 4
    assert original["arms"]["adapted"]["gradient_noise_stream_used"] is False
    assert original["arms"]["adapted"]["gradient_noise_draw_calls"] == 0
    assert original["evaluation_protocol"]["gradient_noise_sampling_frequency"] == (
        "none_beta_zero"
    )
    assert original["evaluation_protocol"]["gradient_noise_antithetic"] is False
    assert "deterministic_zero" in original["controller"]["gradient_objective"]

    missing_channel_args = build_parser().parse_args(
        [
            "--base-checkpoint",
            str(base_checkpoint),
            "--output",
            str(tmp_path / "missing_channel.json"),
            "--device",
            "cpu",
            "--step-size",
            "0.1",
            "--rollout-beta",
            "0.3",
        ]
    )
    with pytest.raises(ValueError, match="provide --channel-calibration"):
        run(missing_channel_args)
    assert not (tmp_path / "missing_channel.json").exists()

    mismatched_channel_args = build_parser().parse_args(
        [
            "--base-checkpoint",
            str(base_checkpoint),
            "--channel-calibration",
            str(calibration_path),
            "--model-action-noise-beta",
            "0.5",
            "--output",
            str(tmp_path / "mismatched_channel.json"),
            "--device",
            "cpu",
            "--step-size",
            "0.1",
            "--rollout-beta",
            "0.3",
        ]
    )
    with pytest.raises(ValueError, match="authoritative"):
        run(mismatched_channel_args)
    assert not (tmp_path / "mismatched_channel.json").exists()

    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(args)
    assert output.read_bytes() == before


def test_validation_rejects_odd_positive_beta_k_and_device_is_concrete():
    with pytest.raises(ValueError, match="even K"):
        ChannelOPEXConfig(
            action_dim=1,
            step_size=0.1,
            channel_beta=0.5,
            gradient_noise_samples=7,
        )
    assert canonical_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert canonical_device("cuda").index is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_implicit_device_multistep_gradient_smoke():
    device = canonical_device("cuda")
    config = ChannelOPEXConfig(
        action_dim=2,
        step_size=0.03,
        channel_beta=0.4,
        gradient_noise_samples=8,
        gradient_steps=2,
    )
    command, details = channel_opex_step(
        LinearQ1([1.0, -0.5]),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 2, device=device),
        torch.full((2,), -1.0, device=device),
        torch.full((2,), 1.0, device=device),
        config,
        make_episode_gradient_generator("cuda", 727),
    )
    assert command.device == device
    assert torch.isfinite(command).all()
    assert details["q1_rows"] == 16
    assert details["q1_gradient_calls"] == 2
