import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_channel_opex import (  # noqa: E402
    ChannelOPEXConfig,
    channel_opex_step,
    make_episode_gradient_generator,
)
from evaluate_inverse_residual_adapter import load_controller  # noqa: E402
from inverse_residual_core import module_state_sha256  # noqa: E402
from td3bc_core import TD3BCAgent, TD3BCConfig  # noqa: E402
from train_opex_distilled_adapter import (  # noqa: E402
    METHOD_NAME,
    batched_channel_opex_teacher,
    build_parser,
    run,
)


class QuadraticQ1:
    def q1_only(self, observations, actions):
        target = 0.55 + 0.05 * observations[:, 0]
        return -(actions[:, 0] - target).square()


class LinearQ1:
    def q1_only(self, observations, actions):
        del observations
        weights = torch.arange(
            1, actions.shape[1] + 1, dtype=actions.dtype, device=actions.device
        )
        return (actions * weights).sum(dim=1)


def test_batched_teacher_batch_one_exactly_matches_evaluator_step():
    config = ChannelOPEXConfig(
        action_dim=1,
        step_size=0.1,
        channel_beta=0.4,
        gradient_noise_samples=8,
        gradient_steps=2,
        delta_max=0.25,
        baseline_transform="inverse",
    )
    observation = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    anchor = torch.tensor([[0.15]], dtype=torch.float32)
    first = make_episode_gradient_generator("cpu", 271828)
    second = make_episode_gradient_generator("cpu", 271828)
    expected, expected_details = channel_opex_step(
        QuadraticQ1(),
        observation,
        anchor,
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        first,
    )
    actual, details = batched_channel_opex_teacher(
        QuadraticQ1(),
        observation,
        anchor,
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        config,
        second,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.get_state(), second.get_state(), rtol=0.0, atol=0.0)
    assert details["q1_rows"] == expected_details["q1_rows"] == 16
    assert details["q1_gradient_calls"] == expected_details["q1_gradient_calls"] == 2
    assert details["action_gradient_vectors"] == 2
    assert float(details["teacher_residual_abs_max"]) == pytest.approx(
        max(np.abs(expected_details["applied_residual"])), abs=1e-7
    )


def test_batched_teacher_is_fixed_anchor_bounded_and_global_rng_isolated():
    config = ChannelOPEXConfig(
        action_dim=2,
        step_size=10.0,
        channel_beta=0.5,
        gradient_noise_samples=8,
        gradient_steps=2,
        delta_max=0.25,
        baseline_transform="inverse",
    )
    observations = torch.zeros(3, 4)
    anchors = torch.tensor([[0.9, -0.9], [0.0, 0.0], [-0.8, 0.8]])
    generator = make_episode_gradient_generator("cpu", 99)
    global_before = torch.random.get_rng_state().clone()
    commands, details = batched_channel_opex_teacher(
        LinearQ1(),
        observations,
        anchors,
        torch.full((2,), -1.0),
        torch.full((2,), 1.0),
        config,
        generator,
    )
    assert torch.all(commands <= 1.0) and torch.all(commands >= -1.0)
    assert torch.all((commands - anchors).abs() <= 0.25 + 1e-7)
    assert details["q1_rows"] == 3 * 8 * 2
    assert details["half_noise_vectors"] == 3 * 4 * 2
    assert details["trust_region_clipped_value_count"] > 0
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert not commands.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batched_teacher_cuda_smoke_uses_concrete_generator_device():
    device = torch.device("cuda")
    config = ChannelOPEXConfig(
        action_dim=2,
        step_size=0.1,
        channel_beta=1.25,
        gradient_noise_samples=8,
        gradient_steps=2,
        delta_max=0.25,
        baseline_transform="inverse",
    )
    observations = torch.zeros(3, 4, device=device)
    anchors = torch.zeros(3, 2, device=device)
    commands, details = batched_channel_opex_teacher(
        LinearQ1(),
        observations,
        anchors,
        torch.full((2,), -1.0, device=device),
        torch.full((2,), 1.0, device=device),
        config,
        make_episode_gradient_generator(device, 107),
    )
    assert commands.device.type == "cuda"
    assert torch.isfinite(commands).all()
    assert details["q1_rows"] == 3 * 8 * 2


def _make_agent():
    config = TD3BCConfig(
        observation_dim=2,
        action_dim=1,
        hidden_dim=8,
        depth=1,
        action_pairing="executed_executed",
    )
    return TD3BCAgent(
        config,
        torch.device("cpu"),
        np.asarray([0.2, -0.1], dtype=np.float32),
        np.asarray([0.8, 1.2], dtype=np.float32),
    )


def _make_inputs(tmp_path):
    dataset = tmp_path / "states.hdf5"
    observations = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(16, 2)
    terminals = np.zeros(16, dtype=np.bool_)
    terminals[[3, 7, 11, 15]] = True
    with h5py.File(dataset, "w") as handle:
        handle["observations"] = observations
        handle["terminals"] = terminals
    torch.manual_seed(901)
    agent = _make_agent()
    base_checkpoint = tmp_path / "base.pt"
    torch.save({"step": 17, "agent": agent.checkpoint()}, base_checkpoint)
    return dataset, base_checkpoint


def _args(dataset, base_checkpoint, output_dir, updates, *extra):
    return build_parser().parse_args(
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
            "--teacher-k",
            "2",
            "--teacher-gradient-steps",
            "2",
            "--teacher-noise-seed",
            "71",
            "--audit-fraction",
            "0.3",
            "--split-seed",
            "1234",
            "--log-period",
            "1",
            "--checkpoint-period",
            "2",
            "--torch-threads",
            "1",
            *extra,
        ]
    )


def _nested_exact(first, second):
    assert type(first) is type(second)
    if torch.is_tensor(first):
        assert torch.equal(first.cpu(), second.cpu())
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _nested_exact(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _nested_exact(left, right)
    else:
        assert first == second


def test_training_smoke_freezes_base_and_emits_evaluator_compatible_checkpoint(tmp_path):
    dataset, base_checkpoint = _make_inputs(tmp_path)
    before_agent, _, _ = __import__(
        "train_inverse_residual_adapter"
    ).load_frozen_physical_agent(base_checkpoint, torch.device("cpu"))
    before_hash = module_state_sha256(
        {
            "actor": before_agent.actor,
            "critic": before_agent.critic,
            "actor_target": before_agent.actor_target,
            "critic_target": before_agent.critic_target,
        }
    )
    output_dir = tmp_path / "distilled"
    summary = run(_args(dataset, base_checkpoint, output_dir, 1))
    checkpoint = torch.load(output_dir / "latest.pt", map_location="cpu")
    config = json.loads((output_dir / "config.json").read_text("utf-8"))
    controller, provenance = load_controller(
        output_dir / "latest.pt", torch.device("cpu")
    )

    assert summary["status"] == "complete"
    assert summary["method"] == METHOD_NAME
    assert summary["base_parameters_unchanged"] is True
    assert summary["base_parameter_sha256_before"] == before_hash
    assert summary["base_parameter_sha256_after"] == before_hash
    assert summary["teacher_accounting"]["q1_input_rows"] == 1 * 4 * 2 * 2
    assert summary["deployment_q1_rows_per_action"] == 0
    assert checkpoint["format"] == "inverse_residual_adapter_v2"
    assert checkpoint["training_method"] == METHOD_NAME
    assert checkpoint["teacher"]["noise_sample_calls"] == 2
    assert checkpoint["teacher"]["half_noise_vectors"] == 1 * 4 * 1 * 2
    assert config["observation_split"]["audit_observation_count"] > 0
    assert config["baseline_precomputation"]["observation_count"] == config[
        "observation_split"
    ]["train_observation_count"]
    assert controller.adapter.config.baseline_transform == "inverse"
    assert provenance["training_observation_split"] == config["observation_split"]


def test_exact_resume_matches_continuous_teacher_and_student_states(tmp_path):
    dataset, base_checkpoint = _make_inputs(tmp_path)
    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    run(_args(dataset, base_checkpoint, continuous_dir, 4))
    run(_args(dataset, base_checkpoint, resumed_dir, 2))
    prefix = (resumed_dir / "progress.jsonl").read_bytes()
    run(_args(dataset, base_checkpoint, resumed_dir, 4, "--resume"))
    continuous = torch.load(continuous_dir / "latest.pt", map_location="cpu")
    resumed = torch.load(resumed_dir / "latest.pt", map_location="cpu")
    assert resumed["step"] == continuous["step"] == 4
    assert resumed["resume_count"] == 1
    assert resumed["teacher"]["q1_input_rows"] == 4 * 4 * 2 * 2
    for field in (
        "adapter",
        "optimizer",
        "teacher",
        "index_generator_state",
        "global_rng_state",
    ):
        _nested_exact(continuous[field], resumed[field])
    progress = (resumed_dir / "progress.jsonl").read_bytes()
    assert progress.startswith(prefix)
    events = [json.loads(line) for line in progress.decode().splitlines()]
    assert [event["step"] for event in events if event["event"] == "train"] == [
        1,
        2,
        3,
        4,
    ]
