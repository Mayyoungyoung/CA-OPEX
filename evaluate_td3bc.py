"""Evaluate a saved TD3+BC checkpoint under clean and noisy actuators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from td3bc_core import TD3BCAgent, TD3BCConfig
from evaluation_controls import (
    COMMAND_TRANSFORMS,
    COMMAND_SATURATION_DEFINITION,
    TRANSFORM_SATURATION_DEFINITION,
    prepare_evaluation_command,
    validate_command_scale,
)
from train_iql import save_json, sha256_file
from train_td3bc import evaluate_with_actuator_noise


def load_agent(checkpoint_path: Path, device: torch.device) -> tuple[TD3BCAgent, int]:
    wrapper = torch.load(checkpoint_path, map_location=device)
    if "agent" not in wrapper:
        raise KeyError("checkpoint must contain an agent payload")
    payload = wrapper["agent"]
    config = TD3BCConfig(**payload["config"])
    mean_value = payload["observation_mean"]
    std_value = payload["observation_std"]
    if torch.is_tensor(mean_value):
        mean_value = mean_value.detach().cpu().numpy()
    if torch.is_tensor(std_value):
        std_value = std_value.detach().cpu().numpy()
    mean = np.asarray(mean_value, dtype=np.float32)
    std = np.asarray(std_value, dtype=np.float32)
    agent = TD3BCAgent(config, device, mean, std)
    agent.actor.load_state_dict(payload["actor"])
    agent.actor.eval()
    return agent, int(wrapper.get("step", payload.get("total_updates", -1)))


def evaluate_with_command_scale(
    agent: TD3BCAgent,
    env_name: str,
    seeds: Sequence[int],
    command_scale: float,
    command_transform: str,
    command_transform_beta: float,
    reference_min: float,
    reference_max: float,
) -> Dict[str, object]:
    """Evaluate the clean actuator after the explicit command-gain control."""

    import gymnasium as gym

    scale = validate_command_scale(command_scale)
    env = gym.make(env_name)
    returns: List[float] = []
    lengths: List[int] = []
    command_value_count = 0
    command_saturation_count = 0
    command_at_bound_count = 0
    transform_saturation_count = 0
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            total_return = 0.0
            for length in range(1, int(env.spec.max_episode_steps or 1000) + 1):
                raw_command = agent.act(
                    observation, env.action_space.low, env.action_space.high
                )
                command, command_audit = prepare_evaluation_command(
                    raw_command,
                    env.action_space.low,
                    env.action_space.high,
                    command_scale=scale,
                    command_transform=command_transform,
                    actuator_noise_beta=command_transform_beta,
                )
                command_value_count += command_audit["command_value_count"]
                command_saturation_count += command_audit[
                    "scaled_command_out_of_bounds_count"
                ]
                command_at_bound_count += command_audit[
                    "transformed_command_at_bound_count"
                ]
                transform_saturation_count += command_audit[
                    "command_transform_saturation_count"
                ]
                observation, reward, terminated, truncated, _ = env.step(command)
                total_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(total_return)
            lengths.append(length)
    finally:
        env.close()
    values = np.asarray(returns, dtype=np.float64)
    normalized = 100.0 * (values - reference_min) / (reference_max - reference_min)
    return {
        "episode_seeds": [int(seed) for seed in seeds],
        "command_scale": scale,
        "command_saturation_fraction": command_saturation_count
        / max(command_value_count, 1),
        "command_saturation_definition": COMMAND_SATURATION_DEFINITION,
        "command_transform": command_transform,
        "command_transform_beta": float(command_transform_beta),
        "command_transform_saturation_fraction": transform_saturation_count
        / max(command_value_count, 1),
        "command_transform_saturation_definition": TRANSFORM_SATURATION_DEFINITION,
        "transformed_command_at_bound_fraction": command_at_bound_count
        / max(command_value_count, 1),
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    command_scale = validate_command_scale(args.command_scale)
    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    agent, step = load_agent(checkpoint_path, device)
    environment_seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
    noise_seeds = [args.eval_noise_seed + index for index in range(args.eval_episodes)]
    clean = evaluate_with_command_scale(
        agent,
        args.env_name,
        environment_seeds,
        command_scale,
        args.command_transform,
        args.eval_action_noise_beta,
        args.reference_min_score,
        args.reference_max_score,
    )
    noisy = evaluate_with_actuator_noise(
        agent,
        args.env_name,
        environment_seeds,
        noise_seeds,
        args.eval_action_noise_beta,
        args.reference_min_score,
        args.reference_max_score,
        command_scale=command_scale,
        command_transform=args.command_transform,
    )
    result: Dict[str, object] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": step,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "implementation": {
            "evaluate_td3bc_sha256": sha256_file(Path(__file__).resolve()),
            "train_td3bc_sha256": sha256_file(
                Path(evaluate_with_actuator_noise.__code__.co_filename).resolve()
            ),
            "evaluation_controls_sha256": sha256_file(
                Path(prepare_evaluation_command.__code__.co_filename).resolve()
            ),
        },
        "environment": args.env_name,
        "command_scale": command_scale,
        "command_transform": args.command_transform,
        "command_transform_beta": float(args.eval_action_noise_beta),
        "action_pipeline": (
            "agent_command -> multiply_by_command_scale -> clip_to_environment_bounds "
            "-> command_transform -> clip_to_environment_bounds "
            "-> add_seeded_actuator_noise_if_noisy -> clip_to_environment_bounds -> env.step"
        ),
        "clean": clean,
        "persistent_action_noise": noisy,
        "wall_time_seconds": time.perf_counter() - started,
    }
    save_json(Path(args.output).resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-name", default="Walker2d-v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=9000)
    parser.add_argument("--eval-noise-seed", type=int, default=19000)
    parser.add_argument("--eval-action-noise-beta", type=float, default=1.0)
    parser.add_argument(
        "--command-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply policy commands by this non-negative gain and clip to the "
            "environment bounds before clean execution or actuator noise."
        ),
    )
    parser.add_argument(
        "--command-transform",
        choices=COMMAND_TRANSFORMS,
        default="identity",
        help=(
            "identity, or a monotone inverse of the known clipped Uniform actuator "
            "channel mean. The transform follows command scaling and uses "
            "--eval-action-noise-beta."
        ),
    )
    parser.add_argument("--reference-min-score", type=float, default=1.629008)
    parser.add_argument("--reference-max-score", type=float, default=4592.3)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
