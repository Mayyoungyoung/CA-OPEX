"""Independently evaluate an inverse-only and residual-adapted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from inverse_residual_core import (
    InverseResidualAdapter,
    InverseResidualConfig,
    InverseResidualController,
    module_state_sha256,
)
from train_inverse_residual_adapter import (
    load_frozen_physical_agent,
    require_new_output_file,
)
from train_iql import save_json, sha256_file


def load_controller(
    adapter_checkpoint_path: Path,
    device: torch.device,
    base_checkpoint_override: Optional[Path] = None,
) -> Tuple[InverseResidualController, Dict[str, object]]:
    payload = torch.load(adapter_checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or payload.get("format") not in {
        "inverse_residual_adapter_v1",
        "inverse_residual_adapter_v2",
    }:
        raise ValueError("unsupported inverse-residual adapter checkpoint")
    required = (
        "adapter_config",
        "adapter",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "base_parameter_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"adapter checkpoint lacks fields: {missing}")
    recorded_base = Path(str(payload["base_checkpoint"])).resolve()
    base_checkpoint = (
        base_checkpoint_override.resolve()
        if base_checkpoint_override is not None
        else recorded_base
    )
    actual_base_hash = sha256_file(base_checkpoint)
    if actual_base_hash != payload["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint SHA256 does not match adapter provenance")
    base_agent, _, base_step = load_frozen_physical_agent(base_checkpoint, device)
    base_parameter_hash = module_state_sha256(
        {
            "actor": base_agent.actor,
            "critic": base_agent.critic,
            "actor_target": base_agent.actor_target,
            "critic_target": base_agent.critic_target,
        }
    )
    if base_parameter_hash != payload["base_parameter_sha256"]:
        raise ValueError("loaded base parameter tensors do not match adapter provenance")
    config = InverseResidualConfig(**payload["adapter_config"])
    if config.observation_dim != base_agent.config.observation_dim or (
        config.action_dim != base_agent.config.action_dim
    ):
        raise ValueError("adapter and base checkpoint dimensions do not match")
    adapter = InverseResidualAdapter(config).to(device)
    adapter.load_state_dict(payload["adapter"], strict=True)
    adapter.eval()
    controller = InverseResidualController(base_agent, adapter)
    training_run_config = payload.get("config")
    if training_run_config is not None and not isinstance(training_run_config, dict):
        raise ValueError("adapter checkpoint config payload must be a mapping")
    training_run_config = training_run_config or {}
    provenance = {
        "adapter_checkpoint": str(adapter_checkpoint_path),
        "adapter_checkpoint_sha256": sha256_file(adapter_checkpoint_path),
        "adapter_step": int(payload.get("step", -1)),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": actual_base_hash,
        "base_checkpoint_step": int(base_step),
        "base_parameter_sha256": base_parameter_hash,
        "training_dataset": training_run_config.get("dataset"),
        "training_observation_split": training_run_config.get(
            "observation_split"
        ),
    }
    return controller, provenance


def evaluate_controller(
    controller: InverseResidualController,
    env_name: str,
    environment_seeds: Sequence[int],
    action_noise_seeds: Sequence[int],
    action_noise_beta: float,
    use_residual: bool,
    reference_min: float,
    reference_max: float,
) -> Dict[str, object]:
    """Evaluate one command rule with paired environment and actuator seeds."""

    import gymnasium as gym

    if action_noise_beta < 0.0:
        raise ValueError("action_noise_beta must be non-negative")
    if len(environment_seeds) != len(action_noise_seeds):
        raise ValueError("environment and actuator-noise seeds must align")
    env = gym.make(env_name)
    expected_max = controller.adapter.config.max_action
    if not np.allclose(env.action_space.low, -expected_max) or not np.allclose(
        env.action_space.high, expected_max
    ):
        env.close()
        raise ValueError("environment action bounds differ from adapter training bounds")
    returns: List[float] = []
    lengths: List[int] = []
    residual_values: List[float] = []
    applied_residual_values: List[float] = []
    inverse_saturation_count = 0
    command_at_bound_count = 0
    action_value_count = 0
    actuator_clip_count = 0
    squared_execution_deltas: List[float] = []
    try:
        for environment_seed, noise_seed in zip(
            environment_seeds, action_noise_seeds
        ):
            observation, _ = env.reset(seed=int(environment_seed))
            rng = np.random.default_rng(int(noise_seed))
            total_return = 0.0
            for length in range(1, int(env.spec.max_episode_steps or 1000) + 1):
                command, details = controller.command(
                    observation,
                    env.action_space.low,
                    env.action_space.high,
                    use_residual=use_residual,
                )
                residual_values.extend(np.abs(details["proposed_residual"]))
                applied_residual_values.extend(np.abs(details["applied_residual"]))
                inverse_saturation_count += int(details["inverse_saturation_count"])
                command_at_bound_count += int(details["command_at_bound_count"])
                action_value_count += int(command.size)
                perturbation = rng.uniform(
                    -action_noise_beta,
                    action_noise_beta,
                    size=command.shape,
                ).astype(np.float32)
                requested = command + perturbation
                executed = np.clip(
                    requested, env.action_space.low, env.action_space.high
                ).astype(np.float32, copy=False)
                actuator_clip_count += int(
                    (
                        (requested < env.action_space.low)
                        | (requested > env.action_space.high)
                    ).sum()
                )
                squared_execution_deltas.extend(
                    np.square(executed - command).reshape(-1).tolist()
                )
                observation, reward, terminated, truncated, _ = env.step(executed)
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
        "use_residual": bool(use_residual),
        "environment_seeds": [int(value) for value in environment_seeds],
        "action_noise_seeds": [int(value) for value in action_noise_seeds],
        "action_noise_beta": float(action_noise_beta),
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
        "proposed_residual_abs_mean": float(np.mean(residual_values)),
        "proposed_residual_abs_max": float(np.max(residual_values)),
        "applied_residual_abs_mean": float(np.mean(applied_residual_values)),
        "inverse_saturation_fraction": inverse_saturation_count
        / max(action_value_count, 1),
        "command_at_bound_fraction": command_at_bound_count
        / max(action_value_count, 1),
        "actuator_clip_fraction": actuator_clip_count
        / max(action_value_count, 1),
        "executed_commanded_action_mse": float(
            np.mean(squared_execution_deltas)
        ),
    }


def paired_difference(
    adapted: Dict[str, object], baseline: Dict[str, object]
) -> Dict[str, object]:
    adapted_returns = np.asarray(adapted["returns"], dtype=np.float64)
    baseline_returns = np.asarray(baseline["returns"], dtype=np.float64)
    if adapted_returns.shape != baseline_returns.shape:
        raise ValueError("paired evaluation return arrays do not align")
    differences = adapted_returns - baseline_returns
    return {
        "adapted_minus_baseline_returns": differences.tolist(),
        "return_difference_mean": float(differences.mean()),
        "return_difference_std": float(differences.std()),
        "positive_episode_count": int((differences > 0.0).sum()),
        "episode_count": int(differences.size),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if not (
        np.isfinite(args.reference_min_score)
        and np.isfinite(args.reference_max_score)
        and args.reference_max_score > args.reference_min_score
    ):
        raise ValueError("normalization references must be finite and ordered")
    output_path = require_new_output_file(Path(args.output))
    device = torch.device(args.device)
    adapter_checkpoint = Path(args.adapter_checkpoint).resolve()
    base_override = (
        Path(args.base_checkpoint).resolve() if args.base_checkpoint else None
    )
    controller, provenance = load_controller(
        adapter_checkpoint, device, base_override
    )
    trained_beta = controller.adapter.config.execution_noise_beta
    beta = trained_beta if args.eval_action_noise_beta is None else float(
        args.eval_action_noise_beta
    )
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("evaluation action-noise beta must be finite and non-negative")
    environment_seeds = [
        args.eval_seed + index for index in range(args.eval_episodes)
    ]
    noise_seeds = [
        args.eval_noise_seed + index for index in range(args.eval_episodes)
    ]
    baseline_only = evaluate_controller(
        controller,
        args.env_name,
        environment_seeds,
        noise_seeds,
        beta,
        False,
        args.reference_min_score,
        args.reference_max_score,
    )
    adapted = evaluate_controller(
        controller,
        args.env_name,
        environment_seeds,
        noise_seeds,
        beta,
        True,
        args.reference_min_score,
        args.reference_max_score,
    )
    result: Dict[str, object] = {
        "status": "complete",
        **provenance,
        "environment": args.env_name,
        "normalization": {
            "reference_min_score": float(args.reference_min_score),
            "reference_max_score": float(args.reference_max_score),
        },
        "channel": {
            "distribution": "iid_uniform_minus_beta_plus_beta_per_step",
            "model_or_calibration_beta": float(trained_beta),
            "environment_rollout_beta": float(beta),
            "beta_mismatch": bool(beta != trained_beta),
        },
        "evaluation_protocol": {
            "paired_environment_and_action_noise_seeds": True,
            "episode_count_per_method": int(args.eval_episodes),
            "selection_rule": "all requested episodes retained",
        },
        "baseline_transform": controller.adapter.config.baseline_transform,
        "baseline_only": baseline_only,
        "adapted": adapted,
        "paired": paired_difference(adapted, baseline_only),
        "implementation": {
            "evaluate_sha256": sha256_file(Path(__file__).resolve()),
            "core_sha256": sha256_file(
                Path(__file__).resolve().with_name("inverse_residual_core.py")
            ),
            "train_sha256": sha256_file(
                Path(__file__).resolve().with_name(
                    "train_inverse_residual_adapter.py"
                )
            ),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    save_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-name", default="Walker2d-v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=9300)
    parser.add_argument("--eval-noise-seed", type=int, default=19300)
    parser.add_argument(
        "--eval-action-noise-beta",
        type=float,
        help=(
            "actual rollout-channel beta; confirmation commands should always "
            "set this explicitly, independently of the calibrated model beta"
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
