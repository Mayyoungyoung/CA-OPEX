"""Paired closed-loop evaluation for a completed CC-FQE/RPI checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cc_fqe_rpi_core import CCFQERPIAgent, CCFQERPIConfig
from train_cc_fqe_rpi import CHECKPOINT_FORMAT, atomic_json, sha256_file


def resolve_evaluation_beta(
    model_beta: float, requested_beta: Optional[float]
) -> float:
    """Resolve the real rollout channel independently of the calibrated model."""

    beta = float(model_beta if requested_beta is None else requested_beta)
    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("evaluation beta must be finite and non-negative")
    return beta


def load_agent(path: Path, device: torch.device) -> Tuple[CCFQERPIAgent, Dict[str, object]]:
    # Loading on CPU preserves serialized generator-state tensor semantics;
    # load_state_dict copies network weights to the requested device below.
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported CC-FQE/RPI training checkpoint")
    agent_payload = payload.get("agent")
    if not isinstance(agent_payload, dict):
        raise ValueError("checkpoint lacks agent payload")
    config = CCFQERPIConfig(**agent_payload["config"])
    if int(agent_payload.get("fqe_steps", -1)) != config.fqe_updates or int(
        agent_payload.get("adapter_steps", -1)
    ) != config.adapter_updates:
        raise ValueError("evaluation requires the fixed completed checkpoint")
    mean = agent_payload["observation_mean"]
    std = agent_payload["observation_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    agent = CCFQERPIAgent(
        config,
        device,
        np.asarray(mean, dtype=np.float32),
        np.asarray(std, dtype=np.float32),
        agent_payload["base_actor"],
        agent_payload["critic"],
    )
    agent.restore(agent_payload)
    if agent.stage != "complete":
        raise ValueError("restored checkpoint is not complete")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint lacks provenance")
    return agent, {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "provenance_fingerprint": payload.get("provenance_fingerprint"),
        "training_provenance": provenance,
    }


def evaluate_condition(
    agent: CCFQERPIAgent,
    *,
    env_name: str,
    environment_seeds: Sequence[int],
    noise_seeds: Sequence[int],
    beta: float,
    use_residual: bool,
    reference_min: float,
    reference_max: float,
) -> Dict[str, object]:
    import gymnasium as gym

    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("evaluation beta must be finite and non-negative")
    if len(environment_seeds) != len(noise_seeds) or not environment_seeds:
        raise ValueError("nonempty environment/noise seed arrays must align")
    env = gym.make(env_name)
    if not np.allclose(env.action_space.low, -agent.config.max_action) or not np.allclose(
        env.action_space.high, agent.config.max_action
    ):
        env.close()
        raise ValueError("environment bounds differ from checkpoint bounds")
    returns: List[float] = []
    lengths: List[int] = []
    proposed_residual_abs: List[float] = []
    applied_residual_abs: List[float] = []
    inverse_saturations = 0
    command_at_bound = 0
    actuator_clips = 0
    action_values = 0
    try:
        for environment_seed, noise_seed in zip(environment_seeds, noise_seeds):
            observation, _ = env.reset(seed=int(environment_seed))
            rng = np.random.default_rng(int(noise_seed))
            total_return = 0.0
            for length in range(1, int(env.spec.max_episode_steps or 1000) + 1):
                command, details = agent.command(
                    observation, use_residual=use_residual
                )
                proposed_residual_abs.extend(
                    np.abs(np.asarray(details["proposed_residual"])).tolist()
                )
                applied_residual_abs.extend(
                    np.abs(np.asarray(details["applied_residual"])).tolist()
                )
                inverse_saturations += int(details["inverse_saturation_count"])
                command_at_bound += int(
                    ((command <= env.action_space.low) | (command >= env.action_space.high)).sum()
                )
                perturbation = rng.uniform(-beta, beta, size=command.shape).astype(
                    np.float32
                )
                requested = command + perturbation
                executed = np.clip(
                    requested, env.action_space.low, env.action_space.high
                ).astype(np.float32, copy=False)
                actuator_clips += int(
                    ((requested < env.action_space.low) | (requested > env.action_space.high)).sum()
                )
                action_values += int(command.size)
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
        "action_noise_seeds": [int(value) for value in noise_seeds],
        "action_noise_beta": float(beta),
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
        "proposed_residual_abs_mean": float(np.mean(proposed_residual_abs)),
        "applied_residual_abs_mean": float(np.mean(applied_residual_abs)),
        "inverse_saturation_fraction": inverse_saturations / max(action_values, 1),
        "command_at_bound_fraction": command_at_bound / max(action_values, 1),
        "actuator_clip_fraction": actuator_clips / max(action_values, 1),
    }


def paired_difference(
    adapted: Mapping[str, object], inverse_only: Mapping[str, object]
) -> Dict[str, object]:
    adapted_values = np.asarray(adapted["returns"], dtype=np.float64)
    baseline_values = np.asarray(inverse_only["returns"], dtype=np.float64)
    if adapted_values.shape != baseline_values.shape:
        raise ValueError("paired return arrays do not align")
    difference = adapted_values - baseline_values
    return {
        "adapted_minus_inverse_returns": difference.tolist(),
        "return_difference_mean": float(difference.mean()),
        "return_difference_std": float(difference.std()),
        "positive_episode_count": int((difference > 0.0).sum()),
        "episode_count": int(difference.size),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    checkpoint_path = Path(args.checkpoint).resolve()
    agent, provenance = load_agent(checkpoint_path, torch.device(args.device))
    beta = resolve_evaluation_beta(
        agent.config.target_beta, args.action_noise_beta
    )
    environment_seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
    noise_seeds = [args.eval_noise_seed + index for index in range(args.eval_episodes)]
    inverse_only = evaluate_condition(
        agent,
        env_name=args.env_name,
        environment_seeds=environment_seeds,
        noise_seeds=noise_seeds,
        beta=beta,
        use_residual=False,
        reference_min=args.reference_min_score,
        reference_max=args.reference_max_score,
    )
    adapted = evaluate_condition(
        agent,
        env_name=args.env_name,
        environment_seeds=environment_seeds,
        noise_seeds=noise_seeds,
        beta=beta,
        use_residual=True,
        reference_min=args.reference_min_score,
        reference_max=args.reference_max_score,
    )
    result: Dict[str, object] = {
        "status": "complete",
        **provenance,
        "environment": args.env_name,
        "channel": {
            "distribution": "iid_uniform_minus_beta_plus_beta_per_step",
            "model_or_calibration_beta": float(agent.config.target_beta),
            "environment_rollout_beta": beta,
            "beta_mismatch": bool(beta != agent.config.target_beta),
        },
        "protocol": {
            "paired_environment_and_action_noise_seeds": True,
            "episodes_per_method": args.eval_episodes,
            "selection_rule": "all requested episodes retained",
        },
        "inverse_only": inverse_only,
        "adapted": adapted,
        "paired": paired_difference(adapted, inverse_only),
        "wall_time_seconds": time.perf_counter() - started,
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(Path(args.output).resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-name", default="Walker2d-v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=9300)
    parser.add_argument("--eval-noise-seed", type=int, default=19300)
    parser.add_argument("--action-noise-beta", type=float)
    parser.add_argument("--reference-min-score", type=float, default=1.629008)
    parser.add_argument("--reference-max-score", type=float, default=4592.3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
