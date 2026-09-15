"""Train equal-budget TD3+BC/HUBL variants on an offline HDF5 dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from evaluation_controls import (
    COMMAND_TRANSFORMS,
    COMMAND_SATURATION_DEFINITION,
    TRANSFORM_SATURATION_DEFINITION,
    prepare_evaluation_command,
    validate_command_scale,
)
from td3bc_core import (
    ACTION_PAIRINGS,
    EXECUTION_NOISE_SCHEMES,
    EXECUTION_NOISE_TWIN_REDUCTIONS,
    TD3BCAgent,
    TD3BCConfig,
)
from train_iql import (
    _load_predictions,
    build_hubl_fields,
    evaluate,
    prepare_dataset,
    save_json,
    seed_everything,
    sha256_file,
)


SUPPORTED_VARIANTS = (
    "td3bc",
    "noise_marginalized_td3bc",
    "noise_marginalized_hubl_constant",
    "iql",
    "hubl_constant",
    "hubl_rank",
    "hubl_horizon",
    "hubl_horizon_shuffled",
    "hubl_horizon_reverse",
    "hubl_rank_horizon",
    "hubl_action_residual",
    "hubl_action_residual_shuffled",
    "cf_mean_rank",
    "cf_lcb_rank",
    "cf_prob_rank",
    "cf_mean_h",
    "cf_lcb_h",
    "cf_prob_h",
)

NOISE_MARGINALIZED_VARIANTS = {
    "noise_marginalized_td3bc": "iql",
    "noise_marginalized_hubl_constant": "hubl_constant",
}


def resolve_action_pairing(variant: str, requested: Optional[str]) -> str:
    """Resolve the action channels without permitting ambiguous NM semantics."""

    if variant in NOISE_MARGINALIZED_VARIANTS:
        if requested not in (None, "executed_commanded"):
            raise ValueError(
                f"{variant} requires --action-pairing "
                "executed_commanded"
            )
        return "executed_commanded"
    return requested or "executed_executed"


def resolve_shared_hubl_variant(variant: str) -> str:
    """Map only the two declared NM methods to the shared HUBL builder."""

    if variant in NOISE_MARGINALIZED_VARIANTS:
        return NOISE_MARGINALIZED_VARIANTS[variant]
    return "iql" if variant == "td3bc" else variant


def _aligned_commanded_actions(
    path: Path,
    *,
    max_episodes: Optional[int],
) -> Optional[np.ndarray]:
    """Load clean commands using exactly ``prepare_dataset``'s keep mask.

    The stochastic collector writes a flat D4RL-style file, but the episodic
    branch is supported as well so action-channel semantics do not depend on
    container layout.  ``None`` means the source genuinely has no command
    channel; callers may only use that fallback for executed/executed.
    """

    pieces: List[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        episode_keys = sorted(
            (key for key in handle.keys() if key.startswith("episode_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        if episode_keys:
            selected = episode_keys[:max_episodes] if max_episodes else episode_keys
            for key in selected:
                group = handle[key]
                if "clean_policy_actions" not in group:
                    return None
                commands = np.asarray(group["clean_policy_actions"], dtype=np.float32)
                terminals = np.asarray(
                    group["terminations"]
                    if "terminations" in group
                    else group["terminals"],
                    dtype=np.bool_,
                ).reshape(-1)
                timeouts = np.asarray(
                    group["truncations"]
                    if "truncations" in group
                    else group["timeouts"],
                    dtype=np.bool_,
                ).reshape(-1)
                if not (commands.shape[0] == terminals.size == timeouts.size):
                    raise ValueError(f"{key}: command/terminal arrays are misaligned")
                keep = np.ones(terminals.size, dtype=np.bool_)
                if bool(timeouts[-1]) or not bool(terminals[-1] | timeouts[-1]):
                    keep[-1] = False
                pieces.append(commands[keep])
        else:
            if "clean_policy_actions" not in handle:
                return None
            commands = np.asarray(handle["clean_policy_actions"], dtype=np.float32)
            actions = np.asarray(handle["actions"], dtype=np.float32)
            terminals = np.asarray(handle["terminals"], dtype=np.bool_).reshape(-1)
            timeouts = (
                np.asarray(handle["timeouts"], dtype=np.bool_).reshape(-1)
                if "timeouts" in handle
                else np.zeros_like(terminals)
            )
            if commands.shape != actions.shape:
                raise ValueError(
                    "clean_policy_actions and actions must have identical shape"
                )
            if not (actions.shape[0] == terminals.size == timeouts.size):
                raise ValueError("flat action/terminal arrays are misaligned")
            boundaries = np.flatnonzero(terminals | timeouts)
            stops = (boundaries + 1).tolist()
            if not stops or stops[-1] != terminals.size:
                stops.append(terminals.size)
            start = 0
            episode_count = 0
            for stop in stops:
                if stop <= start:
                    continue
                keep = np.ones(stop - start, dtype=np.bool_)
                if bool(timeouts[stop - 1]) or not bool(
                    terminals[stop - 1] | timeouts[stop - 1]
                ):
                    keep[-1] = False
                pieces.append(commands[start:stop][keep])
                start = stop
                episode_count += 1
                if max_episodes and episode_count >= max_episodes:
                    break
    if not pieces:
        raise ValueError("dataset contains no retained commanded actions")
    aligned = np.concatenate(pieces, axis=0)
    if not np.all(np.isfinite(aligned)):
        raise ValueError("clean_policy_actions contains non-finite values")
    return aligned


def load_action_channels(
    path: Path,
    executed_actions: np.ndarray,
    *,
    action_pairing: str,
    max_episodes: Optional[int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return aligned command channel plus an auditable channel manifest."""

    commanded = _aligned_commanded_actions(path, max_episodes=max_episodes)
    if commanded is None:
        if action_pairing != "executed_executed":
            raise KeyError(
                f"{action_pairing} requires HDF5 field clean_policy_actions"
            )
        commanded = executed_actions.copy()
        source = "fallback_to_executed_field_absent"
    else:
        source = "hdf5:clean_policy_actions"
    if commanded.shape != executed_actions.shape:
        raise ValueError(
            "aligned clean_policy_actions shape does not match prepared executed actions: "
            f"{commanded.shape} != {executed_actions.shape}"
        )
    delta = executed_actions.astype(np.float64) - commanded.astype(np.float64)
    manifest: Dict[str, object] = {
        "action_pairing": action_pairing,
        "executed_action_source": "hdf5:actions",
        "commanded_action_source": source,
        "critic_action_source": (
            "clean_policy_actions"
            if action_pairing == "commanded_commanded"
            else "actions"
        ),
        "behavior_cloning_action_source": (
            "clean_policy_actions"
            if action_pairing in ("commanded_commanded", "executed_commanded")
            else "actions"
        ),
        "transition_count": int(executed_actions.shape[0]),
        "executed_commanded_mse": float(np.square(delta).mean()),
        "executed_commanded_mean_absolute_delta": float(np.abs(delta).mean()),
        "executed_commanded_max_absolute_delta": float(np.abs(delta).max()),
        "executed_commanded_identical_fraction": float(np.all(delta == 0.0, axis=1).mean()),
    }
    return commanded.astype(np.float32, copy=False), manifest


def evaluate_with_actuator_noise(
    agent: TD3BCAgent,
    env_name: str,
    environment_seeds: Sequence[int],
    action_noise_seeds: Sequence[int],
    beta: float,
    reference_min: float,
    reference_max: float,
    command_scale: float = 1.0,
    command_transform: str = "identity",
) -> Dict[str, object]:
    """Evaluate scaled/clipped commands with persistent actuator noise."""

    import gymnasium as gym

    if beta < 0.0:
        raise ValueError("evaluation action-noise beta must be non-negative")
    command_scale = validate_command_scale(command_scale)
    if command_transform not in COMMAND_TRANSFORMS:
        raise ValueError(
            f"command_transform must be one of {COMMAND_TRANSFORMS}, got "
            f"{command_transform!r}"
        )
    if len(environment_seeds) != len(action_noise_seeds):
        raise ValueError("environment and action-noise seed lists must align")
    env = gym.make(env_name)
    returns: List[float] = []
    lengths: List[int] = []
    clipped_value_count = 0
    action_value_count = 0
    command_saturation_count = 0
    command_at_bound_count = 0
    transform_saturation_count = 0
    squared_deltas: List[float] = []
    for environment_seed, noise_seed in zip(environment_seeds, action_noise_seeds):
        observation, _ = env.reset(seed=int(environment_seed))
        rng = np.random.default_rng(int(noise_seed))
        total_return = 0.0
        for length in range(1, int(env.spec.max_episode_steps or 1000) + 1):
            raw_command = agent.act(
                observation, env.action_space.low, env.action_space.high
            )
            command, command_audit = prepare_evaluation_command(
                raw_command,
                env.action_space.low,
                env.action_space.high,
                command_scale=command_scale,
                command_transform=command_transform,
                actuator_noise_beta=beta,
            )
            command_saturation_count += command_audit[
                "scaled_command_out_of_bounds_count"
            ]
            command_at_bound_count += command_audit[
                "transformed_command_at_bound_count"
            ]
            transform_saturation_count += command_audit[
                "command_transform_saturation_count"
            ]
            perturbation = rng.uniform(-beta, beta, size=command.shape).astype(
                np.float32
            )
            requested = command + perturbation
            executed = np.clip(
                requested, env.action_space.low, env.action_space.high
            ).astype(np.float32, copy=False)
            clipped_value_count += int(
                ((requested < env.action_space.low) | (requested > env.action_space.high)).sum()
            )
            action_value_count += int(command.size)
            squared_deltas.extend(np.square(executed - command).reshape(-1).tolist())
            observation, reward, terminated, truncated, _ = env.step(executed)
            total_return += float(reward)
            if terminated or truncated:
                break
        returns.append(total_return)
        lengths.append(length)
    env.close()
    values = np.asarray(returns, dtype=np.float64)
    normalized = 100.0 * (values - reference_min) / (reference_max - reference_min)
    return {
        "environment_seeds": [int(seed) for seed in environment_seeds],
        "action_noise_seeds": [int(seed) for seed in action_noise_seeds],
        "action_noise_beta": float(beta),
        "action_noise_distribution": "iid_uniform_minus_beta_plus_beta_per_step",
        "command_scale": float(command_scale),
        "command_saturation_fraction": command_saturation_count
        / max(action_value_count, 1),
        "command_saturation_definition": COMMAND_SATURATION_DEFINITION,
        "command_transform": command_transform,
        "command_transform_beta": float(beta),
        "command_transform_saturation_fraction": transform_saturation_count
        / max(action_value_count, 1),
        "command_transform_saturation_definition": TRANSFORM_SATURATION_DEFINITION,
        "transformed_command_at_bound_fraction": command_at_bound_count
        / max(action_value_count, 1),
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
        "action_clip_fraction": clipped_value_count / max(action_value_count, 1),
        "executed_commanded_action_mse": float(np.mean(squared_deltas)),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.train_seed)
    torch.set_num_threads(args.torch_threads)

    effective_action_pairing = resolve_action_pairing(
        args.variant, args.action_pairing
    )
    noise_marginalized = args.variant in NOISE_MARGINALIZED_VARIANTS
    dataset_path = Path(args.dataset).resolve()
    dataset = prepare_dataset(
        dataset_path,
        discount=args.discount,
        noise_seed=args.noise_seed,
        iid_noise_scale=args.iid_noise_scale,
        episode_noise_scale=args.episode_noise_scale,
        max_episodes=args.max_episodes,
    )
    commanded_actions, action_channel_manifest = load_action_channels(
        dataset_path,
        dataset.actions,
        action_pairing=effective_action_pairing,
        max_episodes=args.max_episodes,
    )
    predictions = _load_predictions(
        Path(args.predictions).resolve() if args.predictions else None,
        dataset.noisy_episode_scores.size,
        dataset.observations.shape[0],
    )
    hubl_variant = resolve_shared_hubl_variant(args.variant)
    heuristic, lambdas, mechanism_metrics = build_hubl_fields(
        dataset,
        hubl_variant,
        args.heuristic_discount,
        args.lcb_kappa,
        predictions,
        args.horizon_noise_scale,
        args.horizon_control_seed,
    )

    observation_mean = dataset.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    # TD3+BC uses dataset std plus 1e-3, including dimensions with near-zero spread.
    observation_std = (
        dataset.observations.std(axis=0, dtype=np.float64).astype(np.float32) + 1e-3
    )
    device = torch.device(args.device)
    tensors = {
        "observations": torch.as_tensor(
            dataset.observations, dtype=torch.float32, device=device
        ),
        "executed_actions": torch.as_tensor(
            dataset.actions, dtype=torch.float32, device=device
        ),
        "commanded_actions": torch.as_tensor(
            commanded_actions, dtype=torch.float32, device=device
        ),
        "next_observations": torch.as_tensor(
            dataset.next_observations, dtype=torch.float32, device=device
        ),
        "rewards": torch.as_tensor(
            dataset.noisy_rewards, dtype=torch.float32, device=device
        ),
        "terminals": torch.as_tensor(dataset.terminals, dtype=torch.float32, device=device),
        "heuristic_next": torch.as_tensor(heuristic, dtype=torch.float32, device=device),
        "lambdas": torch.as_tensor(lambdas, dtype=torch.float32, device=device),
    }
    config = TD3BCConfig(
        observation_dim=dataset.observations.shape[1],
        action_dim=dataset.actions.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        max_action=args.max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_frequency=args.policy_frequency,
        alpha=args.alpha,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        action_pairing=effective_action_pairing,
        marginalize_execution_noise=noise_marginalized,
        execution_noise_beta=(
            args.execution_noise_beta if noise_marginalized else 0.0
        ),
        execution_noise_samples=(
            args.execution_noise_samples if noise_marginalized else 1
        ),
        execution_noise_seed=args.execution_noise_seed,
        execution_noise_scheme=(
            args.execution_noise_scheme
            if noise_marginalized
            else "fixed_antithetic"
        ),
        execution_noise_twin_reduction=args.execution_noise_twin_reduction,
    )
    agent = TD3BCAgent(config, device, observation_mean, observation_std)
    execution_noise_metadata = agent.execution_noise_metadata()
    action_channel_manifest["requested_action_pairing"] = args.action_pairing
    config_payload = {
        "arguments": vars(args),
        "algorithm": "TD3+BC",
        "implementation": {
            "train_td3bc_sha256": sha256_file(Path(__file__).resolve()),
            "td3bc_core_sha256": sha256_file(
                Path(__file__).resolve().with_name("td3bc_core.py")
            ),
        },
        "hubl_variant_used_by_shared_builder": hubl_variant,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "schema": dataset.source_schema,
            "episodes": int(dataset.noisy_episode_scores.size),
            "kept_transitions": int(dataset.observations.shape[0]),
            "reward_scale": dataset.reward_scale,
            "noise": dataset.noise_metadata,
            "action_residual": dataset.action_residual_metadata,
            "observation_mean": observation_mean.tolist(),
            "observation_std": observation_std.tolist(),
        },
        "td3_bc": asdict(config),
        "action_channels": action_channel_manifest,
        "execution_noise_marginalization": execution_noise_metadata,
        "evaluation_protocol": {
            "clean": True,
            "persistent_action_noise": {
                "beta": args.eval_action_noise_beta,
                "distribution": "iid_uniform_minus_beta_plus_beta_per_step",
                "environment_seed_start": args.eval_seed,
                "noise_seed_start": args.eval_noise_seed,
                "episodes": args.eval_episodes,
            },
        },
        "mechanism_metrics_before_training": mechanism_metrics,
        "selection_rule": "all requested evaluation episodes; no run exclusion",
    }
    save_json(output_dir / "config.json", config_payload)
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.train_seed + 1729)
    n = dataset.observations.shape[0]
    last_metrics: Dict[str, float] = {}
    evaluations: List[Dict[str, object]] = []
    training_seconds = 0.0
    for step in range(1, args.updates + 1):
        update_started = time.perf_counter()
        indices = torch.randint(
            0, n, (args.batch_size,), generator=generator, device=device
        )
        batch = {key: value[indices] for key, value in tensors.items()}
        last_metrics = agent.update(batch)
        training_seconds += time.perf_counter() - update_started
        if step == 1 or step % args.log_period == 0:
            event = {
                "event": "train",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "optimizer_seconds": training_seconds,
                "updates_per_optimizer_second": step / max(training_seconds, 1e-12),
                **last_metrics,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        if step % args.eval_period == 0 or step == args.updates:
            eval_seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
            noise_seeds = [
                args.eval_noise_seed + index for index in range(args.eval_episodes)
            ]
            clean_metrics = evaluate(
                agent,
                args.env_name,
                eval_seeds,
                args.reference_min_score,
                args.reference_max_score,
            )
            noisy_metrics = evaluate_with_actuator_noise(
                agent,
                args.env_name,
                eval_seeds,
                noise_seeds,
                args.eval_action_noise_beta,
                args.reference_min_score,
                args.reference_max_score,
            )
            evaluation = {
                "event": "evaluation",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                "clean": clean_metrics,
                "persistent_action_noise": noisy_metrics,
            }
            evaluations.append(evaluation)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation, sort_keys=True) + "\n")
            torch.save(
                {"step": step, "agent": agent.checkpoint(), "config": config_payload},
                output_dir / "latest.pt",
            )

    wall_time = time.perf_counter() - started
    actor_update_count = args.updates // args.policy_frequency
    execution_noise_cost = {
        **agent.execution_noise_metadata(),
        "training_target_twin_q_input_rows": int(
            args.updates * args.batch_size * config.execution_noise_samples
        ),
        "training_actor_q1_input_rows": int(
            actor_update_count
            * args.batch_size
            * config.execution_noise_samples
        ),
        "actor_update_count": int(actor_update_count),
        "q_row_multiplier_vs_single_sample": int(
            config.execution_noise_samples
        ),
    }
    summary = {
        "status": "complete",
        "algorithm": "TD3+BC",
        "variant": args.variant,
        "action_pairing": effective_action_pairing,
        "train_seed": args.train_seed,
        "updates": args.updates,
        "wall_time_seconds": wall_time,
        "optimizer_seconds": training_seconds,
        "updates_per_optimizer_second": args.updates / max(training_seconds, 1e-12),
        "final_train_metrics": last_metrics,
        "evaluations": evaluations,
        "final_evaluation": evaluations[-1],
        "mechanism_metrics_before_training": mechanism_metrics,
        "action_channels": action_channel_manifest,
        "execution_noise_marginalization": execution_noise_cost,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        required=True,
        help=(
            "hubl_horizon uses horizon reliability; hubl_horizon_shuffled "
            "globally permutes those weights; hubl_horizon_reverse reverses "
            "them within episodes; hubl_rank_horizon multiplies trajectory "
            "quality rank by horizon reliability; hubl_action_residual uses "
            "the paired executed/commanded next-suffix residual energy and its "
            "shuffled variant is an equal-lambda-multiset control"
        ),
    )
    parser.add_argument(
        "--action-pairing",
        choices=ACTION_PAIRINGS,
        default=None,
        help=(
            "critic/BC action channels: standard executed/executed, naive "
            "commanded/commanded, or dual executed/commanded; defaults to "
            "executed/commanded for noise_marginalized_td3bc and "
            "noise_marginalized_hubl_constant, and executed/executed otherwise"
        ),
    )
    parser.add_argument(
        "--execution-noise-beta",
        type=float,
        default=1.0,
        help=(
            "known iid Uniform[-beta,beta] actuator channel integrated by "
            "noise_marginalized_td3bc"
        ),
    )
    parser.add_argument(
        "--execution-noise-samples",
        type=int,
        default=2,
        help="number K of fixed execution-noise quadrature points",
    )
    parser.add_argument(
        "--execution-noise-seed",
        type=int,
        default=314159,
        help="seed used once to construct the frozen antithetic points",
    )
    parser.add_argument(
        "--execution-noise-scheme",
        choices=EXECUTION_NOISE_SCHEMES,
        default="resampled_antithetic",
    )
    parser.add_argument(
        "--execution-noise-twin-reduction",
        choices=EXECUTION_NOISE_TWIN_REDUCTIONS,
        default="min_of_expectations",
        help=(
            "min_of_expectations first constructs each critic's command "
            "value; expectation_of_min is an extra-pessimism ablation"
        ),
    )
    parser.add_argument("--predictions")
    parser.add_argument("--env-name", default="Walker2d-v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=7001)
    parser.add_argument("--iid-noise-scale", type=float, default=0.0)
    parser.add_argument("--episode-noise-scale", type=float, default=0.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--updates", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--max-action", type=float, default=1.0)
    parser.add_argument("--heuristic-discount", type=float, default=1.0)
    parser.add_argument(
        "--horizon-noise-scale",
        type=float,
        default=0.02,
        help=(
            "c in lambda_t=alpha/(1+c*sum_{j<h}gamma^(2j)); "
            "used by all hubl_*horizon variants and as c for "
            "hubl_action_residual variants"
        ),
    )
    parser.add_argument(
        "--horizon-control-seed",
        type=int,
        default=104729,
        help=(
            "independent seed for the fixed global lambda permutation used "
            "by the fixed global permutation controls"
        ),
    )
    parser.add_argument("--lcb-kappa", type=float, default=0.5)
    parser.add_argument("--eval-period", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=9000)
    parser.add_argument("--eval-noise-seed", type=int, default=19000)
    parser.add_argument("--eval-action-noise-beta", type=float, default=1.0)
    parser.add_argument("--log-period", type=int, default=1_000)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--reference-min-score", type=float, default=1.629008)
    parser.add_argument("--reference-max-score", type=float, default=4592.3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary["final_evaluation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
