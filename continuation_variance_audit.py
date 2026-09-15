"""Audit whether continuation-return variance follows a geometric horizon law.

The action-noise dataset stores the complete MuJoCo ``qpos``/``qvel`` state at
every pre-action observation.  This script restores stratified logged states
and samples *closed-loop* continuations from the frozen embedded SAC behavior
policy.  It never replays clean actions open loop.  Beta-zero and beta-one
rollouts use paired policy/noise random-number streams.

The scientific unit is a restored anchor state.  Replicates estimate the
conditional return distribution at that anchor; they are not counted as
independent dataset samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


DEFAULT_HORIZONS = (1, 5, 10, 25, 50, 100, 200)
DEFAULT_SUFFIX_BIN_STARTS = (1, 9, 33, 65, 129, 257)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def episode_coordinates(
    terminals: np.ndarray, timeouts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return episode id, zero-based step, and inclusive logged suffix length."""

    terminals = np.asarray(terminals, dtype=np.bool_).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=np.bool_).reshape(-1)
    if terminals.shape != timeouts.shape:
        raise ValueError("terminals and timeouts must align")
    n = terminals.size
    episode_ids = np.empty(n, dtype=np.int64)
    steps = np.empty(n, dtype=np.int32)
    suffixes = np.empty(n, dtype=np.int32)
    ends = (np.flatnonzero(terminals | timeouts) + 1).tolist()
    if not ends or ends[-1] != n:
        ends.append(n)
    start = 0
    for episode_id, stop in enumerate(ends):
        if stop <= start:
            continue
        length = stop - start
        episode_ids[start:stop] = episode_id
        steps[start:stop] = np.arange(length, dtype=np.int32)
        suffixes[start:stop] = np.arange(length, 0, -1, dtype=np.int32)
        start = stop
    return episode_ids, steps, suffixes


def suffix_bin_labels(starts: Sequence[int]) -> List[str]:
    starts = tuple(int(value) for value in starts)
    if not starts or starts[0] != 1 or any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError("suffix bin starts must be strictly increasing and begin at 1")
    return [
        f"{lower}-{starts[index + 1] - 1}" if index + 1 < len(starts) else f"{lower}+"
        for index, lower in enumerate(starts)
    ]


def stratified_anchor_indices(
    suffixes: np.ndarray,
    episode_steps: np.ndarray,
    *,
    bin_starts: Sequence[int],
    anchors_per_bin: int,
    maximum_episode_steps: int,
    maximum_rollout_horizon: int,
    seed: int,
    eligible_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Select a fixed number of eligible rows in each realized-suffix stratum."""

    if anchors_per_bin < 1:
        raise ValueError("anchors_per_bin must be positive")
    suffixes = np.asarray(suffixes, dtype=np.int32).reshape(-1)
    episode_steps = np.asarray(episode_steps, dtype=np.int32).reshape(-1)
    if suffixes.shape != episode_steps.shape:
        raise ValueError("suffixes and episode_steps must align")
    starts = np.asarray(tuple(bin_starts), dtype=np.int32)
    labels = suffix_bin_labels(starts)
    # The wrapper truncates after maximum_episode_steps actions.  At a state
    # before action `episode_step`, at least H actions remain iff step <= M-H.
    eligible_time = episode_steps <= maximum_episode_steps - maximum_rollout_horizon
    if eligible_mask is not None:
        eligible_mask = np.asarray(eligible_mask, dtype=np.bool_).reshape(-1)
        if eligible_mask.shape != suffixes.shape:
            raise ValueError("eligible_mask must align with suffixes")
        eligible_time &= eligible_mask
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    selected_labels: List[str] = []
    for bin_index, label in enumerate(labels):
        lower = starts[bin_index]
        upper = starts[bin_index + 1] if bin_index + 1 < starts.size else None
        mask = eligible_time & (suffixes >= lower)
        if upper is not None:
            mask &= suffixes < upper
        candidates = np.flatnonzero(mask)
        if candidates.size < anchors_per_bin:
            raise ValueError(
                f"suffix bin {label} has {candidates.size} eligible rows, "
                f"need {anchors_per_bin}"
            )
        chosen = np.sort(rng.choice(candidates, size=anchors_per_bin, replace=False))
        selected.extend(int(value) for value in chosen)
        selected_labels.extend([label] * anchors_per_bin)
    return np.asarray(selected, dtype=np.int64), selected_labels


def discounted_prefix_returns(
    rewards: np.ndarray, horizons: Sequence[int], discount: float
) -> np.ndarray:
    """Discounted returns at requested horizons, padding termination with zeros."""

    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    horizons = np.asarray(tuple(horizons), dtype=np.int32)
    if horizons.size == 0 or np.any(horizons < 1) or np.any(np.diff(horizons) <= 0):
        raise ValueError("horizons must be positive and strictly increasing")
    weighted = rewards * np.power(float(discount), np.arange(rewards.size))
    cumulative = np.cumsum(weighted)
    output = np.zeros(horizons.size, dtype=np.float64)
    for index, horizon in enumerate(horizons):
        available = min(int(horizon), rewards.size)
        if available:
            output[index] = cumulative[available - 1]
    return output


def effective_horizons(horizons: Sequence[int], discount: float) -> np.ndarray:
    horizons = np.asarray(tuple(horizons), dtype=np.float64)
    if discount == 1.0:
        return horizons
    squared = float(discount) ** 2
    return (1.0 - np.power(squared, horizons)) / (1.0 - squared)


def fit_geometric_curve(
    anchor_variances: np.ndarray, horizons: Sequence[int], discount: float
) -> Dict[str, Any]:
    """Fit mean conditional variance to a*sum_j gamma^(2j), through the origin."""

    values = np.asarray(anchor_variances, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("anchor_variances must be anchor by horizon")
    design = effective_horizons(horizons, discount)
    if values.shape[1] != design.size:
        raise ValueError("variance columns must align with horizons")
    mean_variance = np.nanmean(values, axis=0)
    scale = float(np.dot(design, mean_variance) / np.dot(design, design))
    prediction = scale * design
    residual_sum = float(np.square(mean_variance - prediction).sum())
    total_sum = float(np.square(mean_variance - mean_variance.mean()).sum())
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else None
    denominator = max(float(mean_variance[-1]), np.finfo(np.float64).eps)
    normalized_rmse = float(np.sqrt(np.mean(np.square(mean_variance - prediction))) / denominator)
    monotone_fraction = float(np.mean(np.diff(mean_variance) >= 0.0))
    correlation = (
        float(np.corrcoef(design, mean_variance)[0, 1])
        if np.std(mean_variance) > 0
        else None
    )
    return {
        "effective_horizons": design.tolist(),
        "mean_anchor_variances": mean_variance.tolist(),
        "fitted_scale": scale,
        "fitted_values": prediction.tolist(),
        "r_squared": r_squared,
        "normalized_rmse_at_max_horizon_scale": normalized_rmse,
        "monotone_increment_fraction": monotone_fraction,
        "pearson_correlation_with_effective_horizon": correlation,
    }


def bootstrap_geometric_fit(
    anchor_variances: np.ndarray,
    horizons: Sequence[int],
    discount: float,
    *,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Anchor bootstrap intervals for the geometric-fit summary."""

    values = np.asarray(anchor_variances, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("bootstrap requires at least two anchor rows")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    keys = (
        "fitted_scale",
        "r_squared",
        "normalized_rmse_at_max_horizon_scale",
        "monotone_increment_fraction",
        "pearson_correlation_with_effective_horizon",
    )
    samples: Dict[str, List[float]] = {key: [] for key in keys}
    for _ in range(resamples):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        fit = fit_geometric_curve(values[indices], horizons, discount)
        for key in keys:
            if fit[key] is not None and np.isfinite(fit[key]):
                samples[key].append(float(fit[key]))
    return {
        "unit": "anchor",
        "resamples": resamples,
        "seed": seed,
        "percentiles": [2.5, 50.0, 97.5],
        "intervals": {
            key: np.percentile(observations, [2.5, 50.0, 97.5]).tolist()
            if observations
            else None
            for key, observations in samples.items()
        },
    }


def covariance_diagnostics(
    reward_matrices: Sequence[np.ndarray], horizons: Sequence[int], discount: float
) -> Dict[str, Any]:
    """Pool within-anchor reward covariance and quantify ignored correlations."""

    if not reward_matrices:
        raise ValueError("at least one reward matrix is required")
    maximum = max(int(value) for value in horizons)
    scatter = np.zeros((maximum, maximum), dtype=np.float64)
    degrees = 0
    for matrix in reward_matrices:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != maximum:
            raise ValueError("each reward matrix must be replicate by maximum horizon")
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scatter += centered.T @ centered
        degrees += max(matrix.shape[0] - 1, 0)
    if degrees < 1:
        raise ValueError("at least two replicates per anchor are required")
    covariance = scatter / degrees
    rows = []
    for horizon in horizons:
        horizon = int(horizon)
        block = covariance[:horizon, :horizon]
        weights = np.power(float(discount), np.arange(horizon))
        total = float(weights @ block @ weights)
        diagonal = float(np.dot(np.square(weights), np.diag(block)))
        off_diagonal = total - diagonal
        diagonal_values = np.diag(block)
        diagonal_cv = (
            float(diagonal_values.std() / diagonal_values.mean())
            if diagonal_values.mean() > 0
            else None
        )
        rows.append(
            {
                "horizon": horizon,
                "return_variance_from_covariance": total,
                "diagonal_contribution": diagonal,
                "off_diagonal_contribution": off_diagonal,
                "off_diagonal_fraction": off_diagonal / total if total > 0 else None,
                "reward_variance_diagonal_cv": diagonal_cv,
            }
        )
    return {
        "pooling": "within-anchor centered reward covariance; denominator=sum(R_i-1)",
        "degrees_of_freedom": degrees,
        "by_horizon": rows,
    }


def _set_time_limit_elapsed_step(env: Any, elapsed_step: int) -> str:
    """Set the unique Gymnasium TimeLimit wrapper to the logged episode step."""

    current = env
    visited = set()
    matches = []
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "_elapsed_steps") and hasattr(current, "_max_episode_steps"):
            matches.append(current)
        if not hasattr(current, "env"):
            break
        current = current.env
    if len(matches) != 1:
        raise RuntimeError(f"expected one TimeLimit wrapper, found {len(matches)}")
    matches[0]._elapsed_steps = int(elapsed_step)
    return type(matches[0]).__name__


def _restore_anchor(
    env: Any,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    elapsed_step: int,
    reset_seed: int,
) -> Tuple[np.ndarray, str]:
    env.reset(seed=int(reset_seed))
    wrapper_name = _set_time_limit_elapsed_step(env, elapsed_step)
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "set_state") or not hasattr(unwrapped, "_get_obs"):
        raise TypeError("environment lacks MuJoCo set_state/_get_obs interface")
    unwrapped.set_state(
        np.asarray(qpos, dtype=np.float64).copy(),
        np.asarray(qvel, dtype=np.float64).copy(),
    )
    observation = np.asarray(unwrapped._get_obs(), dtype=np.float64).reshape(-1)
    return observation, wrapper_name


def _validate_logged_step(
    env: Any,
    *,
    anchor: Dict[str, Any],
    reset_seed: int,
) -> Dict[str, Any]:
    restored, wrapper_name = _restore_anchor(
        env,
        qpos=anchor["qpos"],
        qvel=anchor["qvel"],
        elapsed_step=anchor["episode_step"],
        reset_seed=reset_seed,
    )
    observation_error = float(np.max(np.abs(restored - anchor["observation"])))
    following, reward, terminated, truncated, _ = env.step(anchor["action"])
    next_error = float(
        np.max(
            np.abs(
                np.asarray(following, dtype=np.float64).reshape(-1)
                - anchor["next_observation"]
            )
        )
    )
    return {
        "row": int(anchor["row"]),
        "time_limit_wrapper": wrapper_name,
        "restored_observation_max_abs_error": observation_error,
        "logged_next_observation_max_abs_error": next_error,
        "logged_reward_abs_error": abs(float(reward) - float(anchor["reward"])),
        "actual_terminated": bool(terminated),
        "expected_terminated": bool(anchor["terminal"]),
        "actual_truncated": bool(truncated),
        "expected_truncated": bool(anchor["timeout"]),
    }


def _paired_seed(base: int, anchor_ordinal: int, replicate: int, stream: int) -> int:
    # Keep seeds within the signed 63-bit range accepted by Torch.
    sequence = np.random.SeedSequence([base, anchor_ordinal, replicate, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))


def _rollout(
    env: Any,
    policy: Any,
    *,
    anchor: Dict[str, Any],
    beta: float,
    horizons: Sequence[int],
    discount: float,
    policy_seed: int,
    noise_seed: int,
    reset_seed: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int, bool, bool]:
    import torch

    observation, _ = _restore_anchor(
        env,
        qpos=anchor["qpos"],
        qvel=anchor["qvel"],
        elapsed_step=anchor["episode_step"],
        reset_seed=reset_seed,
    )
    generator = torch.Generator(device=torch.device(device).type)
    generator.manual_seed(int(policy_seed))
    noise_rng = np.random.default_rng(int(noise_seed))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    maximum = max(int(value) for value in horizons)
    rewards = np.zeros(maximum, dtype=np.float64)
    terminated = truncated = False
    length = 0
    for step in range(maximum):
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            clean, _ = policy.sample_action(tensor, generator)
        clean_action = clean.cpu().numpy().astype(np.float32, copy=False)
        # Draw in beta-zero runs too, so both conditions consume identical RNG.
        epsilon = noise_rng.uniform(-1.0, 1.0, size=clean_action.shape).astype(np.float32)
        executed = np.clip(clean_action + np.float32(beta) * epsilon, action_low, action_high)
        observation, reward, terminated, truncated, _ = env.step(executed)
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        rewards[step] = float(reward)
        length = step + 1
        if terminated or truncated:
            break
    returns = discounted_prefix_returns(rewards[:length], horizons, discount)
    return returns, rewards, length, bool(terminated), bool(truncated)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from behavior_policy import D4RLTanhGaussianPolicy

    started = time.perf_counter()
    dataset_path = args.dataset.resolve()
    policy_path = args.policy_source.resolve()
    horizons = tuple(int(value) for value in args.horizons)
    if tuple(sorted(set(horizons))) != horizons or horizons[0] < 1:
        raise ValueError("--horizons must be unique, increasing, and positive")
    if tuple(float(value) for value in args.betas) != (0.0, 1.0):
        raise ValueError("this paired audit requires --betas 0 1")
    if args.replicates < 2:
        raise ValueError("--replicates must be at least two")

    with h5py.File(dataset_path, "r") as handle:
        required = (
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
            "timeouts",
            "infos/qpos",
            "infos/qvel",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"dataset missing fields: {missing}")
        terminals = np.asarray(handle["terminals"], dtype=np.bool_)
        timeouts = np.asarray(handle["timeouts"], dtype=np.bool_)
        collector_truncations = (
            np.asarray(handle["collector_truncations"], dtype=np.bool_)
            if "collector_truncations" in handle
            else np.zeros_like(timeouts)
        )
        episode_ids, episode_steps, suffixes = episode_coordinates(terminals, timeouts)

        import gymnasium as gym

        probe = gym.make(args.env)
        maximum_episode_steps = int(probe.spec.max_episode_steps or 1000)
        probe.close()
        rows, labels = stratified_anchor_indices(
            suffixes,
            episode_steps,
            bin_starts=args.suffix_bin_starts,
            anchors_per_bin=args.anchors_per_bin,
            maximum_episode_steps=maximum_episode_steps,
            maximum_rollout_horizon=max(horizons),
            seed=args.selection_seed,
            eligible_mask=~collector_truncations,
        )
        anchors = []
        for ordinal, (row, label) in enumerate(zip(rows, labels)):
            anchors.append(
                {
                    "ordinal": ordinal,
                    "row": int(row),
                    "suffix_bin": label,
                    "episode_id": int(episode_ids[row]),
                    "episode_step": int(episode_steps[row]),
                    "logged_suffix_length": int(suffixes[row]),
                    "observation": np.asarray(handle["observations"][row], dtype=np.float64),
                    "next_observation": np.asarray(
                        handle["next_observations"][row], dtype=np.float64
                    ),
                    "action": np.asarray(handle["actions"][row], dtype=np.float32),
                    "reward": float(handle["rewards"][row]),
                    "terminal": bool(terminals[row]),
                    "timeout": bool(timeouts[row]),
                    "qpos": np.asarray(handle["infos/qpos"][row], dtype=np.float64),
                    "qvel": np.asarray(handle["infos/qvel"][row], dtype=np.float64),
                }
            )

    policy = D4RLTanhGaussianPolicy.from_hdf5(policy_path, device=args.device)
    import gymnasium as gym

    env = gym.make(args.env)
    validation_started = time.perf_counter()
    validations = [
        _validate_logged_step(
            env,
            anchor=anchor,
            reset_seed=args.reset_seed + int(anchor["episode_id"]),
        )
        for anchor in anchors
    ]
    validation_seconds = time.perf_counter() - validation_started
    failures = [
        item
        for item in validations
        if item["restored_observation_max_abs_error"] > args.observation_atol
        or item["logged_next_observation_max_abs_error"] > args.observation_atol
        or item["logged_reward_abs_error"] > args.reward_atol
        or item["actual_terminated"] != item["expected_terminated"]
        or item["actual_truncated"] != item["expected_truncated"]
    ]
    if failures:
        env.close()
        raise RuntimeError(
            f"logged one-step reproduction failed for {len(failures)}/{len(validations)} anchors; "
            f"first failure: {failures[0]}"
        )

    rollout_started = time.perf_counter()
    betas = tuple(float(value) for value in args.betas)
    rewards_by_beta: Dict[float, List[np.ndarray]] = {beta: [] for beta in betas}
    anchor_outputs = []
    total_environment_steps = 0
    for anchor in anchors:
        per_beta_returns: Dict[float, List[np.ndarray]] = {beta: [] for beta in betas}
        per_beta_lengths: Dict[float, List[int]] = {beta: [] for beta in betas}
        per_beta_done: Dict[float, List[Dict[str, bool]]] = {beta: [] for beta in betas}
        per_beta_rewards: Dict[float, List[np.ndarray]] = {beta: [] for beta in betas}
        for replicate in range(args.replicates):
            policy_seed = _paired_seed(args.rollout_seed, anchor["ordinal"], replicate, 0)
            noise_seed = _paired_seed(args.rollout_seed, anchor["ordinal"], replicate, 1)
            reset_seed = _paired_seed(args.rollout_seed, anchor["ordinal"], replicate, 2)
            for beta in betas:
                returns, rewards, length, terminated, truncated = _rollout(
                    env,
                    policy,
                    anchor=anchor,
                    beta=beta,
                    horizons=horizons,
                    discount=args.discount,
                    policy_seed=policy_seed,
                    noise_seed=noise_seed,
                    reset_seed=reset_seed,
                    device=args.device,
                )
                per_beta_returns[beta].append(returns)
                per_beta_rewards[beta].append(rewards)
                per_beta_lengths[beta].append(length)
                per_beta_done[beta].append(
                    {"terminated": terminated, "truncated": truncated}
                )
                total_environment_steps += length
        output: Dict[str, Any] = {
            key: anchor[key]
            for key in (
                "ordinal",
                "row",
                "suffix_bin",
                "episode_id",
                "episode_step",
                "logged_suffix_length",
            )
        }
        output["conditions"] = {}
        output["paired_replicate_seeds"] = [
            {
                "policy": _paired_seed(
                    args.rollout_seed, anchor["ordinal"], replicate, 0
                ),
                "action_noise": _paired_seed(
                    args.rollout_seed, anchor["ordinal"], replicate, 1
                ),
                "reset": _paired_seed(
                    args.rollout_seed, anchor["ordinal"], replicate, 2
                ),
            }
            for replicate in range(args.replicates)
        ]
        for beta in betas:
            return_matrix = np.stack(per_beta_returns[beta])
            reward_matrix = np.stack(per_beta_rewards[beta])
            rewards_by_beta[beta].append(reward_matrix)
            output["conditions"][str(beta)] = {
                "returns": return_matrix.tolist(),
                "return_means": return_matrix.mean(axis=0).tolist(),
                "return_variances_ddof1": return_matrix.var(axis=0, ddof=1).tolist(),
                "rollout_lengths": per_beta_lengths[beta],
                "done": per_beta_done[beta],
            }
        beta_zero = np.stack(per_beta_returns[0.0])
        beta_one = np.stack(per_beta_returns[1.0])
        paired_difference = beta_one - beta_zero
        output["paired_beta1_minus_beta0"] = {
            "return_differences": paired_difference.tolist(),
            "mean": paired_difference.mean(axis=0).tolist(),
            "variance_ddof1": paired_difference.var(axis=0, ddof=1).tolist(),
        }
        anchor_outputs.append(output)
    rollout_seconds = time.perf_counter() - rollout_started
    env.close()

    analyses: Dict[str, Any] = {}
    for beta in betas:
        variances = np.asarray(
            [
                anchor["conditions"][str(beta)]["return_variances_ddof1"]
                for anchor in anchor_outputs
            ],
            dtype=np.float64,
        )
        analyses[str(beta)] = {
            "anchor_variances": variances.tolist(),
            "geometric_fit": fit_geometric_curve(variances, horizons, args.discount),
            "anchor_bootstrap": bootstrap_geometric_fit(
                variances,
                horizons,
                args.discount,
                resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed + int(beta),
            ),
            "covariance_diagnostics": covariance_diagnostics(
                rewards_by_beta[beta], horizons, args.discount
            ),
        }
    paired_variances = np.asarray(
        [
            anchor["paired_beta1_minus_beta0"]["variance_ddof1"]
            for anchor in anchor_outputs
        ],
        dtype=np.float64,
    )
    paired_reward_differences = [
        beta_one - beta_zero
        for beta_one, beta_zero in zip(rewards_by_beta[1.0], rewards_by_beta[0.0])
    ]
    analyses["paired_beta1_minus_beta0"] = {
        "anchor_variances": paired_variances.tolist(),
        "interpretation": (
            "variance of common-random-number return differences; this captures "
            "the actuator perturbation's pathwise effect, not Var(beta1)-Var(beta0)"
        ),
        "geometric_fit": fit_geometric_curve(
            paired_variances, horizons, args.discount
        ),
        "anchor_bootstrap": bootstrap_geometric_fit(
            paired_variances,
            horizons,
            args.discount,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed + 2,
        ),
        "covariance_diagnostics": covariance_diagnostics(
            paired_reward_differences, horizons, args.discount
        ),
    }

    finished = time.perf_counter()
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "scientific_unit": "restored anchor state",
        "replicate_role": "conditional Monte Carlo samples within an anchor",
        "config": {
            "dataset": str(dataset_path),
            "policy_source": str(policy_path),
            "environment": args.env,
            "horizons": list(horizons),
            "discount": args.discount,
            "betas": list(betas),
            "replicates_per_anchor": args.replicates,
            "anchors_per_suffix_bin": args.anchors_per_bin,
            "suffix_bin_starts": list(args.suffix_bin_starts),
            "selection_seed": args.selection_seed,
            "rollout_seed": args.rollout_seed,
            "reset_seed": args.reset_seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "device": args.device,
            "observation_atol": args.observation_atol,
            "reward_atol": args.reward_atol,
            "maximum_episode_steps": maximum_episode_steps,
        },
        "hashes": {
            "dataset_sha256": sha256_file(dataset_path),
            "policy_source_sha256": sha256_file(policy_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "behavior_policy_sha256": sha256_file(
                Path(__file__).resolve().with_name("behavior_policy.py")
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "validation": {
            "passed_all": True,
            "anchor_count": len(validations),
            "records": validations,
            "maximum_restored_observation_abs_error": max(
                item["restored_observation_max_abs_error"] for item in validations
            ),
            "maximum_logged_next_observation_abs_error": max(
                item["logged_next_observation_max_abs_error"] for item in validations
            ),
            "maximum_logged_reward_abs_error": max(
                item["logged_reward_abs_error"] for item in validations
            ),
        },
        "anchors": anchor_outputs,
        "analysis": analyses,
        "timing": {
            "validation_seconds": validation_seconds,
            "rollout_seconds": rollout_seconds,
            "wall_seconds": finished - started,
            "environment_steps": total_environment_steps,
            "environment_steps_per_rollout_second": total_environment_steps
            / max(rollout_seconds, 1e-12),
        },
    }
    atomic_write_json(args.output.resolve(), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", default="Walker2d-v4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--betas", nargs="+", type=float, default=(0.0, 1.0))
    parser.add_argument("--suffix-bin-starts", nargs="+", type=int, default=DEFAULT_SUFFIX_BIN_STARTS)
    parser.add_argument("--anchors-per-bin", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=20260914)
    parser.add_argument("--rollout-seed", type=int, default=20260915)
    parser.add_argument("--reset-seed", type=int, default=20260916)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--observation-atol", type=float, default=1e-5)
    parser.add_argument("--reward-atol", type=float, default=1e-5)
    return parser


def main() -> int:
    payload = run(build_parser().parse_args())
    concise = {
        "status": payload["status"],
        "anchors": payload["validation"]["anchor_count"],
        "timing": payload["timing"],
        "geometric_fit": {
            beta: result["geometric_fit"] for beta, result in payload["analysis"].items()
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
