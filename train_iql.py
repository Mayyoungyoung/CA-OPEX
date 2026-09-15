"""Train equal-budget IQL/HUBL variants on a flat or episodic HDF5 dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from iql_core import (
    IQLAgent,
    IQLConfig,
    expand_episode_values,
    normalized_rank,
    posterior_expected_rank,
)


@dataclass
class PreparedDataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    clean_rewards: np.ndarray
    noisy_rewards: np.ndarray
    terminals: np.ndarray
    episode_ids: np.ndarray
    remaining_steps: np.ndarray
    raw_mc_next: np.ndarray
    clean_mc_next: np.ndarray
    action_residual_next_suffix: Optional[np.ndarray]
    action_residual_metadata: Dict[str, object]
    noisy_episode_scores: np.ndarray
    clean_episode_scores: np.ndarray
    episode_lengths: np.ndarray
    reward_scale: float
    noise_metadata: Dict[str, float]
    source_schema: str


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def discounted_cumsum(rewards: np.ndarray, discount: float) -> np.ndarray:
    output = np.empty_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        output[index] = running
    return output.astype(np.float32)


def _sorted_episode_keys(handle: h5py.File) -> List[str]:
    keys = [key for key in handle.keys() if key.startswith("episode_")]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def _read_episode_arrays(path: Path, max_episodes: Optional[int]) -> Tuple[List[Dict[str, np.ndarray]], str]:
    episodes: List[Dict[str, np.ndarray]] = []
    with h5py.File(path, "r") as handle:
        episode_keys = _sorted_episode_keys(handle)
        if episode_keys:
            selected = episode_keys[:max_episodes] if max_episodes else episode_keys
            for key in selected:
                group = handle[key]
                actions = np.asarray(group["actions"], dtype=np.float32)
                observations = np.asarray(group["observations"], dtype=np.float32)
                rewards = np.asarray(group["rewards"], dtype=np.float32).reshape(-1)
                terminals = np.asarray(
                    group["terminations"] if "terminations" in group else group["terminals"],
                    dtype=np.bool_,
                ).reshape(-1)
                timeouts = np.asarray(
                    group["truncations"] if "truncations" in group else group["timeouts"],
                    dtype=np.bool_,
                ).reshape(-1)
                if observations.shape[0] != actions.shape[0] + 1:
                    raise ValueError(f"{key}: episodic observations must have T+1 rows")
                clean_policy_actions = (
                    np.asarray(group["clean_policy_actions"], dtype=np.float32)
                    if "clean_policy_actions" in group
                    else None
                )
                if (
                    clean_policy_actions is not None
                    and clean_policy_actions.shape != actions.shape
                ):
                    raise ValueError(
                        f"{key}: clean_policy_actions and actions must have identical shape"
                    )
                episodes.append(
                    dict(
                        observations=observations,
                        actions=actions,
                        clean_policy_actions=clean_policy_actions,
                        rewards=rewards,
                        terminals=terminals,
                        timeouts=timeouts,
                    )
                )
            return episodes, "minari_episodic"

        required = ("observations", "actions", "rewards", "terminals")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"missing HDF5 arrays: {missing}")
        observations = np.asarray(handle["observations"], dtype=np.float32)
        actions = np.asarray(handle["actions"], dtype=np.float32)
        clean_policy_actions = (
            np.asarray(handle["clean_policy_actions"], dtype=np.float32)
            if "clean_policy_actions" in handle
            else None
        )
        if clean_policy_actions is not None and clean_policy_actions.shape != actions.shape:
            raise ValueError(
                "clean_policy_actions and actions must have identical shape"
            )
        rewards = np.asarray(handle["rewards"], dtype=np.float32).reshape(-1)
        terminals = np.asarray(handle["terminals"], dtype=np.bool_).reshape(-1)
        timeouts = (
            np.asarray(handle["timeouts"], dtype=np.bool_).reshape(-1)
            if "timeouts" in handle
            else np.zeros_like(terminals)
        )
    n = rewards.size
    if not all(array.shape[0] == n for array in (observations, actions, terminals, timeouts)):
        raise ValueError("flat HDF5 arrays have inconsistent first dimensions")
    boundaries = np.flatnonzero(terminals | timeouts)
    stops = (boundaries + 1).tolist()
    if not stops or stops[-1] != n:
        stops.append(n)
    start = 0
    for stop in stops:
        if stop <= start:
            continue
        # D4RL stores current observations only.  Match qlearning_dataset/HUBL:
        # duplicate the final state for true termination, and later discard the
        # final transition for timeout/incomplete episodes.
        episode_observations = observations[start:stop]
        if terminals[stop - 1]:
            episode_observations = np.concatenate(
                (episode_observations, episode_observations[-1:]), axis=0
            )
        else:
            next_tail = observations[stop : stop + 1]
            if next_tail.size == 0:
                next_tail = episode_observations[-1:]
            episode_observations = np.concatenate((episode_observations, next_tail), axis=0)
        episodes.append(
            dict(
                observations=episode_observations,
                actions=actions[start:stop],
                clean_policy_actions=(
                    clean_policy_actions[start:stop]
                    if clean_policy_actions is not None
                    else None
                ),
                rewards=rewards[start:stop],
                terminals=terminals[start:stop],
                timeouts=timeouts[start:stop],
            )
        )
        start = stop
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes, "d4rl_flat"


def prepare_dataset(
    path: Path,
    *,
    discount: float,
    noise_seed: int,
    iid_noise_scale: float,
    episode_noise_scale: float,
    max_episodes: Optional[int] = None,
) -> PreparedDataset:
    episodes, source_schema = _read_episode_arrays(path, max_episodes)
    if not episodes:
        raise ValueError("dataset contains no episodes")
    clean_all = np.concatenate([episode["rewards"] for episode in episodes])
    base_std = float(clean_all.std())
    # Keep transition noise identical to mechanism_audit.py for a given seed.
    # Episode-correlated noise uses a disjoint stream so enabling it cannot
    # silently change the IID realization.
    iid_rng = np.random.default_rng(noise_seed)
    episode_rng = np.random.default_rng(noise_seed + 1_000_003)
    iid_std = iid_noise_scale * base_std
    episode_std = episode_noise_scale * base_std
    episode_biases = episode_rng.normal(0.0, episode_std, size=len(episodes)).astype(np.float32)

    clean_undiscounted = np.asarray([episode["rewards"].sum() for episode in episodes])
    return_range = float(clean_undiscounted.max() - clean_undiscounted.min())
    if return_range <= 0:
        raise ValueError("clean trajectory return range must be positive")
    reward_scale = 1000.0 / return_range

    # ERR-HUBL uses paired commanded/executed actions to estimate how much
    # actuator perturbation is present in the *next* Monte-Carlo suffix.  The
    # raw residual sequence is deliberately constructed before the learner's
    # timeout/incomplete-episode keep mask.  Consequently, the dropped final
    # raw step still contributes to D_{t+1}, exactly as its reward contributes
    # to raw_mc_next for the last retained transition.
    command_availability = [
        episode.get("clean_policy_actions") is not None for episode in episodes
    ]
    action_residual_suffixes: Optional[List[np.ndarray]] = None
    action_residual_metadata: Dict[str, object]
    if all(command_availability):
        residual_energy_episodes: List[np.ndarray] = []
        for episode_id, episode in enumerate(episodes):
            executed = np.asarray(episode["actions"], dtype=np.float64)
            commanded = np.asarray(
                episode["clean_policy_actions"], dtype=np.float64
            )
            if executed.shape != commanded.shape:
                raise ValueError(
                    f"episode {episode_id}: paired action arrays are misaligned"
                )
            if not (np.all(np.isfinite(executed)) and np.all(np.isfinite(commanded))):
                raise ValueError(
                    f"episode {episode_id}: paired action arrays contain non-finite values"
                )
            squared_residual = np.square(executed - commanded).reshape(
                executed.shape[0], -1
            )
            residual_energy_episodes.append(
                squared_residual.mean(axis=1).astype(np.float32)
            )
        raw_residual_energy = np.concatenate(residual_energy_episodes)
        residual_mean = float(raw_residual_energy.mean(dtype=np.float64))
        residual_std = float(raw_residual_energy.std(dtype=np.float64))
        action_residual_metadata = {
            "available": True,
            "raw_transition_count": int(raw_residual_energy.size),
            "residual_energy_definition": (
                "mean_action_dim((executed_action-clean_policy_action)^2)"
            ),
            "residual_energy_raw_mean": residual_mean,
            "residual_energy_raw_std": residual_std,
            "residual_energy_raw_sequence_sha256": _float32_array_sha256(
                raw_residual_energy, sort_values=False
            ),
            "alignment_semantics": (
                "D_next[t] sums normalized paired-action residual energy over raw "
                "steps k=t+1..T-1 with gamma^(2*(k-t-1)) before the learner "
                "keep mask; a dropped timeout/incomplete final raw step therefore "
                "contributes to the final retained transition, matching raw_mc_next"
            ),
        }
        if residual_mean > 0.0:
            action_residual_suffixes = []
            kept_suffixes: List[np.ndarray] = []
            squared_discount = discount * discount
            for episode, residual_energy in zip(episodes, residual_energy_episodes):
                normalized_energy = residual_energy / residual_mean
                inclusive_suffix = discounted_cumsum(
                    normalized_energy, squared_discount
                )
                next_suffix = np.concatenate(
                    (inclusive_suffix[1:], np.zeros(1, dtype=np.float32))
                )
                length = residual_energy.size
                keep = np.ones(length, dtype=np.bool_)
                if bool(episode["timeouts"][-1]) or not bool(
                    episode["terminals"][-1] | episode["timeouts"][-1]
                ):
                    keep[-1] = False
                kept = next_suffix[keep].astype(np.float32, copy=False)
                action_residual_suffixes.append(kept)
                kept_suffixes.append(kept)
            aligned_suffix = np.concatenate(kept_suffixes)
            action_residual_metadata.update(
                {
                    "normalization_valid": True,
                    "next_suffix_D_mean": float(aligned_suffix.mean()),
                    "next_suffix_D_min": float(aligned_suffix.min()),
                    "next_suffix_D_max": float(aligned_suffix.max()),
                    "next_suffix_D_sequence_sha256": _float32_array_sha256(
                        aligned_suffix, sort_values=False
                    ),
                    # The sequence order is exactly the raw_mc_next/learner
                    # transition order, so this digest also audits alignment.
                    "alignment_sequence_sha256": _float32_array_sha256(
                        aligned_suffix, sort_values=False
                    ),
                }
            )
        else:
            action_residual_metadata.update(
                {
                    "normalization_valid": False,
                    "unavailable_reason": (
                        "paired actions have zero global residual energy; e_bar is zero"
                    ),
                }
            )
    else:
        missing_count = int(len(episodes) - sum(command_availability))
        action_residual_metadata = {
            "available": False,
            "normalization_valid": False,
            "episodes_missing_clean_policy_actions": missing_count,
            "unavailable_reason": (
                "HDF5 field clean_policy_actions is absent from one or more selected episodes"
            ),
            "alignment_semantics": (
                "unavailable because paired commanded/executed actions are required"
            ),
        }

    fields: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "observations",
            "actions",
            "next_observations",
            "clean_rewards",
            "noisy_rewards",
            "terminals",
            "episode_ids",
            "remaining_steps",
            "raw_mc_next",
            "clean_mc_next",
        )
    }
    clean_scores, noisy_scores, episode_lengths = [], [], []
    realized_iid = []
    for episode_id, episode in enumerate(episodes):
        clean = episode["rewards"].astype(np.float32)
        iid = iid_rng.normal(0.0, iid_std, size=clean.size).astype(np.float32)
        noisy = clean + iid + episode_biases[episode_id]
        realized_iid.append(iid)
        clean_return = discounted_cumsum(clean, discount)
        noisy_return = discounted_cumsum(noisy, discount)
        clean_next = np.concatenate((clean_return[1:], np.zeros(1, dtype=np.float32)))
        noisy_next = np.concatenate((noisy_return[1:], np.zeros(1, dtype=np.float32)))
        length = clean.size
        keep = np.ones(length, dtype=np.bool_)
        if bool(episode["timeouts"][-1]) or not bool(
            episode["terminals"][-1] | episode["timeouts"][-1]
        ):
            keep[-1] = False
        fields["observations"].append(episode["observations"][:-1][keep])
        fields["actions"].append(episode["actions"][keep])
        fields["next_observations"].append(episode["observations"][1:][keep])
        fields["clean_rewards"].append(clean[keep] * reward_scale)
        fields["noisy_rewards"].append(noisy[keep] * reward_scale)
        fields["terminals"].append(episode["terminals"].astype(np.float32)[keep])
        fields["episode_ids"].append(np.full(int(keep.sum()), episode_id, dtype=np.int64))
        # Number of rewards contained in the next-state Monte-Carlo suffix.
        # This is zero at a true terminal and remains one at the final retained
        # transition of a timeout/incomplete trajectory because that final raw
        # reward is retained in the truncated return but its transition is not
        # exposed to the learner.
        fields["remaining_steps"].append(
            (length - np.arange(length, dtype=np.int32) - 1)[keep]
        )
        fields["raw_mc_next"].append(noisy_next[keep] * reward_scale)
        fields["clean_mc_next"].append(clean_next[keep] * reward_scale)
        clean_scores.append(float(clean_next[keep].mean()) * reward_scale)
        noisy_scores.append(float(noisy_next[keep].mean()) * reward_scale)
        episode_lengths.append(length)
    concatenated = {key: np.concatenate(value, axis=0) for key, value in fields.items()}
    action_residual_next_suffix = (
        np.concatenate(action_residual_suffixes).astype(np.float32, copy=False)
        if action_residual_suffixes is not None
        else None
    )
    if (
        action_residual_next_suffix is not None
        and action_residual_next_suffix.shape != concatenated["raw_mc_next"].shape
    ):
        raise RuntimeError(
            "action residual next-suffix values do not align with raw_mc_next"
        )
    iid_values = np.concatenate(realized_iid)
    return PreparedDataset(
        observations=concatenated["observations"],
        actions=concatenated["actions"],
        next_observations=concatenated["next_observations"],
        clean_rewards=concatenated["clean_rewards"],
        noisy_rewards=concatenated["noisy_rewards"],
        terminals=concatenated["terminals"],
        episode_ids=concatenated["episode_ids"],
        remaining_steps=concatenated["remaining_steps"],
        raw_mc_next=concatenated["raw_mc_next"],
        clean_mc_next=concatenated["clean_mc_next"],
        action_residual_next_suffix=action_residual_next_suffix,
        action_residual_metadata=action_residual_metadata,
        noisy_episode_scores=np.asarray(noisy_scores, dtype=np.float32),
        clean_episode_scores=np.asarray(clean_scores, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        reward_scale=reward_scale,
        noise_metadata={
            "discount": float(discount),
            "clean_reward_std_raw": base_std,
            "iid_noise_scale": iid_noise_scale,
            "iid_noise_std_raw": iid_std,
            "iid_noise_realized_std_raw": float(iid_values.std()),
            "episode_noise_scale": episode_noise_scale,
            "episode_bias_std_raw": episode_std,
            "episode_bias_realized_std_raw": float(episode_biases.std()),
            "noise_seed": int(noise_seed),
        },
        source_schema=source_schema,
    )


def _load_predictions(path: Optional[Path], episode_count: int, transition_count: int) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path) as archive:
        predictions = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("crossfit_mean", "crossfit_std"):
        if key in predictions and predictions[key].reshape(-1).size != episode_count:
            raise ValueError(f"{key} must contain one value per episode")
    for key in ("crossfit_next_mean", "crossfit_next_std"):
        if key in predictions and predictions[key].reshape(-1).size != transition_count:
            raise ValueError(f"{key} must contain one value per kept transition")
    return predictions


HORIZON_VARIANTS = (
    "hubl_horizon",
    "hubl_horizon_shuffled",
    "hubl_horizon_reverse",
    "hubl_rank_horizon",
)

HORIZON_MULTISET_CONTROL_VARIANTS = (
    "hubl_horizon",
    "hubl_horizon_shuffled",
    "hubl_horizon_reverse",
)

ACTION_RESIDUAL_VARIANTS = (
    "hubl_action_residual",
    "hubl_action_residual_shuffled",
)

TRANSITION_LAMBDA_VARIANTS = HORIZON_VARIANTS + ACTION_RESIDUAL_VARIANTS


def _float32_array_sha256(values: np.ndarray, *, sort_values: bool) -> str:
    """Hash float values with an explicit, platform-independent byte layout."""

    canonical = np.asarray(values, dtype="<f4").reshape(-1)
    if sort_values:
        canonical = np.sort(canonical)
    return hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()


def _contiguous_episode_slices(
    episode_ids: np.ndarray,
) -> List[Tuple[int, slice]]:
    """Return O(N)-constructed slices after validating episode contiguity."""

    ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return []
    if np.any(ids[1:] < ids[:-1]):
        raise ValueError(
            "episode_ids must be nondecreasing so each episode is contiguous"
        )
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.flatnonzero(ids[1:] != ids[:-1]) + 1)
    )
    stops = np.concatenate((starts[1:], np.asarray([ids.size], dtype=np.int64)))
    return [
        (int(ids[start]), slice(int(start), int(stop)))
        for start, stop in zip(starts, stops)
    ]


def build_hubl_fields(
    dataset: PreparedDataset,
    variant: str,
    heuristic_discount: float,
    lcb_kappa: float,
    predictions: Dict[str, np.ndarray],
    horizon_noise_scale: float = 0.0,
    horizon_control_seed: int = 104729,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    episode_count = dataset.noisy_episode_scores.size
    if variant == "iql":
        scores = np.zeros(episode_count, dtype=np.float32)
        episode_lambdas = np.zeros(episode_count, dtype=np.float32)
        heuristic = np.zeros_like(dataset.raw_mc_next)
    elif variant == "hubl_constant":
        scores = dataset.noisy_episode_scores
        episode_lambdas = np.full(episode_count, heuristic_discount, dtype=np.float32)
        heuristic = dataset.raw_mc_next
    elif variant == "hubl_rank":
        scores = dataset.noisy_episode_scores
        episode_lambdas = heuristic_discount * normalized_rank(scores)
        heuristic = dataset.raw_mc_next
    elif variant in HORIZON_VARIANTS:
        if horizon_noise_scale < 0.0:
            raise ValueError("horizon_noise_scale must be non-negative")
        scores = dataset.noisy_episode_scores
        episode_lambdas = np.zeros(episode_count, dtype=np.float32)
        remaining = dataset.remaining_steps.astype(np.float64)
        if dataset.noise_metadata.get("discount", None) is not None:
            discount = float(dataset.noise_metadata["discount"])
        else:
            # Existing result archives predate the explicit field.  The
            # training entry points always use 0.99 unless overridden and now
            # record it below; this fallback keeps unit construction simple.
            discount = 0.99
        if discount == 1.0:
            effective_horizon = remaining
        else:
            effective_horizon = (1.0 - np.power(discount * discount, remaining)) / (
                1.0 - discount * discount
            )
        base_lambdas = (
            heuristic_discount
            / (1.0 + horizon_noise_scale * effective_horizon)
        ).astype(np.float32)
        if variant == "hubl_rank_horizon":
            quality_gate = expand_episode_values(
                normalized_rank(scores), dataset.episode_ids
            ).astype(np.float32)
            lambdas = quality_gate * base_lambdas
            horizon_control = (
                "episode_quality_rank_times_transition_horizon_reliability"
            )
            horizon_control_uses_seed = False
        elif variant == "hubl_horizon":
            lambdas = base_lambdas
            horizon_control = "none"
            horizon_control_uses_seed = False
        elif variant == "hubl_horizon_shuffled":
            if horizon_control_seed < 0:
                raise ValueError("horizon_control_seed must be non-negative")
            permutation = np.random.default_rng(horizon_control_seed).permutation(
                base_lambdas.size
            )
            lambdas = base_lambdas[permutation]
            horizon_control = (
                "global_fixed_seed_permutation_of_transition_lambdas"
            )
            horizon_control_uses_seed = True
        else:
            lambdas = base_lambdas.copy()
            episode_slices = _contiguous_episode_slices(dataset.episode_ids)
            for _, episode_slice in episode_slices:
                lambdas[episode_slice] = base_lambdas[episode_slice][::-1]
            horizon_control = "reverse_lambda_sequence_within_each_episode"
            horizon_control_uses_seed = False
        heuristic = dataset.raw_mc_next
    elif variant in ACTION_RESIDUAL_VARIANTS:
        if horizon_noise_scale < 0.0:
            raise ValueError("horizon_noise_scale must be non-negative")
        if dataset.action_residual_next_suffix is None:
            reason = str(
                dataset.action_residual_metadata.get(
                    "unavailable_reason", "paired action residuals are unavailable"
                )
            )
            if not bool(dataset.action_residual_metadata.get("available", False)):
                raise KeyError(
                    f"{variant} requires HDF5 field clean_policy_actions: {reason}"
                )
            raise ValueError(f"{variant} cannot normalize action residuals: {reason}")
        scores = dataset.noisy_episode_scores
        episode_lambdas = np.zeros(episode_count, dtype=np.float32)
        residual_suffix = np.asarray(
            dataset.action_residual_next_suffix, dtype=np.float32
        )
        if residual_suffix.shape != dataset.raw_mc_next.shape:
            raise ValueError(
                "action residual next-suffix values must align one-to-one with raw_mc_next"
            )
        base_lambdas = (
            heuristic_discount
            / (1.0 + horizon_noise_scale * residual_suffix.astype(np.float64))
        ).astype(np.float32)
        if variant == "hubl_action_residual":
            lambdas = base_lambdas
            action_residual_control = "none"
            action_residual_control_uses_seed = False
        else:
            if horizon_control_seed < 0:
                raise ValueError("horizon_control_seed must be non-negative")
            permutation = np.random.default_rng(horizon_control_seed).permutation(
                base_lambdas.size
            )
            lambdas = base_lambdas[permutation]
            action_residual_control = (
                "global_fixed_seed_permutation_of_transition_lambdas"
            )
            action_residual_control_uses_seed = True
        heuristic = dataset.raw_mc_next
    else:
        required = ("crossfit_mean", "crossfit_std")
        if any(key not in predictions for key in required):
            raise ValueError(f"{variant} requires prediction keys {required}")
        means = predictions["crossfit_mean"].reshape(-1).astype(np.float32)
        stds = np.maximum(predictions["crossfit_std"].reshape(-1), 1e-6).astype(np.float32)
        if variant in ("cf_mean_rank", "cf_mean_h"):
            scores = means
            episode_lambdas = heuristic_discount * normalized_rank(means)
        elif variant in ("cf_lcb_rank", "cf_lcb_h"):
            scores = means - lcb_kappa * stds
            episode_lambdas = heuristic_discount * normalized_rank(scores)
        elif variant in ("cf_prob_rank", "cf_prob_h"):
            scores = means
            episode_lambdas = heuristic_discount * posterior_expected_rank(means, stds)
        else:
            raise ValueError(f"unsupported variant: {variant}")
        if variant.endswith("_h"):
            if "crossfit_next_mean" not in predictions:
                raise ValueError(f"{variant} requires crossfit_next_mean")
            heuristic = predictions["crossfit_next_mean"].reshape(-1).astype(np.float32)
            if predictions.get("prediction_units", np.asarray("raw")).item() == "raw":
                heuristic = heuristic * dataset.reward_scale
        else:
            heuristic = dataset.raw_mc_next
    if variant not in TRANSITION_LAMBDA_VARIANTS:
        lambdas = expand_episode_values(episode_lambdas, dataset.episode_ids).astype(np.float32)
    else:
        lambdas = lambdas.astype(np.float32)
    score_spearman = _spearman(scores, dataset.clean_episode_scores)
    metadata = {
        "episode_score_spearman_vs_clean": (
            score_spearman if np.isfinite(score_spearman) else None
        ),
        "lambda_mean": float(episode_lambdas.mean()),
        "lambda_min": float(episode_lambdas.min()),
        "lambda_max": float(episode_lambdas.max()),
        "heuristic_mse_vs_clean": float(np.mean(np.square(heuristic - dataset.clean_mc_next))),
    }
    if variant in HORIZON_VARIANTS:
        if variant in HORIZON_MULTISET_CONTROL_VARIANTS:
            sorted_base_lambdas = np.sort(base_lambdas)
            sorted_controlled_lambdas = np.sort(lambdas)
            multiset_preserved = bool(
                np.array_equal(sorted_base_lambdas, sorted_controlled_lambdas)
            )
            if variant == "hubl_horizon":
                # Identity mapping; no episode scan is needed.
                per_episode_multisets_preserved = True
            elif variant == "hubl_horizon_shuffled":
                # This control preserves only the global multiset by design.
                per_episode_multisets_preserved = None
            else:
                per_episode_multisets_preserved = bool(
                    all(
                        np.array_equal(
                            np.sort(base_lambdas[episode_slice]),
                            np.sort(lambdas[episode_slice]),
                        )
                        for _, episode_slice in episode_slices
                    )
                )
            if not multiset_preserved:
                raise RuntimeError("horizon control changed the lambda multiset")
            if (
                variant == "hubl_horizon_reverse"
                and not per_episode_multisets_preserved
            ):
                raise RuntimeError(
                    "episode reversal changed an episode lambda multiset"
                )
        else:
            multiset_preserved = None
            per_episode_multisets_preserved = None
        metadata.update(
            {
                # Report the pre-control reductions so equal-multiset controls
                # have byte-identical summary statistics as the main variant.
                "lambda_mean": float(
                    lambdas.mean()
                    if variant == "hubl_rank_horizon"
                    else base_lambdas.mean()
                ),
                "lambda_min": float(
                    lambdas.min()
                    if variant == "hubl_rank_horizon"
                    else base_lambdas.min()
                ),
                "lambda_max": float(
                    lambdas.max()
                    if variant == "hubl_rank_horizon"
                    else base_lambdas.max()
                ),
                "horizon_noise_scale": float(horizon_noise_scale),
                "remaining_steps_mean": float(dataset.remaining_steps.mean()),
                "remaining_steps_max": int(dataset.remaining_steps.max()),
                "effective_horizon_mean": float(effective_horizon.mean()),
                "effective_horizon_max": float(effective_horizon.max()),
                "horizon_control": horizon_control,
                "horizon_control_seed": int(horizon_control_seed),
                "horizon_control_seed_used": horizon_control_uses_seed,
            }
        )
        if variant in HORIZON_MULTISET_CONTROL_VARIANTS:
            metadata.update(
                {
                    "lambda_sequence_sha256_before_control": _float32_array_sha256(
                        base_lambdas, sort_values=False
                    ),
                    "lambda_sequence_sha256_after_control": _float32_array_sha256(
                        lambdas, sort_values=False
                    ),
                    "lambda_multiset_sha256_before_control": _float32_array_sha256(
                        sorted_base_lambdas, sort_values=False
                    ),
                    "lambda_multiset_sha256_after_control": _float32_array_sha256(
                        sorted_controlled_lambdas, sort_values=False
                    ),
                    "lambda_multiset_exactly_preserved": multiset_preserved,
                    "per_episode_lambda_multisets_exactly_preserved": (
                        per_episode_multisets_preserved
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "quality_gate_mean": float(quality_gate.mean()),
                    "quality_gate_min": float(quality_gate.min()),
                    "quality_gate_max": float(quality_gate.max()),
                    "reliability_gate_mean": float(base_lambdas.mean()),
                    "reliability_gate_min": float(base_lambdas.min()),
                    "reliability_gate_max": float(base_lambdas.max()),
                    "product_lambda_mean": float(lambdas.mean()),
                    "product_lambda_min": float(lambdas.min()),
                    "product_lambda_max": float(lambdas.max()),
                    "quality_gate_sequence_sha256": _float32_array_sha256(
                        quality_gate, sort_values=False
                    ),
                    "reliability_gate_sequence_sha256": _float32_array_sha256(
                        base_lambdas, sort_values=False
                    ),
                    "product_lambda_sequence_sha256": _float32_array_sha256(
                        lambdas, sort_values=False
                    ),
                }
            )
    elif variant in ACTION_RESIDUAL_VARIANTS:
        sorted_base_lambdas = np.sort(base_lambdas)
        sorted_controlled_lambdas = np.sort(lambdas)
        multiset_preserved = bool(
            np.array_equal(sorted_base_lambdas, sorted_controlled_lambdas)
        )
        if not multiset_preserved:
            raise RuntimeError(
                "action-residual shuffle control changed the lambda multiset"
            )
        metadata.update(dataset.action_residual_metadata)
        metadata.update(
            {
                "method_name": "ERR-HUBL",
                "lambda_mean": float(base_lambdas.mean()),
                "lambda_min": float(base_lambdas.min()),
                "lambda_max": float(base_lambdas.max()),
                "action_residual_scale": float(horizon_noise_scale),
                "action_residual_control": action_residual_control,
                "action_residual_control_seed": int(horizon_control_seed),
                "action_residual_control_seed_used": (
                    action_residual_control_uses_seed
                ),
                "lambda_sequence_sha256_before_control": _float32_array_sha256(
                    base_lambdas, sort_values=False
                ),
                "lambda_sequence_sha256_after_control": _float32_array_sha256(
                    lambdas, sort_values=False
                ),
                "lambda_multiset_sha256_before_control": _float32_array_sha256(
                    sorted_base_lambdas, sort_values=False
                ),
                "lambda_multiset_sha256_after_control": _float32_array_sha256(
                    sorted_controlled_lambdas, sort_values=False
                ),
                "lambda_multiset_exactly_preserved": multiset_preserved,
                "action_residual_formula": (
                    "lambda_t=alpha/(1+c*D_next[t]); "
                    "D_next[t]=sum_{k=t+1}^{T-1} gamma^(2*(k-t-1)) "
                    "e_k/e_bar"
                ),
            }
        )
    return heuristic, lambdas, metadata


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = normalized_rank(first)
    second_rank = normalized_rank(second)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    agent: IQLAgent,
    env_name: str,
    seeds: Sequence[int],
    reference_min: float,
    reference_max: float,
) -> Dict[str, object]:
    import gymnasium as gym

    env = gym.make(env_name)
    returns = []
    lengths = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        total_return = 0.0
        for length in range(1, int(env.spec.max_episode_steps or 1000) + 1):
            action = agent.act(observation, env.action_space.low, env.action_space.high)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        returns.append(total_return)
        lengths.append(length)
    env.close()
    values = np.asarray(returns, dtype=np.float64)
    normalized = 100.0 * (values - reference_min) / (reference_max - reference_min)
    return {
        "episode_seeds": [int(seed) for seed in seeds],
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
    }


def save_json(path: Path, payload: object) -> None:
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, (np.floating, float)):
            scalar = float(value)
            return scalar if np.isfinite(scalar) else None
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.train_seed)
    torch.set_num_threads(args.torch_threads)
    dataset_path = Path(args.dataset).resolve()
    dataset = prepare_dataset(
        dataset_path,
        discount=args.discount,
        noise_seed=args.noise_seed,
        iid_noise_scale=args.iid_noise_scale,
        episode_noise_scale=args.episode_noise_scale,
        max_episodes=args.max_episodes,
    )
    predictions = _load_predictions(
        Path(args.predictions).resolve() if args.predictions else None,
        dataset.noisy_episode_scores.size,
        dataset.observations.shape[0],
    )
    heuristic, lambdas, mechanism_metrics = build_hubl_fields(
        dataset,
        args.variant,
        args.heuristic_discount,
        args.lcb_kappa,
        predictions,
        args.horizon_noise_scale,
        args.horizon_control_seed,
    )
    device = torch.device(args.device)
    tensors = {
        "observations": torch.as_tensor(dataset.observations, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(dataset.actions, dtype=torch.float32, device=device),
        "next_observations": torch.as_tensor(
            dataset.next_observations, dtype=torch.float32, device=device
        ),
        "rewards": torch.as_tensor(dataset.noisy_rewards, dtype=torch.float32, device=device),
        "terminals": torch.as_tensor(dataset.terminals, dtype=torch.float32, device=device),
        "heuristic_next": torch.as_tensor(heuristic, dtype=torch.float32, device=device),
        "lambdas": torch.as_tensor(lambdas, dtype=torch.float32, device=device),
    }
    config = IQLConfig(
        observation_dim=dataset.observations.shape[1],
        action_dim=dataset.actions.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        discount=args.discount,
        expectile=args.expectile,
        advantage_temperature=args.advantage_temperature,
        target_rate=args.target_rate,
    )
    agent = IQLAgent(config, device)
    config_payload = {
        "arguments": vars(args),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "schema": dataset.source_schema,
            "episodes": int(dataset.noisy_episode_scores.size),
            "kept_transitions": int(dataset.observations.shape[0]),
            "reward_scale": dataset.reward_scale,
            "noise": dataset.noise_metadata,
            "action_residual": dataset.action_residual_metadata,
        },
        "iql": asdict(config),
        "mechanism_metrics_before_training": mechanism_metrics,
        "selection_rule": "all requested evaluation episodes; no run exclusion",
    }
    save_json(output_dir / "config.json", config_payload)
    progress_path = output_dir / "progress.jsonl"
    generator = torch.Generator(device=device)
    generator.manual_seed(args.train_seed + 1729)
    n = dataset.observations.shape[0]
    last_metrics: Dict[str, float] = {}
    evaluations: List[Dict[str, object]] = []
    for step in range(1, args.updates + 1):
        indices = torch.randint(0, n, (args.batch_size,), generator=generator, device=device)
        batch = {key: value[indices] for key, value in tensors.items()}
        last_metrics = agent.update(batch)
        if step == 1 or step % args.log_period == 0:
            event = {
                "event": "train",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                **last_metrics,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        if step % args.eval_period == 0 or step == args.updates:
            eval_seeds = [args.eval_seed + index for index in range(args.eval_episodes)]
            metrics = evaluate(
                agent,
                args.env_name,
                eval_seeds,
                args.reference_min_score,
                args.reference_max_score,
            )
            evaluation = {
                "event": "evaluation",
                "step": step,
                "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            }
            evaluations.append(evaluation)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation, sort_keys=True) + "\n")
            torch.save(
                {"step": step, "agent": agent.checkpoint(), "config": config_payload},
                output_dir / "latest.pt",
            )
    summary = {
        "status": "complete",
        "variant": args.variant,
        "train_seed": args.train_seed,
        "updates": args.updates,
        "wall_time_seconds": time.perf_counter() - started,
        "final_train_metrics": last_metrics,
        "evaluations": evaluations,
        "final_evaluation": evaluations[-1],
        "mechanism_metrics_before_training": mechanism_metrics,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "iql", "hubl_constant", "hubl_rank", "hubl_horizon",
            "hubl_horizon_shuffled", "hubl_horizon_reverse", "hubl_rank_horizon",
            "hubl_action_residual", "hubl_action_residual_shuffled",
            "cf_mean_rank", "cf_lcb_rank",
            "cf_prob_rank", "cf_mean_h", "cf_lcb_h", "cf_prob_h",
        ),
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=3.0)
    parser.add_argument("--target-rate", type=float, default=0.005)
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
