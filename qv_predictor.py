"""Episode-wise cross-fitted Q/V return predictor for CA-HUBL.

The module is deliberately self contained.  It reads the flat D4RL HDF5
schema, reconstructs episodes from ``terminal | timeout`` boundaries, applies
the exact transition-retention convention used by :mod:`train_iql`, and fits
return regressors without ever training on a held-out episode.

The deployable target is the noisy discounted Monte-Carlo return.  Clean
returns are retained only for mechanism audits when reward noise is injected
synthetically.  All predictions are saved in raw reward units; ``train_iql``
performs its own reward normalization when consuming ``crossfit_next_mean``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class Episode:
    """One episode reconstructed from a flat D4RL transition table."""

    episode_id: int
    start: int
    stop: int
    observations: np.ndarray  # T + 1 rows, matching train_iql reconstruction
    actions: np.ndarray
    clean_rewards: np.ndarray
    noisy_rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    source_indices: np.ndarray
    keep: np.ndarray
    clean_returns: np.ndarray
    noisy_returns: np.ndarray


@dataclass(frozen=True)
class PreparedFlatDataset:
    """Flat kept-transition view plus episode-level audit information."""

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    clean_rewards: np.ndarray
    noisy_rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    episode_ids: np.ndarray
    source_indices: np.ndarray
    clean_return_current: np.ndarray
    noisy_return_current: np.ndarray
    clean_return_next: np.ndarray
    noisy_return_next: np.ndarray
    iid_noise: np.ndarray
    episodes: Tuple[Episode, ...]
    episode_biases: np.ndarray
    clean_episode_scores: np.ndarray
    noisy_episode_scores: np.ndarray
    episode_lengths: np.ndarray
    clean_reward_std: float
    iid_noise_std: float
    episode_bias_std: float
    dataset_sha256: str
    kept_order_sha256: str


@dataclass(frozen=True)
class PredictorConfig:
    """Training and data settings recorded verbatim in the output archive."""

    discount: float = 0.99
    noise_seed: int = 20260914
    iid_noise_scale: float = 0.0
    episode_noise_scale: float = 0.0
    folds: int = 5
    fold_seed: int = 31
    model_seed: int = 47
    ensemble_size: int = 3
    backend: str = "mlp"
    hidden_dim: int = 128
    depth: int = 2
    train_steps: int = 3000
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ridge_l2: float = 1e-3
    prediction_batch_size: int = 65536
    device: str = "auto"
    max_episodes: Optional[int] = None


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _hash_named_arrays(named_arrays: Sequence[Tuple[str, np.ndarray]]) -> str:
    """Hash names, shapes, dtypes and C-order bytes to make order auditable."""

    digest = hashlib.sha256()
    for name, value in named_arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def discounted_cumsum(rewards: np.ndarray, discount: float) -> np.ndarray:
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    result = np.empty(rewards.size, dtype=np.float64)
    running = 0.0
    for index in range(rewards.size - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result.astype(np.float32)


def _load_flat_arrays(path: Path) -> Dict[str, np.ndarray]:
    required = ("observations", "actions", "rewards", "terminals")
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"missing flat D4RL HDF5 arrays: {missing}")
        # Episodic Minari files need a different ordering contract and are
        # intentionally rejected here rather than silently flattened.
        if any(key.startswith("episode_") for key in handle.keys()):
            raise ValueError("qv_predictor expects the flat D4RL HDF5 schema")
        arrays = {key: np.asarray(handle[key]) for key in required}
        arrays["timeouts"] = (
            np.asarray(handle["timeouts"])
            if "timeouts" in handle
            else np.zeros_like(arrays["terminals"], dtype=np.bool_)
        )
    n = int(np.asarray(arrays["rewards"]).reshape(-1).size)
    if n == 0:
        raise ValueError("dataset contains no transitions")
    for key, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"{key} has {value.shape[0]} rows, expected {n}")
    observations = np.asarray(arrays["observations"], dtype=np.float32)
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must both be rank-2 arrays")
    return {
        "observations": observations,
        "actions": actions,
        "rewards": np.asarray(arrays["rewards"], dtype=np.float32).reshape(-1),
        "terminals": np.asarray(arrays["terminals"], dtype=np.bool_).reshape(-1),
        "timeouts": np.asarray(arrays["timeouts"], dtype=np.bool_).reshape(-1),
    }


def _episode_ranges(
    terminals: np.ndarray,
    timeouts: np.ndarray,
    max_episodes: Optional[int],
) -> List[Tuple[int, int]]:
    stops = (np.flatnonzero(terminals | timeouts) + 1).tolist()
    n = int(terminals.size)
    if not stops or stops[-1] != n:
        stops.append(n)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for stop in stops:
        stop_i = int(stop)
        if stop_i > start:
            ranges.append((start, stop_i))
            start = stop_i
        if max_episodes is not None and len(ranges) >= max_episodes:
            break
    if not ranges:
        raise ValueError("could not reconstruct any episodes")
    return ranges


def prepare_flat_dataset(
    path: Path,
    *,
    discount: float,
    noise_seed: int,
    iid_noise_scale: float,
    episode_noise_scale: float,
    max_episodes: Optional[int] = None,
) -> PreparedFlatDataset:
    """Load, perturb, and flatten episodes in ``train_iql`` kept order.

    Noise scales multiply the standard deviation of clean rewards in the
    selected episodes.  The RNG draw order intentionally matches
    ``train_iql.prepare_dataset``: all episode biases are sampled first, then
    one IID-noise vector per episode in episode order.  The episode-correlated
    and IID components intentionally use disjoint deterministic streams, so
    enabling one component cannot silently change the realization of the
    other.  This is the same contract as :func:`train_iql.prepare_dataset`.
    """

    if iid_noise_scale < 0.0 or episode_noise_scale < 0.0:
        raise ValueError("noise scales must be non-negative")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    path = Path(path)
    arrays = _load_flat_arrays(path)
    ranges = _episode_ranges(arrays["terminals"], arrays["timeouts"], max_episodes)
    clean_all = np.concatenate([arrays["rewards"][start:stop] for start, stop in ranges])
    clean_reward_std = float(clean_all.std())
    iid_noise_std = float(iid_noise_scale * clean_reward_std)
    episode_bias_std = float(episode_noise_scale * clean_reward_std)
    iid_rng = np.random.default_rng(noise_seed)
    episode_rng = np.random.default_rng(noise_seed + 1_000_003)
    episode_biases = episode_rng.normal(
        0.0, episode_bias_std, size=len(ranges)
    ).astype(np.float32)

    episodes: List[Episode] = []
    collected: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "observations",
            "actions",
            "next_observations",
            "clean_rewards",
            "noisy_rewards",
            "terminals",
            "timeouts",
            "episode_ids",
            "source_indices",
            "clean_return_current",
            "noisy_return_current",
            "clean_return_next",
            "noisy_return_next",
            "iid_noise",
        )
    }
    clean_episode_scores: List[float] = []
    noisy_episode_scores: List[float] = []
    episode_lengths: List[int] = []

    for episode_id, (start, stop) in enumerate(ranges):
        clean = arrays["rewards"][start:stop].astype(np.float32, copy=True)
        iid = iid_rng.normal(0.0, iid_noise_std, size=clean.size).astype(np.float32)
        noisy = (clean + iid + episode_biases[episode_id]).astype(np.float32)
        clean_returns = discounted_cumsum(clean, discount)
        noisy_returns = discounted_cumsum(noisy, discount)
        clean_next = np.concatenate((clean_returns[1:], np.zeros(1, dtype=np.float32)))
        noisy_next = np.concatenate((noisy_returns[1:], np.zeros(1, dtype=np.float32)))

        current_observations = arrays["observations"][start:stop]
        # This is intentionally identical to train_iql._read_episode_arrays.
        # A true terminal has no next row in D4RL, so its current state is
        # duplicated.  A timeout/incomplete last transition is dropped below,
        # making the borrowed next row unobservable to the learner.
        if bool(arrays["terminals"][stop - 1]):
            tail = current_observations[-1:]
        else:
            tail = arrays["observations"][stop : stop + 1]
            if tail.size == 0:
                tail = current_observations[-1:]
        episode_observations = np.concatenate((current_observations, tail), axis=0)

        keep = np.ones(clean.size, dtype=np.bool_)
        if bool(arrays["timeouts"][stop - 1]) or not bool(
            arrays["terminals"][stop - 1] | arrays["timeouts"][stop - 1]
        ):
            keep[-1] = False
        if not np.any(keep):
            raise ValueError(f"episode {episode_id} has no kept transition")
        source_indices = np.arange(start, stop, dtype=np.int64)
        episode = Episode(
            episode_id=episode_id,
            start=start,
            stop=stop,
            observations=episode_observations,
            actions=arrays["actions"][start:stop],
            clean_rewards=clean,
            noisy_rewards=noisy,
            terminals=arrays["terminals"][start:stop],
            timeouts=arrays["timeouts"][start:stop],
            source_indices=source_indices,
            keep=keep,
            clean_returns=clean_returns,
            noisy_returns=noisy_returns,
        )
        episodes.append(episode)

        collected["observations"].append(episode_observations[:-1][keep])
        collected["actions"].append(episode.actions[keep])
        collected["next_observations"].append(episode_observations[1:][keep])
        collected["clean_rewards"].append(clean[keep])
        collected["noisy_rewards"].append(noisy[keep])
        collected["terminals"].append(episode.terminals[keep])
        collected["timeouts"].append(episode.timeouts[keep])
        collected["episode_ids"].append(
            np.full(int(keep.sum()), episode_id, dtype=np.int64)
        )
        collected["source_indices"].append(source_indices[keep])
        collected["clean_return_current"].append(clean_returns[keep])
        collected["noisy_return_current"].append(noisy_returns[keep])
        collected["clean_return_next"].append(clean_next[keep])
        collected["noisy_return_next"].append(noisy_next[keep])
        collected["iid_noise"].append(iid[keep])
        clean_episode_scores.append(float(clean_next[keep].mean()))
        noisy_episode_scores.append(float(noisy_next[keep].mean()))
        episode_lengths.append(clean.size)

    flat = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    kept_order_sha256 = _hash_named_arrays(
        [
            ("observations", flat["observations"]),
            ("actions", flat["actions"]),
            ("next_observations", flat["next_observations"]),
            ("episode_ids", flat["episode_ids"]),
            ("source_indices", flat["source_indices"]),
        ]
    )
    return PreparedFlatDataset(
        observations=np.asarray(flat["observations"], dtype=np.float32),
        actions=np.asarray(flat["actions"], dtype=np.float32),
        next_observations=np.asarray(flat["next_observations"], dtype=np.float32),
        clean_rewards=np.asarray(flat["clean_rewards"], dtype=np.float32),
        noisy_rewards=np.asarray(flat["noisy_rewards"], dtype=np.float32),
        terminals=np.asarray(flat["terminals"], dtype=np.bool_),
        timeouts=np.asarray(flat["timeouts"], dtype=np.bool_),
        episode_ids=np.asarray(flat["episode_ids"], dtype=np.int64),
        source_indices=np.asarray(flat["source_indices"], dtype=np.int64),
        clean_return_current=np.asarray(flat["clean_return_current"], dtype=np.float32),
        noisy_return_current=np.asarray(flat["noisy_return_current"], dtype=np.float32),
        clean_return_next=np.asarray(flat["clean_return_next"], dtype=np.float32),
        noisy_return_next=np.asarray(flat["noisy_return_next"], dtype=np.float32),
        iid_noise=np.asarray(flat["iid_noise"], dtype=np.float32),
        episodes=tuple(episodes),
        episode_biases=episode_biases,
        clean_episode_scores=np.asarray(clean_episode_scores, dtype=np.float32),
        noisy_episode_scores=np.asarray(noisy_episode_scores, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        clean_reward_std=clean_reward_std,
        iid_noise_std=iid_noise_std,
        episode_bias_std=episode_bias_std,
        dataset_sha256=sha256_file(path),
        kept_order_sha256=kept_order_sha256,
    )


def make_episode_folds(episode_count: int, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if folds > episode_count:
        raise ValueError("folds cannot exceed the number of episodes")
    permutation = np.random.default_rng(seed).permutation(episode_count)
    assignments = np.empty(episode_count, dtype=np.int64)
    assignments[permutation] = np.arange(episode_count, dtype=np.int64) % folds
    return assignments


def _safe_mean_std(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values.mean(axis=0), dtype=np.float32)
    std = np.asarray(values.std(axis=0), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _fold_statistics(
    dataset: PreparedFlatDataset, train_indices: np.ndarray
) -> Dict[str, np.ndarray | float]:
    obs_mean, obs_std = _safe_mean_std(dataset.observations[train_indices])
    action_mean, action_std = _safe_mean_std(dataset.actions[train_indices])
    target = dataset.noisy_return_current[train_indices]
    target_mean = float(target.mean())
    target_std = max(float(target.std()), 1e-6)
    return {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def _normalized_features(
    dataset: PreparedFlatDataset,
    indices: np.ndarray,
    statistics: Mapping[str, np.ndarray | float],
    *,
    next_observations: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    observations = (
        dataset.next_observations[indices]
        if next_observations
        else dataset.observations[indices]
    )
    obs = (observations - statistics["obs_mean"]) / statistics["obs_std"]
    actions = (dataset.actions[indices] - statistics["action_mean"]) / statistics[
        "action_std"
    ]
    return np.asarray(obs, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def _member_training_indices(
    dataset: PreparedFlatDataset,
    train_episode_ids: np.ndarray,
    *,
    ensemble_size: int,
    seed: int,
) -> np.ndarray:
    if ensemble_size == 1:
        return np.flatnonzero(np.isin(dataset.episode_ids, train_episode_ids))
    rng = np.random.default_rng(seed)
    sampled_episodes = rng.choice(
        train_episode_ids, size=train_episode_ids.size, replace=True
    )
    chunks = [np.flatnonzero(dataset.episode_ids == episode) for episode in sampled_episodes]
    return np.concatenate(chunks, axis=0)


def _ridge_fit(features: np.ndarray, targets: np.ndarray, l2: float) -> np.ndarray:
    if l2 < 0.0:
        raise ValueError("ridge_l2 must be non-negative")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    design = np.concatenate((x, np.ones((x.shape[0], 1), dtype=np.float64)), axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    try:
        return np.linalg.solve(gram + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty, design.T @ y, rcond=None)[0]


def _ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return (x @ weights[:-1] + weights[-1]).astype(np.float32)


def _fit_ridge_member(
    dataset: PreparedFlatDataset,
    member_train_indices: np.ndarray,
    test_indices: np.ndarray,
    statistics: Mapping[str, np.ndarray | float],
    config: PredictorConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_train, action_train = _normalized_features(
        dataset, member_train_indices, statistics
    )
    target = (
        dataset.noisy_return_current[member_train_indices]
        - float(statistics["target_mean"])
    ) / float(statistics["target_std"])
    v_weights = _ridge_fit(obs_train, target, config.ridge_l2)
    q_weights = _ridge_fit(
        np.concatenate((obs_train, action_train), axis=1), target, config.ridge_l2
    )
    obs_current, action_current = _normalized_features(dataset, test_indices, statistics)
    obs_next, _ = _normalized_features(
        dataset, test_indices, statistics, next_observations=True
    )
    target_mean = float(statistics["target_mean"])
    target_std = float(statistics["target_std"])
    v_current = _ridge_predict(obs_current, v_weights) * target_std + target_mean
    q_current = (
        _ridge_predict(np.concatenate((obs_current, action_current), axis=1), q_weights)
        * target_std
        + target_mean
    )
    v_next = _ridge_predict(obs_next, v_weights) * target_std + target_mean
    return v_current, q_current, v_next


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("the mlp backend requires PyTorch") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fit_mlp_member(
    dataset: PreparedFlatDataset,
    member_train_indices: np.ndarray,
    test_indices: np.ndarray,
    statistics: Mapping[str, np.ndarray | float],
    config: PredictorConfig,
    seed: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised on server
        raise RuntimeError("the mlp backend requires PyTorch") from exc

    if config.train_steps < 1 or config.batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive")
    if config.depth < 1 or config.hidden_dim < 1:
        raise ValueError("depth and hidden_dim must be positive")

    class Regressor(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            width = input_dim
            for _ in range(config.depth):
                layers.extend((nn.Linear(width, config.hidden_dim), nn.SiLU()))
                width = config.hidden_dim
            layers.append(nn.Linear(width, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, value):
            return self.network(value).squeeze(-1)

    _seed_torch(seed)
    torch_device = torch.device(device)
    v_model = Regressor(dataset.observations.shape[1]).to(torch_device)
    q_model = Regressor(dataset.observations.shape[1] + dataset.actions.shape[1]).to(
        torch_device
    )
    optimizer = torch.optim.AdamW(
        list(v_model.parameters()) + list(q_model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sample_rng = np.random.default_rng(seed + 104729)
    obs_mean = np.asarray(statistics["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(statistics["obs_std"], dtype=np.float32)
    action_mean = np.asarray(statistics["action_mean"], dtype=np.float32)
    action_std = np.asarray(statistics["action_std"], dtype=np.float32)
    target_mean = float(statistics["target_mean"])
    target_std = float(statistics["target_std"])

    v_model.train()
    q_model.train()
    for _ in range(config.train_steps):
        selected = member_train_indices[
            sample_rng.integers(0, member_train_indices.size, size=config.batch_size)
        ]
        obs = (dataset.observations[selected] - obs_mean) / obs_std
        action = (dataset.actions[selected] - action_mean) / action_std
        target = (dataset.noisy_return_current[selected] - target_mean) / target_std
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=torch_device)
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=torch_device)
        v_prediction = v_model(obs_tensor)
        q_prediction = q_model(torch.cat((obs_tensor, action_tensor), dim=1))
        loss = (v_prediction - target_tensor).square().mean()
        loss = loss + (q_prediction - target_tensor).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(v_model.parameters()) + list(q_model.parameters()), 10.0
        )
        optimizer.step()

    def predict(kind: str) -> np.ndarray:
        chunks: List[np.ndarray] = []
        v_model.eval()
        q_model.eval()
        with torch.no_grad():
            for start in range(0, test_indices.size, config.prediction_batch_size):
                indices = test_indices[start : start + config.prediction_batch_size]
                raw_obs = (
                    dataset.next_observations[indices]
                    if kind == "v_next"
                    else dataset.observations[indices]
                )
                obs = (raw_obs - obs_mean) / obs_std
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
                if kind == "q_current":
                    action = (dataset.actions[indices] - action_mean) / action_std
                    action_tensor = torch.as_tensor(
                        action, dtype=torch.float32, device=torch_device
                    )
                    prediction = q_model(torch.cat((obs_tensor, action_tensor), dim=1))
                else:
                    prediction = v_model(obs_tensor)
                chunks.append(prediction.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks) * target_std + target_mean

    return predict("v_current"), predict("q_current"), predict("v_next")


def crossfit_predict(
    dataset: PreparedFlatDataset, config: PredictorConfig
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Fit OOF regressors and return archive arrays plus JSON metadata."""

    if config.ensemble_size < 1:
        raise ValueError("ensemble_size must be positive")
    if config.backend not in ("mlp", "ridge"):
        raise ValueError("backend must be 'mlp' or 'ridge'")
    episode_count = len(dataset.episodes)
    fold_ids = make_episode_folds(episode_count, config.folds, config.fold_seed)
    transition_count = int(dataset.observations.shape[0])
    member_shape = (config.ensemble_size, transition_count)
    v_current_members = np.full(member_shape, np.nan, dtype=np.float32)
    q_current_members = np.full(member_shape, np.nan, dtype=np.float32)
    v_next_members = np.full(member_shape, np.nan, dtype=np.float32)
    provenance: List[Dict[str, object]] = []
    resolved_device = _resolve_device(config.device) if config.backend == "mlp" else "cpu"

    for fold in range(config.folds):
        test_episode_ids = np.flatnonzero(fold_ids == fold)
        train_episode_ids = np.flatnonzero(fold_ids != fold)
        if np.intersect1d(train_episode_ids, test_episode_ids).size:
            raise AssertionError("episode leakage detected before fitting")
        train_indices = np.flatnonzero(np.isin(dataset.episode_ids, train_episode_ids))
        test_indices = np.flatnonzero(np.isin(dataset.episode_ids, test_episode_ids))
        if train_indices.size == 0 or test_indices.size == 0:
            raise ValueError(f"fold {fold} has an empty train/test transition set")
        statistics = _fold_statistics(dataset, train_indices)
        member_seeds: List[int] = []
        for member in range(config.ensemble_size):
            member_seed = int(config.model_seed + 1009 * fold + 104729 * member)
            member_seeds.append(member_seed)
            member_train_indices = _member_training_indices(
                dataset,
                train_episode_ids,
                ensemble_size=config.ensemble_size,
                seed=member_seed + 17,
            )
            if np.intersect1d(
                np.unique(dataset.episode_ids[member_train_indices]), test_episode_ids
            ).size:
                raise AssertionError("held-out episode entered a member training set")
            if config.backend == "ridge":
                v_current, q_current, v_next = _fit_ridge_member(
                    dataset, member_train_indices, test_indices, statistics, config
                )
            else:
                v_current, q_current, v_next = _fit_mlp_member(
                    dataset,
                    member_train_indices,
                    test_indices,
                    statistics,
                    config,
                    member_seed,
                    resolved_device,
                )
            v_current_members[member, test_indices] = v_current
            q_current_members[member, test_indices] = q_current
            v_next_members[member, test_indices] = v_next

        normalization_hash = _hash_named_arrays(
            [
                ("obs_mean", np.asarray(statistics["obs_mean"])),
                ("obs_std", np.asarray(statistics["obs_std"])),
                ("action_mean", np.asarray(statistics["action_mean"])),
                ("action_std", np.asarray(statistics["action_std"])),
                ("target_mean", np.asarray([statistics["target_mean"]], dtype=np.float64)),
                ("target_std", np.asarray([statistics["target_std"]], dtype=np.float64)),
            ]
        )
        provenance.append(
            {
                "fold": fold,
                "train_episode_ids": train_episode_ids.tolist(),
                "heldout_episode_ids": test_episode_ids.tolist(),
                "train_transitions": int(train_indices.size),
                "heldout_transitions": int(test_indices.size),
                "member_seeds": member_seeds,
                "normalization_sha256": normalization_hash,
                "target_mean_raw": float(statistics["target_mean"]),
                "target_std_raw": float(statistics["target_std"]),
            }
        )

    for name, predictions in (
        ("v_current", v_current_members),
        ("q_current", q_current_members),
        ("v_next", v_next_members),
    ):
        if not np.all(np.isfinite(predictions)):
            missing = int(np.size(predictions) - np.isfinite(predictions).sum())
            raise RuntimeError(f"{name} contains {missing} unassigned/non-finite values")

    v_current = v_current_members.mean(axis=0).astype(np.float32)
    q_current = q_current_members.mean(axis=0).astype(np.float32)
    v_next = v_next_members.mean(axis=0).astype(np.float32)
    v_current_std = v_current_members.std(axis=0).astype(np.float32)
    q_current_std = q_current_members.std(axis=0).astype(np.float32)
    v_next_std = v_next_members.std(axis=0).astype(np.float32)

    episode_v_next_members = np.empty((config.ensemble_size, episode_count), dtype=np.float32)
    episode_v_current_members = np.empty_like(episode_v_next_members)
    episode_q_members = np.empty_like(episode_v_next_members)
    episode_controllable_members = np.empty_like(episode_v_next_members)
    for episode_id in range(episode_count):
        indices = np.flatnonzero(dataset.episode_ids == episode_id)
        episode_v_next_members[:, episode_id] = v_next_members[:, indices].mean(axis=1)
        episode_v_current_members[:, episode_id] = v_current_members[:, indices].mean(axis=1)
        episode_q_members[:, episode_id] = q_current_members[:, indices].mean(axis=1)
        episode_controllable_members[:, episode_id] = (
            q_current_members[:, indices] - v_current_members[:, indices]
        ).mean(axis=1)

    episode_crossfit_mean = episode_v_next_members.mean(axis=0).astype(np.float32)
    episode_crossfit_std = episode_v_next_members.std(axis=0).astype(np.float32)
    controllable_score = episode_controllable_members.mean(axis=0).astype(np.float32)
    controllable_std = episode_controllable_members.std(axis=0).astype(np.float32)
    kept_fold_ids = fold_ids[dataset.episode_ids]
    fold_integrity = all(
        not set(item["train_episode_ids"]).intersection(item["heldout_episode_ids"])
        for item in provenance
    )
    if not fold_integrity:
        raise AssertionError("fold provenance failed disjointness validation")

    arrays: Dict[str, np.ndarray] = {
        "v_current": v_current,
        "q_current": q_current,
        "v_next": v_next,
        "v_current_std": v_current_std,
        "q_current_std": q_current_std,
        "v_next_std": v_next_std,
        # Compatibility keys consumed by train_iql.py.
        "crossfit_next_mean": v_next,
        "crossfit_next_std": v_next_std,
        "crossfit_mean": episode_crossfit_mean,
        "crossfit_std": episode_crossfit_std,
        "episode_crossfit_mean": episode_crossfit_mean,
        "episode_crossfit_std": episode_crossfit_std,
        "episode_v_current_mean": episode_v_current_members.mean(axis=0).astype(np.float32),
        "episode_q_current_mean": episode_q_members.mean(axis=0).astype(np.float32),
        "controllable_score": controllable_score,
        "controllable_std": controllable_std,
        "fold_ids": fold_ids,
        "kept_fold_ids": kept_fold_ids.astype(np.int64),
        "episode_ids": dataset.episode_ids,
        "kept_source_indices": dataset.source_indices,
        "clean_rewards": dataset.clean_rewards,
        "noisy_rewards": dataset.noisy_rewards,
        "iid_noise": dataset.iid_noise,
        "clean_return_current": dataset.clean_return_current,
        "noisy_return_current": dataset.noisy_return_current,
        "clean_return_next": dataset.clean_return_next,
        "noisy_return_next": dataset.noisy_return_next,
        "clean_episode_scores": dataset.clean_episode_scores,
        "noisy_episode_scores": dataset.noisy_episode_scores,
        "episode_biases": dataset.episode_biases,
        "episode_lengths": dataset.episode_lengths,
        "prediction_units": np.asarray("raw"),
        "dataset_sha256": np.asarray(dataset.dataset_sha256),
        "kept_order_sha256": np.asarray(dataset.kept_order_sha256),
    }
    metadata: Dict[str, object] = {
        "schema_version": 1,
        "method": "episode-fold-crossfit-qv-return-predictor",
        "prediction_units": "raw",
        "episode_crossfit_definition": "mean held-out V(next_state) over kept transitions",
        "controllable_score_definition": "mean held-out [Q(state, action)-V(state)] over kept transitions",
        "uncertainty_definition": "population std across trajectory-bootstrap ensemble members",
        "target_definition": "noisy discounted return-to-go G_t",
        "transition_contract": (
            "D4RL flat order; split at terminal|timeout; keep true-terminal final row; "
            "drop timeout/incomplete final row"
        ),
        "config": asdict(config),
        "resolved_device": resolved_device,
        "dataset_sha256": dataset.dataset_sha256,
        "kept_order_sha256": dataset.kept_order_sha256,
        "source_code_sha256": sha256_file(Path(__file__)),
        "episodes": episode_count,
        "kept_transitions": transition_count,
        "observation_dim": int(dataset.observations.shape[1]),
        "action_dim": int(dataset.actions.shape[1]),
        "clean_reward_std_raw": dataset.clean_reward_std,
        "iid_noise_std_raw": dataset.iid_noise_std,
        "episode_bias_std_raw": dataset.episode_bias_std,
        "episode_bias_realized_std_raw": float(dataset.episode_biases.std()),
        "iid_noise_realized_std_raw": float(dataset.iid_noise.std()),
        "fold_integrity": fold_integrity,
        "fold_provenance": provenance,
    }
    arrays["config_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return arrays, metadata


def save_predictions(
    output: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> Dict[str, object]:
    """Atomically save the NPZ and a checksum-bearing JSON sidecar."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output)
    manifest = dict(metadata)
    manifest["archive"] = str(output.resolve())
    manifest["archive_sha256"] = sha256_file(output)
    sidecar = output.with_suffix(".json")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(sidecar_temporary, sidecar)
    return manifest


def run(args: argparse.Namespace) -> Dict[str, object]:
    config = PredictorConfig(
        discount=args.discount,
        noise_seed=args.noise_seed,
        iid_noise_scale=args.iid_noise_scale,
        episode_noise_scale=args.episode_noise_scale,
        folds=args.folds,
        fold_seed=args.fold_seed,
        model_seed=args.model_seed,
        ensemble_size=args.ensemble_size,
        backend=args.backend,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ridge_l2=args.ridge_l2,
        prediction_batch_size=args.prediction_batch_size,
        device=args.device,
        max_episodes=args.max_episodes,
    )
    started = time.perf_counter()
    dataset = prepare_flat_dataset(
        args.dataset,
        discount=config.discount,
        noise_seed=config.noise_seed,
        iid_noise_scale=config.iid_noise_scale,
        episode_noise_scale=config.episode_noise_scale,
        max_episodes=config.max_episodes,
    )
    arrays, metadata = crossfit_predict(dataset, config)
    metadata["wall_time_seconds"] = float(time.perf_counter() - started)
    # Refresh the embedded record after adding the measured wall time.
    arrays["config_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return save_predictions(args.output, arrays, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--noise-seed", type=int, default=20260914)
    parser.add_argument("--iid-noise-scale", type=float, default=0.0)
    parser.add_argument("--episode-noise-scale", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=31)
    parser.add_argument("--model-seed", type=int, default=47)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--backend", choices=("mlp", "ridge"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument("--prediction-batch-size", type=int, default=65536)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int)
    return parser


def main() -> None:
    manifest = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "archive": manifest["archive"],
                "archive_sha256": manifest["archive_sha256"],
                "episodes": manifest["episodes"],
                "kept_transitions": manifest["kept_transitions"],
                "wall_time_seconds": manifest["wall_time_seconds"],
                "fold_integrity": manifest["fold_integrity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
