"""Core utilities for an out-of-fold reward-noise mechanism audit.

This module is intentionally independent of the HUBL supplementary code.  It
uses only NumPy/HDF5 for data handling and imports PyTorch lazily for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import random

import h5py
import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """One trajectory reconstructed from a flat D4RL transition table."""

    start: int
    stop: int  # exclusive
    observations: np.ndarray
    actions: np.ndarray
    clean_rewards: np.ndarray
    noisy_rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray

    @property
    def length(self) -> int:
        return self.stop - self.start


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def load_d4rl_hdf5(path: Path) -> Dict[str, np.ndarray]:
    """Load the minimal D4RL transition fields and validate their lengths."""

    required = ("observations", "actions", "rewards", "terminals")
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"missing required HDF5 keys: {missing}")
        arrays = {key: np.asarray(handle[key]) for key in required}
        arrays["timeouts"] = (
            np.asarray(handle["timeouts"])
            if "timeouts" in handle
            else np.zeros_like(arrays["terminals"], dtype=np.bool_)
        )

    n = int(arrays["rewards"].shape[0])
    if n == 0:
        raise ValueError("dataset has no transitions")
    for key, value in arrays.items():
        if value.shape[0] != n:
            raise ValueError(f"{key} has {value.shape[0]} rows, expected {n}")
    arrays["rewards"] = np.asarray(arrays["rewards"], dtype=np.float32).reshape(-1)
    arrays["terminals"] = np.asarray(arrays["terminals"], dtype=np.bool_).reshape(-1)
    arrays["timeouts"] = np.asarray(arrays["timeouts"], dtype=np.bool_).reshape(-1)
    arrays["observations"] = np.asarray(arrays["observations"], dtype=np.float32)
    arrays["actions"] = np.asarray(arrays["actions"], dtype=np.float32)
    if arrays["observations"].ndim != 2 or arrays["actions"].ndim != 2:
        raise ValueError("observations and actions must be rank-2 arrays")
    return arrays


def inject_reward_noise(
    clean_rewards: np.ndarray,
    noise_std: float,
    seed: int,
    distribution: str = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject reproducible additive noise, returning (noisy, realized_noise)."""

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        noise = rng.normal(0.0, noise_std, size=clean_rewards.shape)
    elif distribution == "uniform":
        # Same variance as N(0, noise_std**2).
        bound = math.sqrt(3.0) * noise_std
        noise = rng.uniform(-bound, bound, size=clean_rewards.shape)
    else:
        raise ValueError(f"unsupported noise distribution: {distribution}")
    noise = noise.astype(np.float32)
    return (clean_rewards.astype(np.float32) + noise).astype(np.float32), noise


def reconstruct_trajectories(
    arrays: Mapping[str, np.ndarray], noisy_rewards: np.ndarray
) -> List[Trajectory]:
    """Split at terminal/timeout boundaries and retain a trailing trajectory."""

    n = int(arrays["rewards"].shape[0])
    if noisy_rewards.shape != (n,):
        raise ValueError(f"noisy_rewards shape {noisy_rewards.shape}, expected {(n,)}")
    boundaries = np.asarray(arrays["terminals"] | arrays["timeouts"], dtype=np.bool_)
    stops = list(np.flatnonzero(boundaries) + 1)
    if not stops or stops[-1] != n:
        stops.append(n)
    trajectories: List[Trajectory] = []
    start = 0
    for stop in stops:
        stop_i = int(stop)
        if stop_i <= start:
            continue
        trajectories.append(
            Trajectory(
                start=start,
                stop=stop_i,
                observations=arrays["observations"][start:stop_i],
                actions=arrays["actions"][start:stop_i],
                clean_rewards=arrays["rewards"][start:stop_i],
                noisy_rewards=noisy_rewards[start:stop_i],
                terminals=arrays["terminals"][start:stop_i],
                timeouts=arrays["timeouts"][start:stop_i],
            )
        )
        start = stop_i
    if not trajectories:
        raise ValueError("could not reconstruct any trajectories")
    return trajectories


def discounted_return(rewards: np.ndarray, gamma: float) -> float:
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    powers = np.power(np.float64(gamma), np.arange(rewards.shape[0], dtype=np.float64))
    return float(np.dot(powers, rewards.astype(np.float64)))


def trajectory_features(trajectories: Sequence[Trajectory]) -> np.ndarray:
    """Reward-free descriptors for predicting the time-zero return-to-go.

    Features concatenate initial/final/mean/std observation, mean/std action,
    and log trajectory length.  No reward-derived value leaks into the model.
    """

    rows = []
    for trajectory in trajectories:
        obs = trajectory.observations.astype(np.float64)
        act = trajectory.actions.astype(np.float64)
        rows.append(
            np.concatenate(
                [
                    obs[0],
                    obs[-1],
                    obs.mean(axis=0),
                    obs.std(axis=0),
                    act.mean(axis=0),
                    act.std(axis=0),
                    np.asarray([math.log1p(trajectory.length)], dtype=np.float64),
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)


def make_trajectory_folds(n_trajectories: int, n_folds: int, seed: int) -> np.ndarray:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > n_trajectories:
        raise ValueError("n_folds cannot exceed the number of trajectories")
    permutation = np.random.default_rng(seed).permutation(n_trajectories)
    folds = np.empty(n_trajectories, dtype=np.int64)
    folds[permutation] = np.arange(n_trajectories, dtype=np.int64) % n_folds
    return folds


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_crossfit_heteroscedastic_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    hidden_dim: int = 64,
    epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    model_seed: int = 0,
    device: str = "cpu",
    calibration_fraction: float = 0.2,
    early_stopping_patience: int = 40,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Fit one model per held-out trajectory fold and return OOF mean/std.

    Normalization statistics are also fit on the training folds only.  The
    returned provenance lists exact train/test trajectory ids for auditability.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised on GPU host
        raise RuntimeError("PyTorch is required for model fitting") from exc

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    folds = np.asarray(folds, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets have incompatible shapes")
    if folds.shape[0] != targets.shape[0]:
        raise ValueError("folds/targets have incompatible shapes")

    class HeteroscedasticMLP(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, x):
            output = self.net(x)
            return output[:, 0], output[:, 1].clamp(-7.0, 7.0)

    torch_device = torch.device(device)
    means = np.full(targets.shape, np.nan, dtype=np.float32)
    stds = np.full(targets.shape, np.nan, dtype=np.float32)
    provenance: List[Dict[str, object]] = []
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be in (0, 0.5)")
    unique_folds = np.unique(folds)
    for fold in unique_folds:
        train_ids = np.flatnonzero(folds != fold)
        test_ids = np.flatnonzero(folds == fold)
        if train_ids.size == 0 or test_ids.size == 0:
            raise ValueError(f"fold {fold} has empty train or test partition")

        fold_seed = int(model_seed + 1009 * int(fold))
        split_rng = np.random.default_rng(fold_seed + 7919)
        shuffled_train = split_rng.permutation(train_ids)
        calibration_size = max(1, int(round(calibration_fraction * train_ids.size)))
        if train_ids.size - calibration_size < 2:
            raise ValueError("not enough training trajectories for inner calibration")
        calibration_ids = np.sort(shuffled_train[:calibration_size])
        fit_ids = np.sort(shuffled_train[calibration_size:])

        x_mean = features[fit_ids].mean(axis=0)
        x_std = features[fit_ids].std(axis=0)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        y_mean = float(targets[fit_ids].mean())
        y_std = float(targets[fit_ids].std())
        y_std = max(y_std, 1e-6)
        x_train = (features[fit_ids] - x_mean) / x_std
        y_train = (targets[fit_ids] - y_mean) / y_std
        x_calibration = (features[calibration_ids] - x_mean) / x_std
        x_test = (features[test_ids] - x_mean) / x_std

        _seed_everything(fold_seed)
        model = HeteroscedasticMLP(features.shape[1]).to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
        y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=torch_device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(fold_seed)
        calibration_tensor = torch.as_tensor(
            x_calibration, dtype=torch.float32, device=torch_device
        )
        calibration_target_tensor = torch.as_tensor(
            (targets[calibration_ids] - y_mean) / y_std,
            dtype=torch.float32,
            device=torch_device,
        )
        final_loss = float("nan")
        best_calibration_nll = float("inf")
        best_epoch = -1
        best_state = None
        stale_epochs = 0
        model.train()
        for epoch in range(epochs):
            order = torch.randperm(fit_ids.size, generator=generator)
            for begin in range(0, fit_ids.size, batch_size):
                selection = order[begin : begin + batch_size].to(torch_device)
                batch_x = x_tensor[selection]
                batch_y = y_tensor[selection]
                mean_norm, log_variance = model(batch_x)
                inverse_variance = torch.exp(-log_variance)
                loss = 0.5 * (
                    inverse_variance * (batch_y - mean_norm).square() + log_variance
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                final_loss = float(loss.detach().cpu())

            model.eval()
            with torch.no_grad():
                calibration_mean_norm, calibration_log_variance = model(calibration_tensor)
                calibration_nll = float(
                    (
                        0.5
                        * (
                            torch.exp(-calibration_log_variance)
                            * (calibration_target_tensor - calibration_mean_norm).square()
                            + calibration_log_variance
                        )
                    )
                    .mean()
                    .cpu()
                )
            if calibration_nll < best_calibration_nll - 1e-5:
                best_calibration_nll = calibration_nll
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break
            model.train()

        if best_state is None:
            raise RuntimeError("early stopping did not retain a finite model")
        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            calibration_mean_norm, calibration_log_variance = model(calibration_tensor)
            calibration_mean = calibration_mean_norm.cpu().numpy() * y_std + y_mean
            calibration_std = np.exp(
                0.5 * calibration_log_variance.cpu().numpy()
            ) * y_std
            standardized_residual = (
                targets[calibration_ids] - calibration_mean
            ) / np.maximum(calibration_std, 1e-6)
            # A single held-out RMS scale corrects variance collapse without
            # looking at the outer test fold or latent clean returns.
            uncertainty_scale = float(
                np.clip(np.sqrt(np.mean(np.square(standardized_residual))), 0.25, 20.0)
            )
            test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=torch_device)
            mean_norm, log_variance = model(test_tensor)
            fold_means = mean_norm.cpu().numpy() * y_std + y_mean
            fold_stds = (
                np.exp(0.5 * log_variance.cpu().numpy()) * y_std * uncertainty_scale
            )
        means[test_ids] = fold_means.astype(np.float32)
        stds[test_ids] = np.maximum(fold_stds, 1e-6).astype(np.float32)
        provenance.append(
            {
                "fold": int(fold),
                "seed": fold_seed,
                "train_trajectory_ids": train_ids.tolist(),
                "model_fit_trajectory_ids": fit_ids.tolist(),
                "uncertainty_calibration_trajectory_ids": calibration_ids.tolist(),
                "test_trajectory_ids": test_ids.tolist(),
                "final_training_nll": final_loss,
                "best_calibration_nll": best_calibration_nll,
                "best_epoch": best_epoch,
                "target_train_mean": y_mean,
                "target_train_std": y_std,
                "uncertainty_scale": uncertainty_scale,
            }
        )

    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError("some trajectories did not receive finite OOF predictions")
    return means, stds, provenance


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, implemented without SciPy."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = _rankdata(np.asarray(x).reshape(-1))
    y_rank = _rankdata(np.asarray(y).reshape(-1))
    if x_rank.std() == 0 or y_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def top_fraction_precision(score: np.ndarray, truth: np.ndarray, fraction: float = 0.2) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = int(np.asarray(score).size)
    k = max(1, int(math.ceil(fraction * n)))
    predicted = set(np.argsort(score)[-k:].tolist())
    actual = set(np.argsort(truth)[-k:].tolist())
    return len(predicted & actual) / k


def audit_metrics(
    clean_returns: np.ndarray,
    raw_noisy_returns: np.ndarray,
    crossfit_mean: np.ndarray,
    crossfit_std: np.ndarray,
    lcb: np.ndarray,
) -> Dict[str, object]:
    clean = np.asarray(clean_returns, dtype=np.float64)
    std = np.maximum(np.asarray(crossfit_std, dtype=np.float64), 1e-8)

    def score_metrics(score: np.ndarray) -> Dict[str, float]:
        score = np.asarray(score, dtype=np.float64)
        return {
            "spearman_vs_clean": spearman_correlation(score, clean),
            "top20_precision_vs_clean": top_fraction_precision(score, clean, 0.2),
            "mse_vs_clean": float(np.mean(np.square(score - clean))),
        }

    z = (clean - np.asarray(crossfit_mean, dtype=np.float64)) / std
    metrics: Dict[str, object] = {
        "raw_noisy_return": score_metrics(raw_noisy_returns),
        "crossfit_mean": score_metrics(crossfit_mean),
        "lcb": score_metrics(lcb),
        "calibration_against_latent_clean": {
            "coverage_68": float(np.mean(np.abs(z) <= 1.0)),
            "coverage_95": float(np.mean(np.abs(z) <= 1.96)),
            "mean_abs_z": float(np.mean(np.abs(z))),
            "root_mean_square_z": float(np.sqrt(np.mean(np.square(z)))),
            "mean_predicted_std": float(np.mean(std)),
        },
    }
    return metrics


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
