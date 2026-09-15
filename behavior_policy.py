"""Clean-room loader for behavior policies embedded in D4RL v2 HDF5 files.

The locomotion datasets store a two-hidden-layer SAC policy under
``metadata/policy``.  This module has no dependency on the legacy D4RL package
or MuJoCo and validates the reconstruction against ``infos/action_log_probs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import math

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _activation(name: str, value: Tensor) -> Tensor:
    normalized = name.strip().lower()
    if normalized == "relu":
        return F.relu(value)
    if normalized == "tanh":
        return torch.tanh(value)
    if normalized in {"elu", "exponential_linear_unit"}:
        return F.elu(value)
    raise ValueError(f"unsupported policy nonlinearity: {name!r}")


@dataclass(frozen=True)
class LogProbAudit:
    sample_count: int
    seed: int
    mean_absolute_error: float
    root_mean_squared_error: float
    max_absolute_error: float
    pearson_correlation: float
    mean_error: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "seed": self.seed,
            "mean_absolute_error": self.mean_absolute_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "max_absolute_error": self.max_absolute_error,
            "pearson_correlation": self.pearson_correlation,
            "mean_error": self.mean_error,
        }


class D4RLTanhGaussianPolicy(nn.Module):
    """Frozen SAC policy reconstructed from D4RL metadata.

    D4RL stores weights in PyTorch/linear order ``(out_features, in_features)``.
    The final mean and log-standard-deviation heads parameterize a Gaussian in
    pre-tanh space.  Actions are therefore ``tanh(mean + std * epsilon)``.
    """

    def __init__(
        self,
        weights: Dict[str, np.ndarray],
        *,
        nonlinearity: str,
        output_distribution: str,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if output_distribution.strip().lower() != "tanh_gaussian":
            raise ValueError(
                "expected tanh_gaussian output distribution, got "
                f"{output_distribution!r}"
            )
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be less than log_std_max")
        self.nonlinearity = nonlinearity.strip().lower()
        # Validate the metadata eagerly rather than failing after collection.
        _activation(self.nonlinearity, torch.zeros(1))
        self.output_distribution = output_distribution.strip().lower()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        required = (
            "fc0/weight",
            "fc0/bias",
            "fc1/weight",
            "fc1/bias",
            "last_fc/weight",
            "last_fc/bias",
            "last_fc_log_std/weight",
            "last_fc_log_std/bias",
        )
        missing = [name for name in required if name not in weights]
        if missing:
            raise KeyError(f"missing policy arrays: {missing}")
        for name in required:
            self.register_buffer(
                name.replace("/", "__"),
                torch.as_tensor(weights[name], dtype=torch.float32).clone(),
            )
        self._validate_shapes()

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> "D4RLTanhGaussianPolicy":
        dataset_path = Path(path)
        prefix = "metadata/policy"
        with h5py.File(dataset_path, "r") as handle:
            if prefix not in handle:
                raise KeyError(f"{dataset_path} has no {prefix} group")
            policy_group = handle[prefix]
            nonlinearity = _decode_scalar(policy_group["nonlinearity"][()])
            output_distribution = _decode_scalar(
                policy_group["output_distribution"][()]
            )
            weights = {}
            for layer in ("fc0", "fc1", "last_fc", "last_fc_log_std"):
                for parameter in ("weight", "bias"):
                    name = f"{layer}/{parameter}"
                    weights[name] = np.asarray(policy_group[name], dtype=np.float32)
        policy = cls(
            weights,
            nonlinearity=nonlinearity,
            output_distribution=output_distribution,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        return policy.to(device).eval()

    def _buffer(self, name: str) -> Tensor:
        return getattr(self, name.replace("/", "__"))

    def _validate_shapes(self) -> None:
        w0, b0 = self._buffer("fc0/weight"), self._buffer("fc0/bias")
        w1, b1 = self._buffer("fc1/weight"), self._buffer("fc1/bias")
        wm, bm = self._buffer("last_fc/weight"), self._buffer("last_fc/bias")
        ws, bs = (
            self._buffer("last_fc_log_std/weight"),
            self._buffer("last_fc_log_std/bias"),
        )
        if any(t.ndim != 2 for t in (w0, w1, wm, ws)):
            raise ValueError("all policy weights must be rank-2")
        if any(t.ndim != 1 for t in (b0, b1, bm, bs)):
            raise ValueError("all policy biases must be rank-1")
        if w0.shape[0] != b0.shape[0]:
            raise ValueError("fc0 weight/bias shapes do not match")
        if w1.shape != (b1.shape[0], b0.shape[0]):
            raise ValueError("fc1 shapes do not match fc0")
        if wm.shape != (bm.shape[0], b1.shape[0]):
            raise ValueError("mean-head shapes do not match fc1")
        if ws.shape != wm.shape or bs.shape != bm.shape:
            raise ValueError("mean and log-std heads must have equal shapes")

    @property
    def observation_dim(self) -> int:
        return int(self._buffer("fc0/weight").shape[1])

    @property
    def action_dim(self) -> int:
        return int(self._buffer("last_fc/weight").shape[0])

    def forward(self, observations: Tensor) -> Tuple[Tensor, Tensor]:
        observations = torch.as_tensor(
            observations,
            dtype=self._buffer("fc0/weight").dtype,
            device=self._buffer("fc0/weight").device,
        )
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation dim {self.observation_dim}, "
                f"got {observations.shape[-1]}"
            )
        hidden = _activation(
            self.nonlinearity,
            F.linear(
                observations,
                self._buffer("fc0/weight"),
                self._buffer("fc0/bias"),
            ),
        )
        hidden = _activation(
            self.nonlinearity,
            F.linear(
                hidden,
                self._buffer("fc1/weight"),
                self._buffer("fc1/bias"),
            ),
        )
        mean = F.linear(
            hidden,
            self._buffer("last_fc/weight"),
            self._buffer("last_fc/bias"),
        )
        log_std = F.linear(
            hidden,
            self._buffer("last_fc_log_std/weight"),
            self._buffer("last_fc_log_std/bias"),
        ).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    @torch.no_grad()
    def deterministic_action(self, observations: Tensor) -> Tensor:
        mean, _ = self(observations)
        return torch.tanh(mean)

    @torch.no_grad()
    def sample_action(
        self, observations: Tensor, generator: torch.Generator
    ) -> Tuple[Tensor, Tensor]:
        mean, log_std = self(observations)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        pre_tanh = mean + log_std.exp() * noise
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(mean, log_std, pre_tanh, action)
        return action, log_prob

    def _log_prob_from_pre_tanh(
        self, mean: Tensor, log_std: Tensor, pre_tanh: Tensor, action: Tensor
    ) -> Tensor:
        gaussian = (
            -0.5 * ((pre_tanh - mean) / log_std.exp()).square()
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (gaussian - correction).sum(dim=-1)

    def log_prob(self, observations: Tensor, actions: Tensor) -> Tensor:
        mean, log_std = self(observations)
        actions = torch.as_tensor(
            actions, dtype=mean.dtype, device=mean.device
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(actions)
        return self._log_prob_from_pre_tanh(mean, log_std, pre_tanh, actions)


@torch.no_grad()
def audit_embedded_log_probs(
    dataset_path: str | Path,
    policy: Optional[D4RLTanhGaussianPolicy] = None,
    *,
    sample_count: int = 10_000,
    seed: int = 0,
) -> LogProbAudit:
    """Compare reconstructed densities with D4RL's recorded log probabilities."""

    path = Path(dataset_path)
    if policy is None:
        policy = D4RLTanhGaussianPolicy.from_hdf5(path)
    with h5py.File(path, "r") as handle:
        required = ("observations", "actions", "infos/action_log_probs")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"missing audit arrays: {missing}")
        total = int(handle["actions"].shape[0])
        if not 1 <= sample_count <= total:
            raise ValueError(f"sample_count must be in [1, {total}]")
        indices = np.sort(
            np.random.default_rng(seed).choice(total, sample_count, replace=False)
        )
        observations = np.asarray(handle["observations"][indices], dtype=np.float32)
        actions = np.asarray(handle["actions"][indices], dtype=np.float32)
        recorded = np.asarray(
            handle["infos/action_log_probs"][indices], dtype=np.float64
        )
    device = next(policy.buffers()).device
    reconstructed = (
        policy.log_prob(
            torch.as_tensor(observations, device=device),
            torch.as_tensor(actions, device=device),
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    errors = reconstructed - recorded
    correlation = (
        float(np.corrcoef(reconstructed, recorded)[0, 1])
        if sample_count > 1
        else float("nan")
    )
    return LogProbAudit(
        sample_count=sample_count,
        seed=seed,
        mean_absolute_error=float(np.mean(np.abs(errors))),
        root_mean_squared_error=float(np.sqrt(np.mean(errors**2))),
        max_absolute_error=float(np.max(np.abs(errors))),
        pearson_correlation=correlation,
        mean_error=float(np.mean(errors)),
    )
