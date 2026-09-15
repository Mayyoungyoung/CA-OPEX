"""Clean-room IQL and uncertainty-calibrated HUBL primitives.

The implementation follows the equations in the IQL and HUBL papers but does
not copy the unlicensed HUBL supplementary source.  Data preparation and the
training CLI live in :mod:`train_iql`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal


def normalized_rank(values: np.ndarray) -> np.ndarray:
    """Return average ranks in [0, 1], matching HUBL's trajectory rank rule."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot rank an empty array")
    if values.size == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return (ranks / (values.size - 1)).astype(np.float32)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except ImportError:  # pragma: no cover - SciPy is available on the GPU host
        return np.vectorize(lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))(x)


def posterior_expected_rank(
    means: np.ndarray,
    stds: np.ndarray,
    *,
    block_size: int = 256,
    minimum_std: float = 1e-6,
) -> np.ndarray:
    """Expected empirical percentile for independent Gaussian return beliefs.

    For trajectory ``i``, the result is

        mean_{j != i} Phi((mu_i-mu_j) / sqrt(sigma_i^2+sigma_j^2)).

    Unlike ranking point estimates, uncertain extremes shrink toward the middle
    rather than being promoted solely by a lucky observed return.
    """

    means = np.asarray(means, dtype=np.float64).reshape(-1)
    stds = np.maximum(np.asarray(stds, dtype=np.float64).reshape(-1), minimum_std)
    if means.shape != stds.shape:
        raise ValueError("means and stds must have the same shape")
    n = means.size
    if n == 0:
        raise ValueError("cannot rank an empty array")
    if n == 1:
        return np.ones(1, dtype=np.float32)
    result = np.empty(n, dtype=np.float64)
    variance = np.square(stds)
    for begin in range(0, n, block_size):
        stop = min(begin + block_size, n)
        delta = means[begin:stop, None] - means[None, :]
        denom = np.sqrt(variance[begin:stop, None] + variance[None, :])
        probabilities = _normal_cdf(delta / np.maximum(denom, minimum_std))
        # Self-comparisons equal 0.5 and are removed from the sum.
        result[begin:stop] = (probabilities.sum(axis=1) - 0.5) / (n - 1)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def expand_episode_values(values: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if episode_ids.size and (episode_ids.min() < 0 or episode_ids.max() >= values.size):
        raise ValueError("episode id is outside episode-value array")
    return values[episode_ids]


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be positive")
    layers = []
    previous = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(previous, hidden_dim), nn.ReLU()))
        previous = hidden_dim
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.q1 = make_mlp(observation_dim + action_dim, 1, hidden_dim, depth)
        self.q2 = make_mlp(observation_dim + action_dim, 1, hidden_dim, depth)

    def both(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observations, actions), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(observations, actions)
        return torch.minimum(q1, q2)


class ValueFunction(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.network = make_mlp(observation_dim, 1, hidden_dim, depth)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.mean_network = make_mlp(observation_dim, action_dim, hidden_dim, depth)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, observations: torch.Tensor) -> Independent:
        means = self.mean_network(observations)
        stds = self.log_std.clamp(-5.0, 2.0).exp().expand_as(means)
        return Independent(Normal(means, stds), 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mean_network(observations)


def expectile_loss(residual: torch.Tensor, expectile: float) -> torch.Tensor:
    weights = torch.where(residual > 0, expectile, 1.0 - expectile)
    return (weights * residual.square()).mean()


@dataclass(frozen=True)
class IQLConfig:
    observation_dim: int
    action_dim: int
    hidden_dim: int = 256
    depth: int = 2
    learning_rate: float = 3e-4
    discount: float = 0.99
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    target_rate: float = 0.005
    max_advantage_weight: float = 100.0


class IQLAgent:
    """IQL with an optional fixed per-transition HUBL heuristic and weight."""

    def __init__(self, config: IQLConfig, device: torch.device):
        self.config = config
        self.device = device
        self.q = TwinQ(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.depth,
        ).to(device)
        self.target_q = copy.deepcopy(self.q).requires_grad_(False).to(device)
        self.value = ValueFunction(config.observation_dim, config.hidden_dim, config.depth).to(device)
        self.policy = GaussianPolicy(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.depth,
        ).to(device)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=config.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        heuristic_next = batch["heuristic_next"]
        lambdas = batch["lambdas"]

        with torch.no_grad():
            target_q = self.target_q(observations, actions)
        values = self.value(observations)
        advantages = target_q - values
        value_loss = expectile_loss(advantages, self.config.expectile)
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            next_values = self.value(next_observations)
            mixed_next = lambdas * heuristic_next + (1.0 - lambdas) * next_values
            q_target = rewards + (1.0 - terminals) * self.config.discount * mixed_next
        q1, q2 = self.q.both(observations, actions)
        q_loss = 0.5 * ((q1 - q_target).square().mean() + (q2 - q_target).square().mean())
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        weights = torch.exp(self.config.advantage_temperature * advantages.detach()).clamp(
            max=self.config.max_advantage_weight
        )
        log_probabilities = self.policy.distribution(observations).log_prob(actions)
        policy_loss = -(weights * log_probabilities).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                self.target_q.parameters(), self.q.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.target_rate)
                target_parameter.add_(source_parameter, alpha=self.config.target_rate)

        return {
            "value_loss": float(value_loss.detach()),
            "q_loss": float(q_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "advantage_mean": float(advantages.detach().mean()),
            "advantage_weight_mean": float(weights.mean()),
            "q_target_mean": float(q_target.mean()),
            "heuristic_mean": float(heuristic_next.mean()),
            "lambda_mean": float(lambdas.mean()),
        }

    @torch.no_grad()
    def act(self, observation: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        action = self.policy(tensor).squeeze(0).cpu().numpy()
        return np.clip(action, action_low, action_high)

    def checkpoint(self) -> Dict[str, object]:
        return {
            "config": self.config.__dict__,
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
