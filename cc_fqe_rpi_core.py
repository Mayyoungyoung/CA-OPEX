"""Channel-consistent fitted-Q evaluation and one-step residual control.

This module is deliberately separate from the exploratory TD3+BC and
inverse-residual implementations.  It implements one narrowly scoped fallback:

1. initialise a physical-action critic and frozen desired-action actor from an
   executed/executed HUBL checkpoint;
2. fit the critic to the continuation of the inverse-only command policy under
   a declared execution channel; and
3. freeze that critic and take one bounded, support-regularised residual policy
   improvement step for the target channel.

For a physical action ``a`` and command ``u``, let

    g_beta(u, eps) = clip(u + eps),  eps ~ Uniform[-beta, beta]^d.

The fitted-Q target is

    y = r + gamma (1-terminal)
        min_i mean_j Q_i^-(s', g_beta_c(u0(s'), eps_j)),

where ``beta_c`` is either the target channel (the method) or the source
channel (the equal-compute continuation control).  There is no TD3 target
smoothing and no HUBL Monte-Carlo-return mixing.  The current critic regression
always uses the action that actually generated the logged transition.

The second stage keeps both critics fixed and optimises

    - mean[Qbar_min(u_delta) - Qbar_min(u0)] / frozen_Q_scale
    + residual_penalty * mean[(delta / delta_max)^2]
    + support_penalty * source_support_expansion(u_delta).

Common antithetic samples are used for the two command values.  The support
penalty is an auditable extrapolation control, not an identification guarantee.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from evaluation_controls import inverse_uniform_clipped_mean
from td3bc_core import DeterministicPolicy, TwinCritic, make_mlp


CONTINUATION_MODES = ("target", "source")


def canonical_torch_device(device: torch.device | str) -> torch.device:
    """Resolve implicit CUDA to the concrete current CUDA device."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


@dataclass(frozen=True)
class CCFQERPIConfig:
    """All quantities that alter the fitted operator or residual policy."""

    observation_dim: int
    action_dim: int
    base_hidden_dim: int = 256
    base_depth: int = 2
    adapter_hidden_dim: int = 128
    adapter_depth: int = 2
    max_action: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    source_beta: float = 1.0
    target_beta: float = 1.2498949
    continuation_mode: str = "target"
    train_channel_samples: int = 4
    audit_channel_samples: int = 64
    continuation_noise_seed: int = 314159
    actor_noise_seed: int = 314160
    audit_noise_seed: int = 314161
    fqe_updates: int = 20_000
    adapter_updates: int = 5_000
    critic_learning_rate: float = 3e-4
    adapter_learning_rate: float = 3e-4
    delta_max: float = 0.25
    value_alpha: float = 1.0
    residual_penalty: float = 1.0
    support_penalty: float = 1.0
    q_scale_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.base_hidden_dim <= 0 or self.base_depth <= 0:
            raise ValueError("base network dimensions must be positive")
        if self.adapter_hidden_dim <= 0 or self.adapter_depth <= 0:
            raise ValueError("adapter network dimensions must be positive")
        if self.max_action <= 0.0 or not np.isfinite(self.max_action):
            raise ValueError("max_action must be finite and positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must lie in [0, 1]")
        for name, value in (
            ("source_beta", self.source_beta),
            ("target_beta", self.target_beta),
        ):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.continuation_mode not in CONTINUATION_MODES:
            raise ValueError(
                f"continuation_mode must be one of {CONTINUATION_MODES}"
            )
        for name, count in (
            ("train_channel_samples", self.train_channel_samples),
            ("audit_channel_samples", self.audit_channel_samples),
        ):
            if count <= 0 or count % 2 != 0:
                raise ValueError(f"{name} must be a positive even count")
        for name, seed in (
            ("continuation_noise_seed", self.continuation_noise_seed),
            ("actor_noise_seed", self.actor_noise_seed),
            ("audit_noise_seed", self.audit_noise_seed),
        ):
            if seed < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.fqe_updates <= 0 or self.adapter_updates <= 0:
            raise ValueError("both fixed training stages must have positive updates")
        if self.critic_learning_rate <= 0.0 or self.adapter_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        for name, value in (
            ("delta_max", self.delta_max),
            ("value_alpha", self.value_alpha),
            ("residual_penalty", self.residual_penalty),
            ("support_penalty", self.support_penalty),
        ):
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.q_scale_epsilon <= 0.0 or not np.isfinite(self.q_scale_epsilon):
            raise ValueError("q_scale_epsilon must be finite and positive")

    @property
    def continuation_beta(self) -> float:
        return self.target_beta if self.continuation_mode == "target" else self.source_beta


class AntitheticUniformChannel:
    """Per-state antithetic channel with an isolated, checkpointable RNG."""

    def __init__(
        self,
        *,
        action_dim: int,
        sample_count: int,
        beta: float,
        max_action: float,
        seed: int,
        device: torch.device,
    ):
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if sample_count <= 0 or sample_count % 2:
            raise ValueError("antithetic sample_count must be positive and even")
        if beta < 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be finite and non-negative")
        if max_action <= 0.0:
            raise ValueError("max_action must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.action_dim = int(action_dim)
        self.sample_count = int(sample_count)
        self.beta = float(beta)
        self.max_action = float(max_action)
        self.seed = int(seed)
        self.device = canonical_torch_device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.draw_calls = 0
        self.half_vectors_drawn = 0

    def sample(
        self, batch_size: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = canonical_torch_device(device)
        if device != self.device:
            raise ValueError("sample device differs from channel generator device")
        half_count = self.sample_count // 2
        if self.beta == 0.0:
            half = torch.zeros(
                (batch_size, half_count, self.action_dim),
                dtype=dtype,
                device=device,
            )
        else:
            half = torch.rand(
                (batch_size, half_count, self.action_dim),
                generator=self.generator,
                dtype=dtype,
                device=device,
            )
            half = (2.0 * half - 1.0) * self.beta
            self.half_vectors_drawn += batch_size * half_count
        self.draw_calls += 1
        return torch.cat((half, -half), dim=1)

    def apply(self, commands: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        expected = (commands.shape[0], self.sample_count, self.action_dim)
        if commands.ndim != 2 or commands.shape[1] != self.action_dim:
            raise ValueError("commands must have shape [batch, action_dim]")
        if tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}")
        return (commands[:, None, :] + noise).clamp(
            -self.max_action, self.max_action
        )

    def checkpoint(self) -> Dict[str, object]:
        return {
            "action_dim": self.action_dim,
            "sample_count": self.sample_count,
            "beta": self.beta,
            "max_action": self.max_action,
            "seed": self.seed,
            "generator_state": self.generator.get_state(),
            "draw_calls": self.draw_calls,
            "half_vectors_drawn": self.half_vectors_drawn,
        }

    def restore(self, payload: Mapping[str, object]) -> None:
        for field in ("action_dim", "sample_count", "beta", "max_action", "seed"):
            if payload.get(field) != getattr(self, field):
                raise ValueError(f"channel checkpoint mismatch for {field}")
        state = payload.get("generator_state")
        if not isinstance(state, torch.Tensor):
            raise ValueError("channel checkpoint lacks generator_state")
        self.generator.set_state(state)
        self.draw_calls = int(payload.get("draw_calls", 0))
        self.half_vectors_drawn = int(payload.get("half_vectors_drawn", 0))

    def metadata(self) -> Dict[str, object]:
        return {
            "distribution": "iid_uniform_minus_beta_plus_beta_per_action_dimension",
            "scheme": "resampled_antithetic_per_state",
            "action_dim": self.action_dim,
            "sample_count": self.sample_count,
            "beta": self.beta,
            "max_action": self.max_action,
            "seed": self.seed,
            "draw_calls": self.draw_calls,
            "half_vectors_drawn": self.half_vectors_drawn,
            "dedicated_torch_generator": True,
        }


class ResidualCommandAdapter(nn.Module):
    """Zero-initialised bounded command residual."""

    def __init__(self, config: CCFQERPIConfig):
        super().__init__()
        self.config = config
        self.network = make_mlp(
            config.observation_dim,
            config.action_dim,
            config.adapter_hidden_dim,
            config.adapter_depth,
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("adapter MLP must end in a Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, normalized_observations: torch.Tensor) -> torch.Tensor:
        return self.config.delta_max * torch.tanh(
            self.network(normalized_observations)
        )


def _marginalized_twins(
    critic: TwinCritic,
    normalized_observations: torch.Tensor,
    physical_actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if normalized_observations.ndim != 2 or physical_actions.ndim != 3:
        raise ValueError("expected observations [B,D] and physical actions [B,K,A]")
    batch_size, sample_count, _ = physical_actions.shape
    if normalized_observations.shape[0] != batch_size:
        raise ValueError("observation/action batch sizes differ")
    expanded = normalized_observations[:, None, :].expand(-1, sample_count, -1)
    q1, q2 = critic.both(
        expanded.reshape(batch_size * sample_count, -1),
        physical_actions.reshape(batch_size * sample_count, -1),
    )
    return (
        q1.reshape(batch_size, sample_count).mean(dim=1),
        q2.reshape(batch_size, sample_count).mean(dim=1),
    )


def source_support_expansion(
    commands: torch.Tensor,
    logged_source_commands: torch.Tensor,
    *,
    source_beta: float,
    target_beta: float,
    max_action: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-state target-support expansion and violating-dimension rate.

    The known source channel conditional on a logged command has componentwise
    support ``[clip(u_log-beta_s), clip(u_log+beta_s)]``.  The penalty measures
    how far the candidate target-channel support extends beyond that interval.
    It cannot address state-distribution shift or finite conditional coverage.
    """

    if commands.shape != logged_source_commands.shape or commands.ndim != 2:
        raise ValueError("commands and logged_source_commands must align")
    source_low = (logged_source_commands - source_beta).clamp(
        -max_action, max_action
    )
    source_high = (logged_source_commands + source_beta).clamp(
        -max_action, max_action
    )
    target_low = (commands - target_beta).clamp(-max_action, max_action)
    target_high = (commands + target_beta).clamp(-max_action, max_action)
    lower_excess = (source_low - target_low).clamp_min(0.0)
    upper_excess = (target_high - source_high).clamp_min(0.0)
    width = 2.0 * max_action
    penalty = (
        lower_excess.square() + upper_excess.square()
    ).mean(dim=1) / (width * width)
    violation = ((lower_excess > 0.0) | (upper_excess > 0.0)).float().mean(dim=1)
    return penalty, violation


class CCFQERPIAgent:
    """Two-stage learner with exact stage and channel-RNG checkpointing."""

    def __init__(
        self,
        config: CCFQERPIConfig,
        device: torch.device,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        base_actor_state: Mapping[str, torch.Tensor],
        critic_state: Mapping[str, torch.Tensor],
    ):
        self.config = config
        self.device = canonical_torch_device(device)
        mean = np.asarray(observation_mean, dtype=np.float32).reshape(1, -1)
        std = np.asarray(observation_std, dtype=np.float32).reshape(1, -1)
        expected = (1, config.observation_dim)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(f"normalizer must have shape {expected}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("normalizer must be finite with positive std")
        self.observation_mean = torch.as_tensor(mean, device=self.device)
        self.observation_std = torch.as_tensor(std, device=self.device)

        self.base_actor = DeterministicPolicy(
            config.observation_dim,
            config.action_dim,
            config.base_hidden_dim,
            config.base_depth,
            config.max_action,
        ).to(self.device)
        self.base_actor.load_state_dict(base_actor_state, strict=True)
        self.base_actor.requires_grad_(False).eval()

        self.critic = TwinCritic(
            config.observation_dim,
            config.action_dim,
            config.base_hidden_dim,
            config.base_depth,
        ).to(self.device)
        self.critic.load_state_dict(critic_state, strict=True)
        # Reset the target lag at the changed fitted-Q objective.
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).to(
            self.device
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

        self.adapter = ResidualCommandAdapter(config).to(self.device)
        self.adapter_optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=config.adapter_learning_rate
        )
        self.continuation_channel = AntitheticUniformChannel(
            action_dim=config.action_dim,
            sample_count=config.train_channel_samples,
            beta=config.continuation_beta,
            max_action=config.max_action,
            seed=config.continuation_noise_seed,
            device=self.device,
        )
        self.actor_channel = AntitheticUniformChannel(
            action_dim=config.action_dim,
            sample_count=config.train_channel_samples,
            beta=config.target_beta,
            max_action=config.max_action,
            seed=config.actor_noise_seed,
            device=self.device,
        )
        self.fqe_steps = 0
        self.adapter_steps = 0
        self.value_scale: Optional[float] = None

    @property
    def stage(self) -> str:
        if self.fqe_steps < self.config.fqe_updates:
            return "fqe"
        if self.value_scale is None:
            return "value_scale_calibration"
        if self.adapter_steps < self.config.adapter_updates:
            return "adapter"
        return "complete"

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.observation_mean) / self.observation_std

    @torch.no_grad()
    def inverse_baseline_commands(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        normalized = self.normalize(observations)
        desired = self.base_actor(normalized)
        desired_np = desired.detach().cpu().numpy()
        low = np.full((self.config.action_dim,), -self.config.max_action)
        high = np.full((self.config.action_dim,), self.config.max_action)
        commands, audit = inverse_uniform_clipped_mean(
            desired_np, low, high, self.config.target_beta
        )
        return (
            torch.as_tensor(commands, dtype=observations.dtype, device=self.device),
            audit,
        )

    def compose_command(
        self,
        normalized_observations: torch.Tensor,
        inverse_commands: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposed_residual = self.adapter(normalized_observations)
        command = (inverse_commands + proposed_residual).clamp(
            -self.config.max_action, self.config.max_action
        )
        return command, proposed_residual, command - inverse_commands

    @torch.no_grad()
    def compute_fqe_target(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        required = ("next_observations", "next_inverse_commands", "rewards", "terminals")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"FQE batch lacks fields: {missing}")
        next_observations = self.normalize(batch["next_observations"])
        commands = batch["next_inverse_commands"]
        if noise is None:
            noise = self.continuation_channel.sample(
                commands.shape[0], dtype=commands.dtype, device=commands.device
            )
        physical_actions = self.continuation_channel.apply(commands, noise)
        q1_mean, q2_mean = _marginalized_twins(
            self.critic_target, next_observations, physical_actions
        )
        continuation = torch.minimum(q1_mean, q2_mean)
        target = batch["rewards"] + self.config.discount * (
            1.0 - batch["terminals"]
        ) * continuation
        return target, {
            "target_q1_mean": q1_mean,
            "target_q2_mean": q2_mean,
            "target_twin_gap": (q1_mean - q2_mean).abs(),
            "target_continuation": continuation,
        }

    def fqe_update(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        if self.stage != "fqe":
            raise RuntimeError(f"FQE update requested in stage {self.stage}")
        if "observations" not in batch or "executed_actions" not in batch:
            raise KeyError("FQE regression requires observations and executed_actions")
        target, diagnostics = self.compute_fqe_target(batch)
        observations = self.normalize(batch["observations"])
        q1, q2 = self.critic.both(observations, batch["executed_actions"])
        loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self._soft_update_target()
        self.fqe_steps += 1
        return {
            "fqe_loss": float(loss.detach()),
            "current_q1_mean": float(q1.detach().mean()),
            "current_q2_mean": float(q2.detach().mean()),
            "target_mean": float(target.detach().mean()),
            "target_twin_gap_mean": float(
                diagnostics["target_twin_gap"].detach().mean()
            ),
            "executed_action_regression": 1.0,
            "hubl_mc_mixing": 0.0,
            "td3_target_smoothing": 0.0,
        }

    def _soft_update_target(self) -> None:
        with torch.no_grad():
            for target, source in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target.mul_(1.0 - self.config.tau)
                target.add_(source, alpha=self.config.tau)

    def enter_adapter_stage(self, value_scale: float) -> None:
        if self.fqe_steps != self.config.fqe_updates:
            raise RuntimeError("adapter stage requires all fixed FQE updates")
        if self.value_scale is not None:
            raise RuntimeError("value scale was already fixed")
        if value_scale <= 0.0 or not np.isfinite(value_scale):
            raise ValueError("value_scale must be finite and positive")
        self.value_scale = max(float(value_scale), self.config.q_scale_epsilon)
        self.critic.requires_grad_(False).eval()
        self.critic_target.requires_grad_(False).eval()

    def adapter_objective(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.value_scale is None:
            raise RuntimeError("adapter objective requires a frozen value scale")
        required = ("observations", "inverse_commands", "logged_commands")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"adapter batch lacks fields: {missing}")
        normalized = self.normalize(batch["observations"])
        inverse_commands = batch["inverse_commands"]
        command, proposed_residual, applied_residual = self.compose_command(
            normalized, inverse_commands
        )
        if noise is None:
            noise = self.actor_channel.sample(
                command.shape[0], dtype=command.dtype, device=command.device
            )
        adapted_actions = self.actor_channel.apply(command, noise)
        baseline_actions = self.actor_channel.apply(inverse_commands, noise)
        adapted_q1, adapted_q2 = _marginalized_twins(
            self.critic, normalized, adapted_actions
        )
        adapted_value = torch.minimum(adapted_q1, adapted_q2)
        with torch.no_grad():
            baseline_q1, baseline_q2 = _marginalized_twins(
                self.critic, normalized, baseline_actions
            )
            baseline_value = torch.minimum(baseline_q1, baseline_q2)
        value_gain = adapted_value - baseline_value
        if self.config.delta_max == 0.0:
            residual_l2 = proposed_residual.square().mean(dim=1)
        else:
            residual_l2 = (
                proposed_residual / self.config.delta_max
            ).square().mean(dim=1)
        support_cost, support_violation = source_support_expansion(
            command,
            batch["logged_commands"],
            source_beta=self.config.source_beta,
            target_beta=self.config.target_beta,
            max_action=self.config.max_action,
        )
        loss = (
            -self.config.value_alpha * value_gain.mean() / self.value_scale
            + self.config.residual_penalty * residual_l2.mean()
            + self.config.support_penalty * support_cost.mean()
        )
        preclip = inverse_commands + proposed_residual
        diagnostics = {
            "adapted_value": adapted_value,
            "baseline_value": baseline_value,
            "value_gain": value_gain,
            "adapted_twin_gap": (adapted_q1 - adapted_q2).abs(),
            "baseline_twin_gap": (baseline_q1 - baseline_q2).abs(),
            "residual_l2": residual_l2,
            "proposed_residual_abs": proposed_residual.abs().mean(dim=1),
            "applied_residual_abs": applied_residual.abs().mean(dim=1),
            "support_cost": support_cost,
            "support_violation_fraction": support_violation,
            "command_saturation_fraction": (
                (preclip < -self.config.max_action)
                | (preclip > self.config.max_action)
            ).float().mean(dim=1),
        }
        return loss, diagnostics

    def adapter_update(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        if self.stage != "adapter":
            raise RuntimeError(f"adapter update requested in stage {self.stage}")
        loss, diagnostics = self.adapter_objective(batch)
        self.adapter_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.adapter_optimizer.step()
        self.adapter_steps += 1
        result = {"adapter_loss": float(loss.detach())}
        result.update(
            {
                f"{key}_mean": float(value.detach().mean())
                for key, value in diagnostics.items()
            }
        )
        return result

    @torch.no_grad()
    def command(
        self, observation: np.ndarray, *, use_residual: bool = True
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        inverse, inverse_audit = self.inverse_baseline_commands(tensor)
        normalized = self.normalize(tensor)
        if use_residual:
            command, proposed, applied = self.compose_command(normalized, inverse)
        else:
            command = inverse
            proposed = torch.zeros_like(inverse)
            applied = torch.zeros_like(inverse)
        return command.squeeze(0).cpu().numpy(), {
            "use_residual": bool(use_residual),
            "inverse_baseline_command": inverse.squeeze(0).cpu().numpy().tolist(),
            "proposed_residual": proposed.squeeze(0).cpu().numpy().tolist(),
            "applied_residual": applied.squeeze(0).cpu().numpy().tolist(),
            "inverse_saturation_count": int(
                inverse_audit["command_transform_saturation_count"]
            ),
        }

    def checkpoint(self) -> Dict[str, object]:
        return {
            "format": "cc_fqe_rpi_agent_v1",
            "config": asdict(self.config),
            "observation_mean": self.observation_mean.detach().cpu(),
            "observation_std": self.observation_std.detach().cpu(),
            "fqe_steps": self.fqe_steps,
            "adapter_steps": self.adapter_steps,
            "value_scale": self.value_scale,
            "base_actor": self.base_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "adapter": self.adapter.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "adapter_optimizer": self.adapter_optimizer.state_dict(),
            "continuation_channel": self.continuation_channel.checkpoint(),
            "actor_channel": self.actor_channel.checkpoint(),
            "stage": self.stage,
        }

    def restore(self, payload: Mapping[str, object]) -> None:
        if payload.get("format") != "cc_fqe_rpi_agent_v1":
            raise ValueError("unsupported CC-FQE/RPI agent checkpoint")
        if payload.get("config") != asdict(self.config):
            raise ValueError("agent checkpoint configuration mismatch")
        self.base_actor.load_state_dict(payload["base_actor"], strict=True)
        self.critic.load_state_dict(payload["critic"], strict=True)
        self.critic_target.load_state_dict(payload["critic_target"], strict=True)
        self.adapter.load_state_dict(payload["adapter"], strict=True)
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.adapter_optimizer.load_state_dict(payload["adapter_optimizer"])
        self.continuation_channel.restore(payload["continuation_channel"])
        self.actor_channel.restore(payload["actor_channel"])
        self.fqe_steps = int(payload["fqe_steps"])
        self.adapter_steps = int(payload["adapter_steps"])
        saved_scale = payload.get("value_scale")
        self.value_scale = None if saved_scale is None else float(saved_scale)
        if not 0 <= self.fqe_steps <= self.config.fqe_updates:
            raise ValueError("invalid restored FQE step count")
        if not 0 <= self.adapter_steps <= self.config.adapter_updates:
            raise ValueError("invalid restored adapter step count")
        if self.adapter_steps and self.value_scale is None:
            raise ValueError("adapter progress without a frozen value scale")
        if self.value_scale is not None:
            self.critic.requires_grad_(False).eval()
            self.critic_target.requires_grad_(False).eval()

    def metadata(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "fqe_steps": self.fqe_steps,
            "adapter_steps": self.adapter_steps,
            "value_scale": self.value_scale,
            "continuation_mode": self.config.continuation_mode,
            "continuation_beta": self.config.continuation_beta,
            "target_beta": self.config.target_beta,
            "source_beta": self.config.source_beta,
            "critic_current_action": "logged_executed_action",
            "hubl_mc_mixing": False,
            "td3_target_policy_smoothing": False,
            "twin_reduction": "min_of_channel_expectations",
            "continuation_channel": self.continuation_channel.metadata(),
            "actor_channel": self.actor_channel.metadata(),
        }


@torch.no_grad()
def audit_policy_k64(
    agent: CCFQERPIAgent,
    observations: torch.Tensor,
    inverse_commands: torch.Tensor,
    logged_commands: torch.Tensor,
    indices: Sequence[int],
    *,
    batch_size: int = 128,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Run an independent K=64-style audit without advancing training RNGs."""

    if batch_size <= 0:
        raise ValueError("audit batch_size must be positive")
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    if index_array.size == 0:
        raise ValueError("audit indices must be nonempty")
    if np.any(index_array < 0) or np.any(index_array >= observations.shape[0]):
        raise IndexError("audit index out of range")
    channel = AntitheticUniformChannel(
        action_dim=agent.config.action_dim,
        sample_count=agent.config.audit_channel_samples,
        beta=agent.config.target_beta,
        max_action=agent.config.max_action,
        seed=agent.config.audit_noise_seed,
        device=agent.device,
    )
    pieces: Dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "indices",
            "adapted_value",
            "baseline_value",
            "value_gain",
            "adapted_twin_gap",
            "baseline_twin_gap",
            "proposed_residual_abs",
            "applied_residual_abs",
            "support_cost",
            "support_violation_fraction",
            "command_saturation_fraction",
        )
    }
    for start in range(0, index_array.size, batch_size):
        selected_np = index_array[start : start + batch_size]
        selected = torch.as_tensor(selected_np, dtype=torch.long)
        obs = observations[selected].to(agent.device)
        inverse = inverse_commands[selected].to(agent.device)
        logged = logged_commands[selected].to(agent.device)
        normalized = agent.normalize(obs)
        command, proposed, applied = agent.compose_command(normalized, inverse)
        noise = channel.sample(
            command.shape[0], dtype=command.dtype, device=agent.device
        )
        adapted_actions = channel.apply(command, noise)
        baseline_actions = channel.apply(inverse, noise)
        aq1, aq2 = _marginalized_twins(agent.critic, normalized, adapted_actions)
        bq1, bq2 = _marginalized_twins(agent.critic, normalized, baseline_actions)
        adapted = torch.minimum(aq1, aq2)
        baseline = torch.minimum(bq1, bq2)
        support_cost, violation = source_support_expansion(
            command,
            logged,
            source_beta=agent.config.source_beta,
            target_beta=agent.config.target_beta,
            max_action=agent.config.max_action,
        )
        preclip = inverse + proposed
        tensors = {
            "adapted_value": adapted,
            "baseline_value": baseline,
            "value_gain": adapted - baseline,
            "adapted_twin_gap": (aq1 - aq2).abs(),
            "baseline_twin_gap": (bq1 - bq2).abs(),
            "proposed_residual_abs": proposed.abs().mean(dim=1),
            "applied_residual_abs": applied.abs().mean(dim=1),
            "support_cost": support_cost,
            "support_violation_fraction": violation,
            "command_saturation_fraction": (
                (preclip < -agent.config.max_action)
                | (preclip > agent.config.max_action)
            ).float().mean(dim=1),
        }
        pieces["indices"].append(selected_np.copy())
        for key, value in tensors.items():
            pieces[key].append(value.detach().cpu().numpy().astype(np.float32))
    arrays = {
        key: np.concatenate(value, axis=0) for key, value in pieces.items()
    }
    return arrays, {
        "audit_semantics": "independent_target_channel_common_random_numbers",
        "sample_count": agent.config.audit_channel_samples,
        "observation_count": int(index_array.size),
        "channel": channel.metadata(),
        "training_channel_rng_advanced": False,
    }
