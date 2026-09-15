"""Value-aware residual commands on top of an exact actuator-mean inverse.

The frozen base actor denotes a desired physical action.  For a known channel
``a_exec = clip(u + Uniform[-beta,beta], -max_action, max_action)``, its exact
mean inverse gives a baseline command ``u0``.  A small adapter changes only that
command,

    delta(s) = delta_max * tanh(f_psi(s))
    u(s) = clip(u0(s) + delta(s), -max_action, max_action).

The physical Q1 critic remains frozen.  The main estimator uses paired
antithetic channel samples for both ``u`` and ``u0`` and minimizes

    -alpha * mean(Qbar(u) - Qbar(u0)) / frozen_Q0_scale
    + penalty * mean((delta / delta_max)^2).

The paired baseline fixes the value scale and lowers comparison variance.  It
does not change the adapter gradient.  A mechanism ablation instead evaluates
``Q1`` at the exact clipped-channel conditional mean, repeating that
deterministic action K times.  Both estimators therefore cost exactly ``2*K``
Q1 input rows per state: K for the adapted command and K for the declared
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

from evaluation_controls import inverse_uniform_clipped_mean
from td3bc_core import TD3BCAgent, TwinCritic, make_mlp


BASELINE_TRANSFORMS = ("inverse", "identity")
VALUE_ESTIMATORS = ("sampled_expected_q1", "q1_at_channel_mean")


@dataclass(frozen=True)
class InverseResidualConfig:
    observation_dim: int
    action_dim: int
    hidden_dim: int = 128
    depth: int = 2
    max_action: float = 1.0
    execution_noise_beta: float = 1.0
    execution_noise_samples: int = 2
    execution_noise_seed: int = 271828
    delta_max: float = 0.25
    alpha: float = 1.0
    residual_penalty: float = 1.0
    q_scale_epsilon: float = 1e-6
    baseline_transform: str = "inverse"
    value_estimator: str = "sampled_expected_q1"

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.hidden_dim <= 0 or self.depth <= 0:
            raise ValueError("hidden_dim and depth must be positive")
        if self.max_action <= 0.0 or not np.isfinite(self.max_action):
            raise ValueError("max_action must be finite and positive")
        if self.execution_noise_beta < 0.0 or not np.isfinite(
            self.execution_noise_beta
        ):
            raise ValueError("execution_noise_beta must be finite and non-negative")
        if self.execution_noise_samples <= 0:
            raise ValueError("execution_noise_samples must be positive")
        if (
            self.execution_noise_beta > 0.0
            and self.execution_noise_samples % 2 != 0
        ):
            raise ValueError("positive-beta antithetic sampling requires K=2m")
        if self.execution_noise_seed < 0:
            raise ValueError("execution_noise_seed must be non-negative")
        if self.delta_max < 0.0 or not np.isfinite(self.delta_max):
            raise ValueError("delta_max must be finite and non-negative")
        if self.alpha < 0.0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and non-negative")
        if self.residual_penalty < 0.0 or not np.isfinite(
            self.residual_penalty
        ):
            raise ValueError("residual_penalty must be finite and non-negative")
        if self.q_scale_epsilon <= 0.0:
            raise ValueError("q_scale_epsilon must be positive")
        if self.baseline_transform not in BASELINE_TRANSFORMS:
            raise ValueError(
                f"baseline_transform must be one of {BASELINE_TRANSFORMS}"
            )
        if self.value_estimator not in VALUE_ESTIMATORS:
            raise ValueError(f"value_estimator must be one of {VALUE_ESTIMATORS}")


class InverseResidualAdapter(nn.Module):
    """Small zero-initialized state-conditioned residual command network."""

    def __init__(self, config: InverseResidualConfig):
        super().__init__()
        self.config = config
        self.network = make_mlp(
            config.observation_dim,
            config.action_dim,
            config.hidden_dim,
            config.depth,
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError("adapter MLP must end in a Linear layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def residual(self, normalized_observations: torch.Tensor) -> torch.Tensor:
        if normalized_observations.ndim != 2:
            raise ValueError("normalized_observations must have rank two")
        return self.config.delta_max * torch.tanh(
            self.network(normalized_observations)
        )

    def compose_command(
        self,
        normalized_observations: torch.Tensor,
        baseline_commands: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if baseline_commands.ndim != 2 or (
            baseline_commands.shape[-1] != self.config.action_dim
        ):
            raise ValueError(
                "baseline_commands must have shape [batch, action_dim]"
            )
        if normalized_observations.shape[0] != baseline_commands.shape[0]:
            raise ValueError("observation and command batch sizes must match")
        proposed_residual = self.residual(normalized_observations)
        command = (baseline_commands + proposed_residual).clamp(
            -self.config.max_action, self.config.max_action
        )
        applied_residual = command - baseline_commands
        return command, proposed_residual, applied_residual


class ResampledAntitheticChannel:
    """Per-state execution-noise sampler isolated from all training RNGs."""

    def __init__(self, config: InverseResidualConfig, device: torch.device):
        self.config = config
        # Canonicalize ``cuda`` to the concrete device used by tensors
        # (normally ``cuda:0``).  Comparing torch.device("cuda") directly with
        # tensor.device spuriously reports a mismatch even though both address
        # the same accelerator.
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and requested_device.index is None:
            requested_device = torch.device(
                "cuda", torch.cuda.current_device()
            )
        self.device = requested_device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.execution_noise_seed)
        self.draw_calls = 0
        self.half_vectors_drawn = 0

    def sample(
        self, batch_size: int, *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device != self.device:
            raise ValueError("sampling device differs from channel generator device")
        self.draw_calls += 1
        if self.config.execution_noise_beta == 0.0:
            return torch.zeros(
                (
                    batch_size,
                    self.config.execution_noise_samples,
                    self.config.action_dim,
                ),
                dtype=dtype,
                device=device,
            )
        half_count = self.config.execution_noise_samples // 2
        half = torch.rand(
            (batch_size, half_count, self.config.action_dim),
            generator=self.generator,
            dtype=dtype,
            device=device,
        )
        half = (2.0 * half - 1.0) * self.config.execution_noise_beta
        self.half_vectors_drawn += batch_size * half_count
        return torch.cat((half, -half), dim=1)

    def apply(self, commands: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        expected = (
            commands.shape[0],
            self.config.execution_noise_samples,
            self.config.action_dim,
        )
        if commands.ndim != 2 or commands.shape[1] != self.config.action_dim:
            raise ValueError("commands must have shape [batch, action_dim]")
        if tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}")
        return (commands[:, None, :] + noise).clamp(
            -self.config.max_action, self.config.max_action
        )

    def checkpoint(self) -> Dict[str, object]:
        return {
            "generator_state": self.generator.get_state().cpu(),
            "draw_calls": int(self.draw_calls),
            "half_vectors_drawn": int(self.half_vectors_drawn),
        }

    def restore(self, payload: Mapping[str, object]) -> None:
        state = payload.get("generator_state")
        if not isinstance(state, torch.Tensor):
            raise ValueError("channel checkpoint lacks generator_state")
        self.generator.set_state(state.detach().cpu())
        self.draw_calls = int(payload.get("draw_calls", 0))
        self.half_vectors_drawn = int(payload.get("half_vectors_drawn", 0))

    def metadata(self) -> Dict[str, object]:
        return {
            "distribution": "iid_uniform_minus_beta_plus_beta_per_action_dimension",
            "beta": float(self.config.execution_noise_beta),
            "sample_count": int(self.config.execution_noise_samples),
            "scheme": "resampled_antithetic_per_state",
            "seed": int(self.config.execution_noise_seed),
            "draw_calls": int(self.draw_calls),
            "half_vectors_drawn": int(self.half_vectors_drawn),
            "global_torch_rng_consumed": False,
            "checkpoint_contains_generator_state": True,
            "exact_training_resume_implemented": True,
        }


def expected_clipped_uniform_action_torch(
    command: torch.Tensor,
    low: float | torch.Tensor,
    high: float | torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Differentiably compute ``E[clip(command + U[-beta,beta], low, high)]``.

    This is the torch counterpart of
    :func:`evaluation_controls.expected_clipped_uniform_action`. Integrating
    ``clip(x,l,h) = x + (l-x)_+ - (x-h)_+`` gives the squared-hinge expression
    below. This is an exact channel mean rather than a Monte Carlo estimate,
    and gradients propagate through the command.
    """

    if not torch.is_tensor(command) or not torch.is_floating_point(command):
        raise TypeError("command must be a floating-point torch tensor")
    noise_beta = float(beta)
    if not np.isfinite(noise_beta) or noise_beta < 0.0:
        raise ValueError("uniform actuator-noise beta must be finite and non-negative")
    low_tensor = torch.as_tensor(low, dtype=command.dtype, device=command.device)
    high_tensor = torch.as_tensor(high, dtype=command.dtype, device=command.device)
    try:
        _, broadcast_low, broadcast_high = torch.broadcast_tensors(
            command, low_tensor, high_tensor
        )
    except RuntimeError as exc:
        raise ValueError("action bounds are not broadcastable to command shape") from exc
    if bool(torch.any(broadcast_low > broadcast_high)):
        raise ValueError("action-space lower bounds exceed upper bounds")
    if noise_beta == 0.0:
        return torch.maximum(torch.minimum(command, broadcast_high), broadcast_low)

    beta_tensor = command.new_tensor(noise_beta)
    upper = command + beta_tensor
    lower = command - beta_tensor
    lower_hinge_integral = (
        torch.relu(broadcast_low - lower).square()
        - torch.relu(broadcast_low - upper).square()
    ) / (4.0 * beta_tensor)
    upper_hinge_integral = (
        torch.relu(upper - broadcast_high).square()
        - torch.relu(lower - broadcast_high).square()
    ) / (4.0 * beta_tensor)
    return command + lower_hinge_integral - upper_hinge_integral


def value_estimator_action_rows(
    commands: torch.Tensor,
    channel: ResampledAntitheticChannel,
    *,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct exactly K physical-action rows for the configured estimator.

    The sampled estimator can return newly sampled noise so a second command
    receives common random numbers. The channel-mean ablation rejects supplied
    noise and repeats its deterministic exact mean K times; consequently it
    consumes no channel RNG while matching the sampled estimator's Q1 row cost.
    """

    config = channel.config
    if config.value_estimator == "sampled_expected_q1":
        if noise is None:
            noise = channel.sample(
                commands.shape[0], dtype=commands.dtype, device=commands.device
            )
        return channel.apply(commands, noise), noise
    if config.value_estimator != "q1_at_channel_mean":
        raise AssertionError("InverseResidualConfig failed estimator validation")
    if noise is not None:
        raise ValueError("q1_at_channel_mean does not accept execution-noise samples")
    channel_mean = expected_clipped_uniform_action_torch(
        commands,
        -config.max_action,
        config.max_action,
        config.execution_noise_beta,
    )
    repeated = channel_mean[:, None, :].repeat(
        1, config.execution_noise_samples, 1
    )
    return repeated, None


def value_estimator_metadata(config: InverseResidualConfig) -> Dict[str, object]:
    """Return explicit provenance for the training-time value estimator."""

    sampled = config.value_estimator == "sampled_expected_q1"
    return {
        "name": config.value_estimator,
        "q1_rows_per_command_state": int(config.execution_noise_samples),
        "uses_fresh_execution_noise_samples": sampled,
        "uses_common_random_numbers_for_adapted_and_baseline": sampled,
        "channel_rng_consumed": sampled,
        "deterministic_exact_channel_mean": not sampled,
        "deterministic_channel_mean_repeated_for_equal_q1_rows": not sampled,
        "channel_mean_implementation": (
            None
            if sampled
            else "differentiable_exact_uniform_clip_squared_hinge_integral"
        ),
    }


def marginalized_q1_from_physical_actions(
    critic: TwinCritic,
    normalized_observations: torch.Tensor,
    physical_actions: torch.Tensor,
) -> torch.Tensor:
    if normalized_observations.ndim != 2 or physical_actions.ndim != 3:
        raise ValueError("expected observations [B,D] and physical actions [B,K,A]")
    batch_size, sample_count, _ = physical_actions.shape
    expanded_observations = normalized_observations[:, None, :].expand(
        -1, sample_count, -1
    )
    values = critic.q1_only(
        expanded_observations.reshape(batch_size * sample_count, -1),
        physical_actions.reshape(batch_size * sample_count, -1),
    )
    return values.reshape(batch_size, sample_count).mean(dim=1)


def marginalized_twin_from_physical_actions(
    critic: TwinCritic,
    normalized_observations: torch.Tensor,
    physical_actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return each frozen critic's channel-marginalized command value."""

    if normalized_observations.ndim != 2 or physical_actions.ndim != 3:
        raise ValueError("expected observations [B,D] and physical actions [B,K,A]")
    batch_size, sample_count, _ = physical_actions.shape
    expanded_observations = normalized_observations[:, None, :].expand(
        -1, sample_count, -1
    )
    q1, q2 = critic.both(
        expanded_observations.reshape(batch_size * sample_count, -1),
        physical_actions.reshape(batch_size * sample_count, -1),
    )
    return (
        q1.reshape(batch_size, sample_count).mean(dim=1),
        q2.reshape(batch_size, sample_count).mean(dim=1),
    )


def adapter_objective(
    adapter: InverseResidualAdapter,
    frozen_critic: TwinCritic,
    normalized_observations: torch.Tensor,
    baseline_commands: torch.Tensor,
    channel: ResampledAntitheticChannel,
    value_scale: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the paired value-improvement objective and diagnostic tensors."""

    command, proposed_residual, applied_residual = adapter.compose_command(
        normalized_observations, baseline_commands
    )
    adapted_actions, shared_noise = value_estimator_action_rows(command, channel)
    baseline_actions, _ = value_estimator_action_rows(
        baseline_commands, channel, noise=shared_noise
    )
    adapted_q = marginalized_q1_from_physical_actions(
        frozen_critic, normalized_observations, adapted_actions
    )
    with torch.no_grad():
        baseline_q = marginalized_q1_from_physical_actions(
            frozen_critic, normalized_observations, baseline_actions
        )
    q_denominator = torch.as_tensor(
        value_scale,
        dtype=adapted_q.dtype,
        device=adapted_q.device,
    ).detach()
    if q_denominator.numel() != 1 or float(q_denominator) <= 0.0:
        raise ValueError("value_scale must be a positive scalar")
    normalized_value_gain = (adapted_q - baseline_q).mean() / q_denominator
    if adapter.config.delta_max == 0.0:
        normalized_residual_l2 = proposed_residual.square().mean()
    else:
        normalized_residual_l2 = (
            proposed_residual / adapter.config.delta_max
        ).square().mean()
    loss = (
        -adapter.config.alpha * normalized_value_gain
        + adapter.config.residual_penalty * normalized_residual_l2
    )
    preclip = baseline_commands + proposed_residual
    diagnostics = {
        "adapted_q_mean": adapted_q.mean(),
        "baseline_q_mean": baseline_q.mean(),
        "normalized_value_gain": normalized_value_gain,
        "q_scale_denominator": q_denominator,
        "normalized_residual_l2": normalized_residual_l2,
        "proposed_residual_abs_mean": proposed_residual.abs().mean(),
        "proposed_residual_abs_max": proposed_residual.abs().max(),
        "applied_residual_abs_mean": applied_residual.abs().mean(),
        "command_saturation_fraction": (
            (preclip < -adapter.config.max_action)
            | (preclip > adapter.config.max_action)
        ).float().mean(),
        "command_at_bound_fraction": (
            (command <= -adapter.config.max_action)
            | (command >= adapter.config.max_action)
        ).float().mean(),
    }
    return loss, diagnostics


def exact_inverse_baseline_commands(
    desired_physical_actions: torch.Tensor,
    config: InverseResidualConfig,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Apply the same NumPy analytic/bisection inverse used by evaluation."""

    if desired_physical_actions.ndim != 2 or (
        desired_physical_actions.shape[1] != config.action_dim
    ):
        raise ValueError("desired actions must have shape [batch, action_dim]")
    desired = desired_physical_actions.detach().cpu().numpy()
    low = np.full((config.action_dim,), -config.max_action, dtype=desired.dtype)
    high = np.full((config.action_dim,), config.max_action, dtype=desired.dtype)
    commands, audit = inverse_uniform_clipped_mean(
        desired, low, high, config.execution_noise_beta
    )
    return (
        torch.as_tensor(
            commands,
            dtype=desired_physical_actions.dtype,
            device=desired_physical_actions.device,
        ),
        audit,
    )


def baseline_commands_from_desired(
    desired_physical_actions: torch.Tensor,
    config: InverseResidualConfig,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Apply the declared inverse or equal-capacity identity control."""

    if config.baseline_transform == "inverse":
        return exact_inverse_baseline_commands(desired_physical_actions, config)
    if config.baseline_transform != "identity":
        raise AssertionError("InverseResidualConfig failed transform validation")
    commands = desired_physical_actions.detach().clamp(
        -config.max_action, config.max_action
    )
    return commands, {
        "command_transform_value_count": int(commands.numel()),
        "command_transform_saturation_count": 0,
        "transformed_command_at_bound_count": int(
            (
                (commands <= -config.max_action)
                | (commands >= config.max_action)
            ).sum()
        ),
    }


class InverseResidualController:
    """Inference wrapper combining a frozen base policy and residual adapter."""

    def __init__(
        self,
        base_agent: TD3BCAgent,
        adapter: InverseResidualAdapter,
    ):
        self.base_agent = base_agent
        self.adapter = adapter

    @torch.no_grad()
    def command(
        self,
        observation: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        use_residual: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        desired = self.base_agent.act(observation, action_low, action_high)
        if self.adapter.config.baseline_transform == "inverse":
            baseline, inverse_audit = inverse_uniform_clipped_mean(
                desired,
                action_low,
                action_high,
                self.adapter.config.execution_noise_beta,
            )
        else:
            baseline = np.clip(desired, action_low, action_high)
            inverse_audit = {
                "command_transform_value_count": int(np.asarray(desired).size),
                "command_transform_saturation_count": 0,
                "transformed_command_at_bound_count": int(
                    ((baseline <= action_low) | (baseline >= action_high)).sum()
                ),
            }
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.base_agent.device,
        ).reshape(1, -1)
        normalized = self.base_agent.normalize_observations(observation_tensor)
        baseline_tensor = torch.as_tensor(
            baseline,
            dtype=torch.float32,
            device=self.base_agent.device,
        ).reshape(1, -1)
        if use_residual:
            command, proposed, applied = self.adapter.compose_command(
                normalized, baseline_tensor
            )
        else:
            command = baseline_tensor
            proposed = torch.zeros_like(command)
            applied = torch.zeros_like(command)
        command_array = command.squeeze(0).cpu().numpy()
        details: Dict[str, object] = {
            "desired_physical_action": desired.tolist(),
            "baseline_transform": self.adapter.config.baseline_transform,
            "baseline_command": np.asarray(baseline).tolist(),
            "proposed_residual": proposed.squeeze(0).cpu().numpy().tolist(),
            "applied_residual": applied.squeeze(0).cpu().numpy().tolist(),
            "inverse_saturation_count": int(
                inverse_audit["command_transform_saturation_count"]
            ),
            "command_at_bound_count": int(
                ((command_array <= action_low) | (command_array >= action_high)).sum()
            ),
        }
        return command_array, details


def module_state_sha256(modules: Mapping[str, nn.Module]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes for mutation auditing."""

    digest = hashlib.sha256()
    for module_name in sorted(modules):
        state = modules[module_name].state_dict()
        for tensor_name in sorted(state):
            tensor = state[tensor_name].detach().cpu().contiguous()
            digest.update(module_name.encode("utf-8"))
            digest.update(tensor_name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
