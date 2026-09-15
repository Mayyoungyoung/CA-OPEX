"""Evaluate a channel-aware extension of OPEX with a frozen physical critic.

The original OPEX rule is the single test-time update
``a <- a + step_size * grad_a Q(s, a)`` from Park et al., *Is Value
Learning Really the Main Bottleneck in Offline RL?* (NeurIPS 2024):
https://proceedings.neurips.cc/paper_files/paper/2024/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html

This evaluator can use either the base actor action itself (``identity``) or a
clipped-channel mean inverse as the pre-gradient anchor.  The special setting
identity/beta=0/K=1/T=1/delta>=the action-space diameter preserves the original
single-step OPEX structure.  Settings that model a nonzero action channel,
start from its inverse, use antithetic E[Q1], or take multiple steps are
explicitly stronger channel-aware extensions rather than unchanged OPEX.
Every iterate is projected into the fixed intersection of the action bounds
and an L-infinity ball around the initial anchor.  The actor and critic are
never updated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluation_controls import inverse_uniform_clipped_mean
from train_inverse_residual_adapter import (
    load_frozen_physical_agent,
    require_new_output_file,
    resolve_channel_calibration,
)


OPEX_REFERENCE_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
    "8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html"
)
CHANNEL_OPEX_DELTA_MAX = 0.25
MAX_TORCH_SEED = (1 << 63) - 1
BASELINE_TRANSFORMS = ("inverse", "identity")


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json_new(path: Path, payload: object) -> None:
    """Atomically publish JSON without ever replacing an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # A hard-link publication is atomic and fails if another process has
        # created the destination since require_new_output_file ran.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def canonical_device(device: str | torch.device) -> torch.device:
    """Resolve an implicit CUDA device to the concrete tensor device."""

    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
    return requested


def _validate_seed(seed: int, name: str) -> int:
    value = int(seed)
    if value < 0 or value > MAX_TORCH_SEED:
        raise ValueError(f"{name} must lie in [0, {MAX_TORCH_SEED}]")
    return value


@dataclass(frozen=True)
class ChannelOPEXConfig:
    action_dim: int
    step_size: float
    channel_beta: float
    gradient_noise_samples: int = 8
    gradient_steps: int = 1
    delta_max: float = CHANNEL_OPEX_DELTA_MAX
    baseline_transform: str = "inverse"

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not np.isfinite(self.step_size) or self.step_size < 0.0:
            raise ValueError("step_size must be finite and non-negative")
        if not np.isfinite(self.channel_beta) or self.channel_beta < 0.0:
            raise ValueError("channel_beta must be finite and non-negative")
        if self.gradient_noise_samples <= 0:
            raise ValueError("gradient_noise_samples must be positive")
        if self.channel_beta > 0.0 and self.gradient_noise_samples % 2 != 0:
            raise ValueError("positive-beta antithetic sampling requires even K=2m")
        if not np.isfinite(self.delta_max) or self.delta_max < 0.0:
            raise ValueError("delta_max must be finite and non-negative")
        if self.gradient_steps <= 0:
            raise ValueError("gradient_steps must be positive")
        if self.baseline_transform not in BASELINE_TRANSFORMS:
            raise ValueError(
                f"baseline_transform must be one of {BASELINE_TRANSFORMS}"
            )


def make_episode_gradient_generator(
    device: str | torch.device, seed: int
) -> torch.Generator:
    resolved = canonical_device(device)
    generator = torch.Generator(device=resolved)
    generator.manual_seed(_validate_seed(seed, "gradient-noise seed"))
    return generator


def sample_antithetic_channel_noise(
    batch_size: int,
    config: ChannelOPEXConfig,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Draw independent-per-state antithetic target-channel perturbations."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved = canonical_device(device)
    generator_device = canonical_device(generator.device)
    if generator_device != resolved:
        raise ValueError("gradient-noise generator and action tensor devices differ")
    shape = (batch_size, config.gradient_noise_samples, config.action_dim)
    if config.channel_beta == 0.0:
        return torch.zeros(shape, dtype=dtype, device=resolved)
    half_count = config.gradient_noise_samples // 2
    half = torch.rand(
        (batch_size, half_count, config.action_dim),
        generator=generator,
        dtype=dtype,
        device=resolved,
    )
    half = (2.0 * half - 1.0) * config.channel_beta
    return torch.cat((half, -half), dim=1)


def channel_opex_step(
    critic: object,
    normalized_observation: torch.Tensor,
    baseline_command: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    config: ChannelOPEXConfig,
    gradient_generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run projected CA-OPEX; T=1 is the original OPEX update structure."""

    if normalized_observation.ndim != 2 or normalized_observation.shape[0] != 1:
        raise ValueError("normalized_observation must have shape [1, observation_dim]")
    if tuple(baseline_command.shape) != (1, config.action_dim):
        raise ValueError("baseline_command must have shape [1, action_dim]")
    if baseline_command.device != normalized_observation.device:
        raise ValueError("observation and baseline command devices differ")
    low = torch.as_tensor(
        action_low,
        dtype=baseline_command.dtype,
        device=baseline_command.device,
    ).reshape(1, config.action_dim)
    high = torch.as_tensor(
        action_high,
        dtype=baseline_command.dtype,
        device=baseline_command.device,
    ).reshape(1, config.action_dim)
    if bool(torch.any(low > high)):
        raise ValueError("action-space lower bounds exceed upper bounds")
    if bool(torch.any(baseline_command < low)) or bool(
        torch.any(baseline_command > high)
    ):
        raise ValueError("pre-gradient baseline command lies outside action bounds")

    anchor = baseline_command.detach().clone()
    trust_low = torch.maximum(low, anchor - config.delta_max)
    trust_high = torch.minimum(high, anchor + config.delta_max)
    command = anchor
    expected_q1_values: List[float] = []
    q1_row_std_values: List[float] = []
    gradient_l2_values: List[float] = []
    gradient_abs_max_values: List[float] = []
    raw_step_abs_max_values: List[float] = []
    trust_region_clipped_value_count = 0
    for _ in range(config.gradient_steps):
        current = command.detach().clone().requires_grad_(True)
        noise = sample_antithetic_channel_noise(
            1,
            config,
            generator=gradient_generator,
            dtype=current.dtype,
            device=current.device,
        )
        if config.channel_beta == 0.0:
            # ``current`` is already inside the projected feasible set.  Feed
            # it directly to Q for the degenerate channel so T=1,K=1 exactly
            # has the OPEX gradient grad_u Q(s,u).  Applying clamp here would
            # inject PyTorch's 0.5 boundary derivative at u==low/high.
            physical_actions = current[:, None, :].expand(
                -1, config.gradient_noise_samples, -1
            )
        else:
            physical_actions = torch.maximum(
                torch.minimum(current[:, None, :] + noise, high[:, None, :]),
                low[:, None, :],
            )
        observation_rows = normalized_observation[:, None, :].expand(
            -1, config.gradient_noise_samples, -1
        )
        q1_rows = critic.q1_only(
            observation_rows.reshape(config.gradient_noise_samples, -1),
            physical_actions.reshape(config.gradient_noise_samples, config.action_dim),
        )
        if tuple(q1_rows.shape) != (config.gradient_noise_samples,):
            raise ValueError("critic.q1_only must return one scalar per Q row")
        expected_q1 = q1_rows.mean()
        gradient = torch.autograd.grad(
            expected_q1,
            current,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        if not bool(torch.isfinite(q1_rows).all()) or not bool(
            torch.isfinite(gradient).all()
        ):
            raise FloatingPointError("non-finite Q value or command gradient")
        raw_step = config.step_size * gradient
        unprojected = current + raw_step
        trust_region_clipped_value_count += int(
            ((unprojected.detach() < trust_low) | (unprojected.detach() > trust_high))
            .sum()
            .cpu()
        )
        command = torch.maximum(torch.minimum(unprojected, trust_high), trust_low)
        expected_q1_values.append(float(expected_q1.detach().cpu()))
        q1_row_std_values.append(
            float(q1_rows.detach().to(torch.float64).std(unbiased=False).cpu())
        )
        gradient_l2_values.append(float(gradient.detach().norm(p=2).cpu()))
        gradient_abs_max_values.append(float(gradient.detach().abs().max().cpu()))
        raw_step_abs_max_values.append(float(raw_step.detach().abs().max().cpu()))

    proposed_residual = command - anchor
    applied_residual = proposed_residual
    details: Dict[str, object] = {
        "q1_rows": int(config.gradient_noise_samples * config.gradient_steps),
        "q1_gradient_calls": int(config.gradient_steps),
        "q1_expected_before_update": float(np.mean(expected_q1_values)),
        "q1_expected_by_gradient_step": expected_q1_values,
        "q1_row_std_before_update": float(np.mean(q1_row_std_values)),
        "gradient_l2": float(np.mean(gradient_l2_values)),
        "gradient_abs_max": float(np.max(gradient_abs_max_values)),
        "raw_residual_abs_max": float(np.max(raw_step_abs_max_values)),
        "trust_region_clipped_value_count": trust_region_clipped_value_count,
        "proposed_residual": proposed_residual.detach().squeeze(0).cpu().tolist(),
        "applied_residual": applied_residual.detach().squeeze(0).cpu().tolist(),
        "command_at_bound_count": int(
            ((command.detach() <= low) | (command.detach() >= high)).sum().cpu()
        ),
    }
    return command.detach(), details


def command_from_observation(
    base_agent: object,
    observation: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    config: ChannelOPEXConfig,
    *,
    adapted: bool,
    gradient_generator: Optional[torch.Generator],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the declared anchor or its projected CA-OPEX extension."""

    desired = base_agent.act(observation, action_low, action_high)
    if config.baseline_transform == "inverse":
        baseline, inverse_audit = inverse_uniform_clipped_mean(
            desired,
            action_low,
            action_high,
            config.channel_beta,
        )
    elif config.baseline_transform == "identity":
        baseline = np.clip(desired, action_low, action_high)
        inverse_audit = {
            "command_transform_saturation_count": 0,
        }
    else:
        raise AssertionError("ChannelOPEXConfig failed baseline validation")
    baseline_array = np.asarray(baseline, dtype=np.float32)
    common: Dict[str, object] = {
        "desired_physical_action": np.asarray(desired).tolist(),
        "baseline_command": baseline_array.tolist(),
        "baseline_transform": config.baseline_transform,
        "inverse_saturation_count": int(
            inverse_audit["command_transform_saturation_count"]
        ),
    }
    if not adapted:
        return baseline_array, {
            **common,
            "q1_rows": 0,
            "q1_gradient_calls": 0,
            "proposed_residual": np.zeros_like(baseline_array).tolist(),
            "applied_residual": np.zeros_like(baseline_array).tolist(),
            "command_at_bound_count": int(
                ((baseline_array <= action_low) | (baseline_array >= action_high)).sum()
            ),
        }
    if gradient_generator is None:
        raise ValueError("adapted command requires an episode-local gradient generator")

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=base_agent.device,
    ).reshape(1, -1)
    normalized = base_agent.normalize_observations(observation_tensor)
    baseline_tensor = torch.as_tensor(
        baseline_array,
        dtype=torch.float32,
        device=base_agent.device,
    ).reshape(1, -1)
    command, opex_details = channel_opex_step(
        base_agent.critic,
        normalized,
        baseline_tensor,
        torch.as_tensor(action_low, dtype=torch.float32, device=base_agent.device),
        torch.as_tensor(action_high, dtype=torch.float32, device=base_agent.device),
        config,
        gradient_generator,
    )
    return command.squeeze(0).cpu().numpy(), {**common, **opex_details}


def _optional_mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _optional_max(values: Sequence[float]) -> Optional[float]:
    return float(np.max(values)) if values else None


def evaluate_arm(
    base_agent: object,
    config: ChannelOPEXConfig,
    env_name: str,
    environment_seeds: Sequence[int],
    action_noise_seeds: Sequence[int],
    gradient_noise_seeds: Sequence[int],
    rollout_beta: float,
    *,
    adapted: bool,
    reference_min: float,
    reference_max: float,
) -> Dict[str, object]:
    """Evaluate one arm with independent environment, actuator, and gradient RNGs."""

    import gymnasium as gym

    if not np.isfinite(rollout_beta) or rollout_beta < 0.0:
        raise ValueError("rollout_beta must be finite and non-negative")
    episode_count = len(environment_seeds)
    if episode_count == 0 or len(action_noise_seeds) != episode_count or (
        len(gradient_noise_seeds) != episode_count
    ):
        raise ValueError("environment, actuator, and gradient seed lists must align")
    for seed in (*environment_seeds, *action_noise_seeds, *gradient_noise_seeds):
        _validate_seed(int(seed), "evaluation seed")

    env = gym.make(env_name)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if low.shape != (config.action_dim,) or high.shape != (config.action_dim,):
        env.close()
        raise ValueError("environment action dimension differs from base checkpoint")
    if not np.allclose(low, -base_agent.config.max_action) or not np.allclose(
        high, base_agent.config.max_action
    ):
        env.close()
        raise ValueError("environment action bounds differ from base checkpoint")

    returns: List[float] = []
    lengths: List[int] = []
    q1_rows_by_episode: List[int] = []
    q1_values: List[float] = []
    gradient_l2_values: List[float] = []
    gradient_abs_max_values: List[float] = []
    raw_residual_abs_max_values: List[float] = []
    proposed_residual_abs_values: List[float] = []
    applied_residual_abs_values: List[float] = []
    trust_region_clipped_count = 0
    inverse_saturation_count = 0
    command_at_bound_count = 0
    actuator_clip_count = 0
    action_value_count = 0
    q1_gradient_call_count = 0
    squared_execution_deltas: List[float] = []
    started = time.perf_counter()
    try:
        for episode_index, (environment_seed, action_noise_seed) in enumerate(
            zip(environment_seeds, action_noise_seeds)
        ):
            observation, _ = env.reset(seed=int(environment_seed))
            actuator_rng = np.random.default_rng(int(action_noise_seed))
            gradient_generator = (
                make_episode_gradient_generator(
                    base_agent.device, int(gradient_noise_seeds[episode_index])
                )
                if adapted
                else None
            )
            episode_return = 0.0
            episode_q_rows = 0
            max_steps = int(env.spec.max_episode_steps or 1000)
            for length in range(1, max_steps + 1):
                command, details = command_from_observation(
                    base_agent,
                    np.asarray(observation, dtype=np.float32),
                    low,
                    high,
                    config,
                    adapted=adapted,
                    gradient_generator=gradient_generator,
                )
                inverse_saturation_count += int(details["inverse_saturation_count"])
                command_at_bound_count += int(details["command_at_bound_count"])
                q_rows = int(details["q1_rows"])
                episode_q_rows += q_rows
                if adapted:
                    q1_gradient_call_count += int(details["q1_gradient_calls"])
                    q1_values.append(float(details["q1_expected_before_update"]))
                    gradient_l2_values.append(float(details["gradient_l2"]))
                    gradient_abs_max_values.append(float(details["gradient_abs_max"]))
                    raw_residual_abs_max_values.append(
                        float(details["raw_residual_abs_max"])
                    )
                    trust_region_clipped_count += int(
                        details["trust_region_clipped_value_count"]
                    )
                proposed_residual_abs_values.extend(
                    np.abs(np.asarray(details["proposed_residual"])).reshape(-1).tolist()
                )
                applied_residual_abs_values.extend(
                    np.abs(np.asarray(details["applied_residual"])).reshape(-1).tolist()
                )

                perturbation = actuator_rng.uniform(
                    -rollout_beta,
                    rollout_beta,
                    size=command.shape,
                ).astype(np.float32)
                requested = command + perturbation
                executed = np.clip(requested, low, high).astype(np.float32, copy=False)
                actuator_clip_count += int(
                    ((requested < low) | (requested > high)).sum()
                )
                action_value_count += int(command.size)
                squared_execution_deltas.extend(
                    np.square(executed - command).reshape(-1).tolist()
                )
                observation, reward, terminated, truncated, _ = env.step(executed)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
            q1_rows_by_episode.append(episode_q_rows)
    finally:
        env.close()

    values = np.asarray(returns, dtype=np.float64)
    normalized = 100.0 * (values - reference_min) / (reference_max - reference_min)
    q1_rows_total = int(sum(q1_rows_by_episode))
    expected_rows = int(
        sum(lengths)
        * (config.gradient_noise_samples * config.gradient_steps if adapted else 0)
    )
    if q1_rows_total != expected_rows:
        raise RuntimeError("Q-row accounting disagrees with episode lengths")
    return {
        "arm": (
            "channel_opex"
            if adapted
            else f"{config.baseline_transform}_anchor_baseline"
        ),
        "environment_seeds": [int(seed) for seed in environment_seeds],
        "action_noise_seeds": [int(seed) for seed in action_noise_seeds],
        "gradient_noise_seeds": [int(seed) for seed in gradient_noise_seeds],
        "gradient_noise_stream_used": bool(adapted and config.channel_beta > 0.0),
        "gradient_noise_sampling_frequency": (
            "per_gradient_step"
            if adapted and config.channel_beta > 0.0
            else "none_beta_zero_or_baseline_arm"
        ),
        "gradient_noise_draw_calls": int(
            sum(lengths) * config.gradient_steps
            if adapted and config.channel_beta > 0.0
            else 0
        ),
        "action_noise_beta": float(rollout_beta),
        "returns": values.tolist(),
        "lengths": lengths,
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std()),
        "q1_rows_per_action": int(
            config.gradient_noise_samples * config.gradient_steps if adapted else 0
        ),
        "q1_rows_by_episode": q1_rows_by_episode,
        "q1_rows_total": q1_rows_total,
        "q1_gradient_calls": int(q1_gradient_call_count),
        "q1_expected_before_update_mean": _optional_mean(q1_values),
        "gradient_l2_mean": _optional_mean(gradient_l2_values),
        "gradient_abs_max": _optional_max(gradient_abs_max_values),
        "raw_residual_abs_max": _optional_max(raw_residual_abs_max_values),
        "trust_region_clipped_fraction": trust_region_clipped_count
        / max(action_value_count * config.gradient_steps, 1),
        "proposed_residual_abs_mean": _optional_mean(
            proposed_residual_abs_values
        ),
        "applied_residual_abs_mean": _optional_mean(applied_residual_abs_values),
        "applied_residual_abs_max": _optional_max(applied_residual_abs_values),
        "inverse_saturation_fraction": inverse_saturation_count
        / max(action_value_count, 1),
        "command_at_bound_fraction": command_at_bound_count
        / max(action_value_count, 1),
        "actuator_clip_fraction": actuator_clip_count / max(action_value_count, 1),
        "executed_commanded_action_mse": float(
            np.mean(squared_execution_deltas)
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }


def paired_difference(
    adapted: Mapping[str, object], baseline: Mapping[str, object]
) -> Dict[str, object]:
    adapted_returns = np.asarray(adapted["returns"], dtype=np.float64)
    baseline_returns = np.asarray(baseline["returns"], dtype=np.float64)
    if adapted_returns.shape != baseline_returns.shape:
        raise ValueError("paired evaluation return arrays do not align")
    differences = adapted_returns - baseline_returns
    return {
        "adapted_minus_baseline_returns": differences.tolist(),
        "return_difference_mean": float(differences.mean()),
        "return_difference_std": float(differences.std()),
        "positive_episode_count": int((differences > 0.0).sum()),
        "episode_count": int(differences.size),
    }


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if not (
        np.isfinite(args.reference_min_score)
        and np.isfinite(args.reference_max_score)
        and args.reference_max_score > args.reference_min_score
    ):
        raise ValueError("normalization references must be finite and ordered")
    if not np.isfinite(args.rollout_action_noise_beta) or (
        args.rollout_action_noise_beta < 0.0
    ):
        raise ValueError("rollout action-noise beta must be finite and non-negative")

    output_path = require_new_output_file(Path(args.output))
    device = canonical_device(args.device)
    base_checkpoint = Path(args.base_checkpoint).resolve()
    calibration_path = (
        Path(args.channel_calibration).resolve()
        if args.channel_calibration is not None
        else None
    )
    base_checkpoint_sha = sha256_file(base_checkpoint)
    base_agent, base_wrapper, base_step = load_frozen_physical_agent(
        base_checkpoint, device
    )
    channel_beta, calibration = resolve_channel_calibration(
        calibration_path, args.model_action_noise_beta
    )
    config = ChannelOPEXConfig(
        action_dim=int(base_agent.config.action_dim),
        step_size=float(args.step_size),
        channel_beta=float(channel_beta),
        gradient_noise_samples=int(args.gradient_noise_samples),
        gradient_steps=int(args.gradient_steps),
        delta_max=float(args.delta_max),
        baseline_transform=str(args.baseline_transform),
    )
    if calibration_path is not None:
        if calibration["action_dim"] != config.action_dim:
            raise ValueError("calibration and base checkpoint action dimensions differ")
        if not np.isclose(
            calibration["action_low"], -base_agent.config.max_action
        ) or not np.isclose(
            calibration["action_high"], base_agent.config.max_action
        ):
            raise ValueError("calibration and base checkpoint action bounds differ")

    environment_seeds = [
        _validate_seed(args.eval_seed + index, "environment seed")
        for index in range(args.eval_episodes)
    ]
    action_noise_seeds = [
        _validate_seed(args.eval_noise_seed + index, "actuator-noise seed")
        for index in range(args.eval_episodes)
    ]
    gradient_noise_seeds = [
        _validate_seed(args.gradient_noise_seed + index, "gradient-noise seed")
        for index in range(args.eval_episodes)
    ]

    baseline = evaluate_arm(
        base_agent,
        config,
        args.env_name,
        environment_seeds,
        action_noise_seeds,
        gradient_noise_seeds,
        float(args.rollout_action_noise_beta),
        adapted=False,
        reference_min=float(args.reference_min_score),
        reference_max=float(args.reference_max_score),
    )
    adapted = evaluate_arm(
        base_agent,
        config,
        args.env_name,
        environment_seeds,
        action_noise_seeds,
        gradient_noise_seeds,
        float(args.rollout_action_noise_beta),
        adapted=True,
        reference_min=float(args.reference_min_score),
        reference_max=float(args.reference_max_score),
    )

    source = Path(__file__).resolve()
    baseline_environment_steps = int(sum(baseline["lengths"]))
    adapted_environment_steps = int(sum(adapted["lengths"]))
    total_environment_steps = baseline_environment_steps + adapted_environment_steps
    total_wall_time = time.perf_counter() - started
    original_structure = bool(
        config.baseline_transform == "identity"
        and config.channel_beta == 0.0
        and config.gradient_noise_samples == 1
        and config.gradient_steps == 1
        and config.delta_max >= 2.0 * float(base_agent.config.max_action)
    )
    method_id = (
        "original_structure_opex_t1"
        if original_structure
        else f"channel_aware_opex_{config.baseline_transform}_anchor"
    )
    method_scope = (
        "original single-step OPEX structure with deterministic TD3+BC actor/Q1 "
        "and action-bound projection"
        if original_structure
        else (
            "stronger channel-aware extension of OPEX; not an unchanged "
            "reproduction of the original paper"
        )
    )
    result: Dict[str, object] = {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "method_id": method_id,
        "method_scope": method_scope,
        "opex_reference": {
            "paper": "Is Value Learning Really the Main Bottleneck in Offline RL?",
            "venue": "NeurIPS 2024",
            "url": OPEX_REFERENCE_URL,
            "original_single_step_rule": "a <- a + step_size * grad_a Q(s, a)",
        },
        "environment": args.env_name,
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": base_checkpoint_sha,
            "step": int(base_step),
            "format": base_wrapper.get("format"),
        },
        "calibration": calibration,
        "channel": {
            "gradient_model": "iid_uniform_additive_then_clip",
            "model_or_calibration_beta": float(channel_beta),
            "environment_rollout_beta": float(args.rollout_action_noise_beta),
            "beta_mismatch": bool(
                float(channel_beta) != float(args.rollout_action_noise_beta)
            ),
        },
        "controller": {
            "baseline_transform": config.baseline_transform,
            "step_size": float(config.step_size),
            "gradient_steps": int(config.gradient_steps),
            "K": int(config.gradient_noise_samples),
            "execution_noise_samples": int(config.gradient_noise_samples),
            "model_beta": float(channel_beta),
            "gradient_noise_seed": int(args.gradient_noise_seed),
            "q_reducer": "mean_q1",
            "delta_max": float(config.delta_max),
            "calibration_mode": calibration["source"],
            "critic": "frozen_q1",
            "gradient_objective": (
                "q1_of_action_bounded_command_with_deterministic_zero_channel_noise"
                if config.channel_beta == 0.0
                else (
                    "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
                    "uniform_noise_per_gradient_step"
                )
            ),
            "gradient_steps_per_action": int(config.gradient_steps),
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
        },
        "evaluation_protocol": {
            "environment": args.env_name,
            "episode_count_per_arm": int(args.eval_episodes),
            "paired_environment_and_action_noise_seeds": True,
            "environment_rng_namespace": "gymnasium_env_reset",
            "actuator_noise_rng_namespace": "numpy_generator_per_episode",
            "gradient_noise_rng_namespace": (
                "none_beta_zero_deterministic_objective"
                if config.channel_beta == 0.0
                else (
                    "torch_generator_per_episode_fresh_antithetic_"
                    "per_gradient_step"
                )
            ),
            "gradient_noise_sampling_frequency": (
                "none_beta_zero"
                if config.channel_beta == 0.0
                else "per_gradient_step"
            ),
            "gradient_noise_antithetic": bool(config.channel_beta > 0.0),
            "gradient_noise_seed_start": int(args.gradient_noise_seed),
            "environment_seed_start": int(args.eval_seed),
            "action_noise_seed_start": int(args.eval_noise_seed),
            "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling": True,
            "selection_rule": "all requested episodes retained",
        },
        "normalization": {
            "reference_min_score": float(args.reference_min_score),
            "reference_max_score": float(args.reference_max_score),
        },
        "arms": {"baseline_only": baseline, "adapted": adapted},
        "paired": paired_difference(adapted, baseline),
        "cost": {
            "cost_scope": "adapted_arm_deployment_controller_only",
            "environment_steps": adapted_environment_steps,
            "q1_forward_rows": int(adapted["q1_rows_total"]),
            "q1_backward_rows": int(adapted["q1_rows_total"]),
            "q1_backward_calls": int(adapted["q1_gradient_calls"]),
            "base_actor_rows": adapted_environment_steps,
            "baseline_environment_steps": baseline_environment_steps,
            "baseline_base_actor_rows": baseline_environment_steps,
            "paired_evaluation_environment_steps": total_environment_steps,
            "baseline_q1_rows_total": int(baseline["q1_rows_total"]),
            "adapted_q1_rows_total": int(adapted["q1_rows_total"]),
            "adapted_q1_rows_per_environment_step": int(
                config.gradient_noise_samples * config.gradient_steps
            ),
            "adapted_backward_calls": int(adapted["q1_gradient_calls"]),
            "wall_time_seconds": float(adapted["wall_time_seconds"]),
            "baseline_wall_time_seconds": float(baseline["wall_time_seconds"]),
            "paired_evaluation_wall_time_seconds": total_wall_time,
        },
        "implementation": {
            "evaluate_sha256": sha256_file(source),
            "inverse_residual_core_sha256": sha256_file(
                source.with_name("inverse_residual_core.py")
            ),
            "td3bc_core_sha256": sha256_file(source.with_name("td3bc_core.py")),
            "train_td3bc_sha256": sha256_file(source.with_name("train_td3bc.py")),
            "evaluation_controls_sha256": sha256_file(
                source.with_name("evaluation_controls.py")
            ),
            "train_inverse_residual_adapter_sha256": sha256_file(
                source.with_name("train_inverse_residual_adapter.py")
            ),
        },
        "wall_time_seconds": total_wall_time,
    }
    save_json_new(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--channel-calibration")
    parser.add_argument(
        "--model-action-noise-beta",
        type=float,
        help=(
            "known model-channel beta when no calibration is supplied; when "
            "combined with calibration it must exactly match beta_mle"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-name", default="Walker2d-v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step-size", type=float, required=True)
    parser.add_argument(
        "--baseline-transform",
        choices=BASELINE_TRANSFORMS,
        default="inverse",
    )
    parser.add_argument(
        "--gradient-noise-samples",
        "--k",
        dest="gradient_noise_samples",
        type=int,
        default=8,
        help="K antithetic target-channel Q1 rows per test-time action",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help=(
            "projected action-gradient steps; one preserves original OPEX's "
            "single-step structure, values above one are stronger CA-OPEX"
        ),
    )
    parser.add_argument("--gradient-noise-seed", type=int, default=59300)
    parser.add_argument(
        "--delta-max",
        type=float,
        default=CHANNEL_OPEX_DELTA_MAX,
        help="fixed-anchor L-infinity projection radius",
    )
    parser.add_argument(
        "--rollout-action-noise-beta",
        "--rollout-beta",
        dest="rollout_action_noise_beta",
        type=float,
        required=True,
        help="actual rollout-channel beta, independent of calibrated model beta",
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=39300)
    parser.add_argument("--eval-noise-seed", type=int, default=49300)
    parser.add_argument("--reference-min-score", type=float, default=1.629008)
    parser.add_argument("--reference-max-score", type=float, default=4592.3)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
