"""Pure, dependency-light action transforms shared by TD3+BC evaluators."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


COMMAND_TRANSFORMS = ("identity", "uniform_mean_inverse")
COMMAND_SATURATION_DEFINITION = (
    "fraction_of_scaled_command_values_strictly_outside_environment_bounds_before_clip"
)
TRANSFORM_SATURATION_DEFINITION = (
    "fraction_of_desired_mean_action_values_outside_the_uniform_clipped_channel_"
    "mean_range_achievable_by_bounded_commands"
)


def validate_command_scale(command_scale: float) -> float:
    """Validate and canonicalize the command-gain evaluation control."""

    scale = float(command_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("command_scale must be finite and non-negative")
    return scale


def scale_and_clip_command(
    command: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    command_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Scale an agent command, then clip it to environment action bounds.

    The returned counters let callers aggregate command saturation separately
    from actuator-noise clipping. Multiplication uses the command dtype, so the
    default scale of one is bitwise unchanged for in-bounds floating commands.
    """

    scale = validate_command_scale(command_scale)
    command_array = np.asarray(command)
    if not np.issubdtype(command_array.dtype, np.floating):
        command_array = command_array.astype(np.float32)
    low_array = np.asarray(low, dtype=command_array.dtype)
    high_array = np.asarray(high, dtype=command_array.dtype)
    try:
        broadcast_low = np.broadcast_to(low_array, command_array.shape)
        broadcast_high = np.broadcast_to(high_array, command_array.shape)
    except ValueError as exc:
        raise ValueError("action bounds are not broadcastable to command shape") from exc
    if np.any(broadcast_low > broadcast_high):
        raise ValueError("action-space lower bounds exceed upper bounds")

    scale_value = np.asarray(scale, dtype=command_array.dtype)
    scaled = np.multiply(command_array, scale_value)
    outside = (scaled < broadcast_low) | (scaled > broadcast_high)
    clipped = np.clip(scaled, broadcast_low, broadcast_high).astype(
        command_array.dtype, copy=False
    )
    at_bound = (clipped <= broadcast_low) | (clipped >= broadcast_high)
    audit = {
        "command_value_count": int(command_array.size),
        "scaled_command_out_of_bounds_count": int(outside.sum()),
        "scaled_command_at_bound_count": int(at_bound.sum()),
    }
    return clipped, audit


def _validate_uniform_beta(beta: float) -> float:
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("uniform actuator-noise beta must be finite and non-negative")
    return value


def expected_clipped_uniform_action(
    command: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute E[clip(command + Uniform[-beta,beta], low, high)] exactly."""

    noise_beta = _validate_uniform_beta(beta)
    command_array = np.asarray(command, dtype=np.float64)
    low_array = np.broadcast_to(np.asarray(low, dtype=np.float64), command_array.shape)
    high_array = np.broadcast_to(np.asarray(high, dtype=np.float64), command_array.shape)
    if np.any(low_array > high_array):
        raise ValueError("action-space lower bounds exceed upper bounds")
    if noise_beta == 0.0:
        return np.clip(command_array, low_array, high_array)
    upper = command_array + noise_beta
    lower = command_array - noise_beta
    # clip(x,l,h) = x + (l-x)_+ - (x-h)_+. Integrating the two
    # hinges yields squared endpoints. This form avoids subtracting two large
    # antiderivative values when beta is small and the command is interior.
    lower_hinge_integral = (
        np.square(np.maximum(low_array - lower, 0.0))
        - np.square(np.maximum(low_array - upper, 0.0))
    ) / (4.0 * noise_beta)
    upper_hinge_integral = (
        np.square(np.maximum(upper - high_array, 0.0))
        - np.square(np.maximum(lower - high_array, 0.0))
    ) / (4.0 * noise_beta)
    return command_array + lower_hinge_integral - upper_hinge_integral


def inverse_uniform_clipped_mean(
    desired_mean_action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    beta: float,
    *,
    iterations: int = 48,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Invert the monotone clipped-uniform channel mean over bounded commands.

    Desired values outside the channel's achievable mean range are mapped to the
    nearest bounded command and counted as inverse-transform saturation.
    """

    noise_beta = _validate_uniform_beta(beta)
    desired_input = np.asarray(desired_mean_action)
    if not np.issubdtype(desired_input.dtype, np.floating):
        desired_input = desired_input.astype(np.float32)
    desired = desired_input.astype(np.float64)
    low_array = np.broadcast_to(np.asarray(low, dtype=np.float64), desired.shape)
    high_array = np.broadcast_to(np.asarray(high, dtype=np.float64), desired.shape)
    if np.any(low_array > high_array):
        raise ValueError("action-space lower bounds exceed upper bounds")
    desired = np.clip(desired, low_array, high_array)
    if noise_beta == 0.0:
        result = desired.astype(desired_input.dtype, copy=False)
        return result, {
            "command_transform_value_count": int(desired.size),
            "command_transform_saturation_count": 0,
            "transformed_command_at_bound_count": int(
                ((result <= low_array) | (result >= high_array)).sum()
            ),
        }
    if iterations <= 0:
        raise ValueError("inverse iterations must be positive")

    minimum_mean = expected_clipped_uniform_action(
        low_array, low_array, high_array, noise_beta
    )
    maximum_mean = expected_clipped_uniform_action(
        high_array, low_array, high_array, noise_beta
    )
    saturated_low = desired < minimum_mean
    saturated_high = desired > maximum_mean
    attainable_target = np.clip(desired, minimum_mean, maximum_mean)
    lower_command = low_array.copy()
    upper_command = high_array.copy()
    for _ in range(iterations):
        midpoint = 0.5 * (lower_command + upper_command)
        midpoint_mean = expected_clipped_uniform_action(
            midpoint, low_array, high_array, noise_beta
        )
        move_lower = midpoint_mean < attainable_target
        lower_command = np.where(move_lower, midpoint, lower_command)
        upper_command = np.where(move_lower, upper_command, midpoint)
    transformed = 0.5 * (lower_command + upper_command)
    transformed = np.where(saturated_low, low_array, transformed)
    transformed = np.where(saturated_high, high_array, transformed)
    transformed = transformed.astype(desired_input.dtype, copy=False)
    audit = {
        "command_transform_value_count": int(desired.size),
        "command_transform_saturation_count": int(
            (saturated_low | saturated_high).sum()
        ),
        "transformed_command_at_bound_count": int(
            ((transformed <= low_array) | (transformed >= high_array)).sum()
        ),
    }
    return transformed, audit


def prepare_evaluation_command(
    raw_command: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    command_scale: float = 1.0,
    command_transform: str = "identity",
    actuator_noise_beta: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Apply the documented scale-then-channel-inverse evaluation pipeline."""

    desired, scale_audit = scale_and_clip_command(
        raw_command, low, high, command_scale
    )
    if command_transform not in COMMAND_TRANSFORMS:
        raise ValueError(
            f"command_transform must be one of {COMMAND_TRANSFORMS}, got "
            f"{command_transform!r}"
        )
    if command_transform == "identity":
        transformed = desired
        transform_audit = {
            "command_transform_value_count": int(desired.size),
            "command_transform_saturation_count": 0,
            "transformed_command_at_bound_count": int(
                ((desired <= np.asarray(low)) | (desired >= np.asarray(high))).sum()
            ),
        }
    else:
        transformed, transform_audit = inverse_uniform_clipped_mean(
            desired, low, high, actuator_noise_beta
        )
    return transformed, {**scale_audit, **transform_audit}
