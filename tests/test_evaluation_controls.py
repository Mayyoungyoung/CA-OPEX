import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_controls import (
    expected_clipped_uniform_action,
    inverse_uniform_clipped_mean,
    prepare_evaluation_command,
    scale_and_clip_command,
    validate_command_scale,
)


def test_default_scale_is_bitwise_unchanged_for_in_bounds_float32_commands():
    command = np.asarray([0.25, -0.75, 0.0], dtype=np.float32)
    low = np.full(3, -1.0, dtype=np.float32)
    high = np.full(3, 1.0, dtype=np.float32)
    transformed, audit = scale_and_clip_command(command, low, high)
    np.testing.assert_array_equal(transformed, command)
    assert transformed.dtype == command.dtype
    assert audit == {
        "command_value_count": 3,
        "scaled_command_out_of_bounds_count": 0,
        "scaled_command_at_bound_count": 0,
    }


def test_default_control_preserves_existing_noisy_action_pipeline_bitwise():
    raw_command = np.asarray([0.9, -0.8, 0.1], dtype=np.float32)
    perturbation = np.asarray([0.4, -0.4, 0.2], dtype=np.float32)
    low = np.full(3, -1.0, dtype=np.float32)
    high = np.full(3, 1.0, dtype=np.float32)
    previous_executed = np.clip(raw_command + perturbation, low, high).astype(
        np.float32, copy=False
    )
    transformed, _ = prepare_evaluation_command(
        raw_command,
        low,
        high,
        command_scale=1.0,
        command_transform="identity",
        actuator_noise_beta=1.0,
    )
    controlled_executed = np.clip(transformed + perturbation, low, high).astype(
        np.float32, copy=False
    )
    np.testing.assert_array_equal(controlled_executed, previous_executed)


def test_scale_precedes_command_clip_and_reports_saturation_exactly():
    command = np.asarray([0.75, -0.25, 0.5], dtype=np.float32)
    transformed, audit = scale_and_clip_command(
        command,
        np.asarray([-1.0, -1.0, -0.4], dtype=np.float32),
        np.asarray([1.0, 1.0, 0.8], dtype=np.float32),
        command_scale=2.0,
    )
    np.testing.assert_array_equal(
        transformed, np.asarray([1.0, -0.5, 0.8], dtype=np.float32)
    )
    assert audit["command_value_count"] == 3
    assert audit["scaled_command_out_of_bounds_count"] == 2
    assert audit["scaled_command_at_bound_count"] == 2


@pytest.mark.parametrize("scale", (-1.0, float("nan"), float("inf")))
def test_command_scale_must_be_finite_and_non_negative(scale):
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_command_scale(scale)


def test_zero_scale_is_valid_and_produces_zero_commands():
    transformed, audit = scale_and_clip_command(
        np.asarray([0.9, -0.4], dtype=np.float64),
        np.asarray([-1.0, -1.0]),
        np.asarray([1.0, 1.0]),
        command_scale=0.0,
    )
    np.testing.assert_array_equal(transformed, np.zeros(2))
    assert audit["scaled_command_out_of_bounds_count"] == 0


def test_uniform_mean_inverse_matches_independent_high_precision_quadrature():
    desired = np.asarray([-0.6, 0.0, 0.6], dtype=np.float64)
    low = np.full(3, -1.0)
    high = np.full(3, 1.0)
    beta = 0.7
    command, audit = inverse_uniform_clipped_mean(desired, low, high, beta)
    recovered_analytic = expected_clipped_uniform_action(command, low, high, beta)
    np.testing.assert_allclose(recovered_analytic, desired, atol=1e-12, rtol=0.0)

    # Midpoint quadrature is deliberately independent of the analytic integral
    # used by the implementation.
    sample_count = 200_000
    noise = -beta + (np.arange(sample_count) + 0.5) * (2.0 * beta / sample_count)
    recovered_quadrature = np.clip(
        command[:, None] + noise[None, :], low[:, None], high[:, None]
    ).mean(axis=1)
    np.testing.assert_allclose(recovered_quadrature, desired, atol=2e-6, rtol=0.0)
    assert audit["command_transform_saturation_count"] == 0


def test_uniform_inverse_saturates_unattainable_targets_and_is_monotone():
    desired = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    command, audit = inverse_uniform_clipped_mean(desired, -1.0, 1.0, beta=1.0)
    assert np.all(np.diff(command) >= 0.0)
    assert command[0] == -1.0
    assert command[-1] == 1.0
    assert audit["command_transform_saturation_count"] == 2


def test_zero_beta_uniform_inverse_is_strict_identity():
    desired = np.asarray([-1.0, -0.2, 0.3, 1.0], dtype=np.float32)
    transformed, audit = prepare_evaluation_command(
        desired,
        np.full(4, -1.0, dtype=np.float32),
        np.full(4, 1.0, dtype=np.float32),
        command_transform="uniform_mean_inverse",
        actuator_noise_beta=0.0,
    )
    np.testing.assert_array_equal(transformed, desired)
    assert audit["command_transform_saturation_count"] == 0


def test_uniform_channel_mean_is_stable_and_continuous_at_tiny_beta():
    command = np.asarray([-0.25, 0.0, 0.75], dtype=np.float64)
    expected = expected_clipped_uniform_action(command, -1.0, 1.0, beta=1e-12)
    np.testing.assert_array_equal(expected, command)
    inverse, audit = inverse_uniform_clipped_mean(command, -1.0, 1.0, beta=1e-12)
    np.testing.assert_allclose(inverse, command, atol=1e-14, rtol=0.0)
    assert audit["command_transform_saturation_count"] == 0
