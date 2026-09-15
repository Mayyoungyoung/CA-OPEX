import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continuation_variance_audit import (
    bootstrap_geometric_fit,
    covariance_diagnostics,
    discounted_prefix_returns,
    effective_horizons,
    episode_coordinates,
    fit_geometric_curve,
    stratified_anchor_indices,
)


def test_episode_coordinates_include_current_logged_action():
    terminals = np.asarray([0, 1, 0, 0], dtype=np.bool_)
    timeouts = np.asarray([0, 0, 0, 1], dtype=np.bool_)
    episode_ids, steps, suffixes = episode_coordinates(terminals, timeouts)
    np.testing.assert_array_equal(episode_ids, [0, 0, 1, 1])
    np.testing.assert_array_equal(steps, [0, 1, 0, 1])
    np.testing.assert_array_equal(suffixes, [2, 1, 2, 1])


def test_discounted_prefix_returns_zero_pad_after_termination():
    values = discounted_prefix_returns(np.asarray([2.0, 4.0]), [1, 2, 5], 0.5)
    np.testing.assert_allclose(values, [2.0, 4.0, 4.0])


def test_geometric_fit_is_exact_for_geometric_variance():
    horizons = [1, 5, 10]
    design = effective_horizons(horizons, 0.9)
    variances = np.stack((2.0 * design, 4.0 * design))
    result = fit_geometric_curve(variances, horizons, 0.9)
    assert result["fitted_scale"] == pytest.approx(3.0)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["normalized_rmse_at_max_horizon_scale"] == pytest.approx(0.0)
    bootstrap = bootstrap_geometric_fit(
        variances, horizons, 0.9, resamples=20, seed=3
    )
    assert bootstrap["intervals"]["r_squared"] == pytest.approx([1.0, 1.0, 1.0])


def test_stratified_anchor_selection_obeys_time_limit_and_is_deterministic():
    suffixes = np.asarray([1, 2, 3, 4, 5, 6])
    steps = np.asarray([0, 1, 9, 2, 3, 4])
    kwargs = dict(
        bin_starts=[1, 4],
        anchors_per_bin=1,
        maximum_episode_steps=10,
        maximum_rollout_horizon=5,
        seed=7,
    )
    first, labels = stratified_anchor_indices(suffixes, steps, **kwargs)
    second, _ = stratified_anchor_indices(suffixes, steps, **kwargs)
    np.testing.assert_array_equal(first, second)
    assert labels == ["1-3", "4+"]
    assert np.all(steps[first] <= 5)


def test_covariance_diagnostics_detects_off_diagonal_dependence():
    # Perfectly correlated reward innovations at two times make half of the
    # undiscounted return variance an off-diagonal contribution.
    matrix = np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    result = covariance_diagnostics([matrix], [1, 2], 1.0)
    row = result["by_horizon"][1]
    assert row["off_diagonal_fraction"] == pytest.approx(0.5)
