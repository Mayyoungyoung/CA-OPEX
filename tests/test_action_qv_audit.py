from __future__ import annotations

import numpy as np

from action_qv_audit import (
    correlation,
    episode_ranges,
    paired_bootstrap_vs_reference,
    quantile_selection,
)


def test_episode_ranges_retains_final_partial_segment() -> None:
    terminals = np.asarray([0, 1, 0, 0, 0], dtype=bool)
    timeouts = np.asarray([0, 0, 0, 1, 0], dtype=bool)
    assert episode_ranges(terminals, timeouts) == [(0, 2), (2, 4), (4, 5)]


def test_correlations_and_stable_quantile_selection() -> None:
    score = np.arange(10, dtype=np.float64)
    target = score * 2.0
    metrics = correlation(score, target)
    assert metrics["n"] == 10
    assert np.isclose(metrics["pearson"], 1.0)
    assert np.isclose(metrics["spearman"], 1.0)
    result = quantile_selection(score, {"target": target}, fraction=0.2)
    assert result["top_indices"] == [8, 9]
    assert result["bottom_indices"] == [0, 1]
    assert result["targets"]["target"]["top_minus_bottom"] == 16.0


def test_paired_bootstrap_is_seeded_and_favors_perfect_candidate() -> None:
    target = np.asarray([3.0, -1.0, 2.0, 0.0, 5.0, 4.0, 1.0])
    signals = {
        "reference": -target,
        "candidate": target.copy(),
    }
    kwargs = dict(
        signals=signals,
        targets={"target": target},
        reference="reference",
        candidates=("candidate",),
        replicates=30,
        seed=17,
        top_fraction=0.2,
    )
    first = paired_bootstrap_vs_reference(**kwargs)
    second = paired_bootstrap_vs_reference(**kwargs)
    assert first == second
    interval = first["comparisons"]["candidate"]["target"][
        "spearman_difference_candidate_minus_reference"
    ]
    assert interval["mean"] > 1.5
    assert interval["lower_95"] > 0.0
