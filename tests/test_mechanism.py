from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from mechanism_audit import create_synthetic_hdf5
from mechanism_core import (
    audit_metrics,
    inject_reward_noise,
    load_d4rl_hdf5,
    make_trajectory_folds,
    reconstruct_trajectories,
    spearman_correlation,
    top_fraction_precision,
    trajectory_features,
)


def test_reconstruction_includes_terminal_timeout_and_trailing(tmp_path):
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.arange(18, dtype=np.float32).reshape(6, 3)
        handle["actions"] = np.zeros((6, 2), dtype=np.float32)
        handle["rewards"] = np.arange(6, dtype=np.float32)
        handle["terminals"] = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.bool_)
        handle["timeouts"] = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.bool_)
    arrays = load_d4rl_hdf5(path)
    trajectories = reconstruct_trajectories(arrays, arrays["rewards"].copy())
    assert [(t.start, t.stop) for t in trajectories] == [(0, 2), (2, 4), (4, 6)]
    assert [t.length for t in trajectories] == [2, 2, 2]


def test_noise_is_seeded_recorded_and_variance_matched():
    clean = np.linspace(-1, 1, 10000, dtype=np.float32)
    first, noise_first = inject_reward_noise(clean, 2.0, 101, "normal")
    second, noise_second = inject_reward_noise(clean, 2.0, 101, "normal")
    other, _ = inject_reward_noise(clean, 2.0, 102, "normal")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(noise_first, noise_second)
    np.testing.assert_allclose(first - clean, noise_first, atol=2e-7)
    assert not np.array_equal(first, other)
    assert noise_first.std() == pytest.approx(2.0, rel=0.04)


def test_folds_are_deterministic_balanced_and_trajectory_level():
    first = make_trajectory_folds(23, 5, 7)
    second = make_trajectory_folds(23, 5, 7)
    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first)
    assert counts.max() - counts.min() <= 1
    for fold in range(5):
        train = set(np.flatnonzero(first != fold).tolist())
        test = set(np.flatnonzero(first == fold).tolist())
        assert train.isdisjoint(test)


def test_trajectory_features_never_use_rewards(tmp_path):
    path = tmp_path / "synthetic.hdf5"
    create_synthetic_hdf5(path, n_trajectories=8, seed=2)
    arrays = load_d4rl_hdf5(path)
    trajectories = reconstruct_trajectories(arrays, arrays["rewards"])
    features_before = trajectory_features(trajectories)
    modified = dict(arrays)
    modified["rewards"] = arrays["rewards"] + 10000
    trajectories_after = reconstruct_trajectories(modified, modified["rewards"])
    np.testing.assert_array_equal(features_before, trajectory_features(trajectories_after))


def test_rank_and_top_fraction_metrics():
    clean = np.arange(10, dtype=np.float32)
    reverse = clean[::-1]
    assert spearman_correlation(clean, clean) == pytest.approx(1.0)
    assert spearman_correlation(reverse, clean) == pytest.approx(-1.0)
    assert top_fraction_precision(clean, clean, 0.2) == 1.0
    assert top_fraction_precision(reverse, clean, 0.2) == 0.0
    metrics = audit_metrics(clean, clean + 1, clean, np.ones(10), clean - 1)
    assert metrics["crossfit_mean"]["mse_vs_clean"] == 0.0
    assert metrics["calibration_against_latent_clean"]["coverage_95"] == 1.0


def test_synthetic_schema(tmp_path):
    path = tmp_path / "synthetic.hdf5"
    create_synthetic_hdf5(path, n_trajectories=10, min_length=3, max_length=5, seed=9)
    arrays = load_d4rl_hdf5(path)
    trajectories = reconstruct_trajectories(arrays, arrays["rewards"])
    assert len(trajectories) == 10
    assert sum(t.length for t in trajectories) == arrays["rewards"].size
    assert set(np.unique(arrays["terminals"] | arrays["timeouts"])) == {False, True}


def test_crossfit_predictions_and_provenance_when_torch_available():
    torch = pytest.importorskip("torch")
    from mechanism_core import fit_crossfit_heteroscedastic_mlp

    rng = np.random.default_rng(4)
    features = rng.normal(size=(24, 3)).astype(np.float32)
    targets = (2 * features[:, 0] - features[:, 1] + rng.normal(size=24)).astype(np.float32)
    folds = make_trajectory_folds(24, 3, 5)
    means, stds, provenance = fit_crossfit_heteroscedastic_mlp(
        features, targets, folds, hidden_dim=8, epochs=2, batch_size=16, model_seed=6
    )
    assert np.isfinite(means).all()
    assert (stds > 0).all()
    assert len(provenance) == 3
    for record in provenance:
        train = set(record["train_trajectory_ids"])
        fit = set(record["model_fit_trajectory_ids"])
        calibration = set(record["uncertainty_calibration_trajectory_ids"])
        test = set(record["test_trajectory_ids"])
        assert fit | calibration == train
        assert fit.isdisjoint(calibration)
        assert train.isdisjoint(test)
