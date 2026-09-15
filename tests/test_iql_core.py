import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iql_core import normalized_rank, posterior_expected_rank
from train_iql import (
    _contiguous_episode_slices,
    build_parser as build_iql_parser,
    build_hubl_fields,
    discounted_cumsum,
    prepare_dataset,
)


def _write_paired_episodic_dataset(path, energy_episodes, boundary_types):
    """Write scalar-action episodes whose squared residuals equal ``energy``."""

    import h5py

    with h5py.File(path, "w") as handle:
        for episode_id, (energies, boundary_type) in enumerate(
            zip(energy_episodes, boundary_types)
        ):
            energies = np.asarray(energies, dtype=np.float32)
            length = energies.size
            group = handle.create_group(f"episode_{episode_id}")
            group["observations"] = np.arange(
                (length + 1) * 2, dtype=np.float32
            ).reshape(length + 1, 2)
            group["actions"] = np.sqrt(energies)[:, None]
            group["clean_policy_actions"] = np.zeros((length, 1), dtype=np.float32)
            group["rewards"] = np.full(
                length, float(episode_id + 2), dtype=np.float32
            )
            terminations = np.zeros(length, dtype=np.bool_)
            truncations = np.zeros(length, dtype=np.bool_)
            if boundary_type == "terminal":
                terminations[-1] = True
            elif boundary_type == "timeout":
                truncations[-1] = True
            else:
                raise ValueError(boundary_type)
            group["terminations"] = terminations
            group["truncations"] = truncations


def _prepare(path, discount):
    return prepare_dataset(
        path,
        discount=discount,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )


def test_discounted_cumsum():
    result = discounted_cumsum(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), 0.5)
    np.testing.assert_allclose(result, [2.75, 3.5, 3.0])


def test_normalized_rank_ties():
    result = normalized_rank(np.asarray([3.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(result, [1.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])


def test_posterior_rank_matches_order_at_low_uncertainty():
    means = np.asarray([-2.0, 0.0, 2.0])
    ranks = posterior_expected_rank(means, np.full(3, 1e-4))
    np.testing.assert_allclose(ranks, [0.0, 0.5, 1.0], atol=1e-5)


def test_posterior_rank_shrinks_uncertain_extreme():
    means = np.asarray([0.0, 1.0, 2.0])
    confident = posterior_expected_rank(means, np.full(3, 0.01))
    uncertain = posterior_expected_rank(means, np.asarray([0.01, 0.01, 100.0]))
    assert uncertain[-1] < confident[-1]
    assert uncertain[-1] == pytest.approx(0.50598, abs=0.01)


def test_episode_hubl_fields(synthetic_hdf5):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    heuristic, lambdas, metrics = build_hubl_fields(dataset, "hubl_rank", 1.0, 0.5, {})
    assert heuristic.shape == lambdas.shape == dataset.noisy_rewards.shape
    assert 0.0 <= lambdas.min() <= lambdas.max() <= 1.0
    assert metrics["episode_score_spearman_vs_clean"] == pytest.approx(1.0)


def test_horizon_reliability_is_closed_form_and_not_quality_rank(synthetic_hdf5):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    heuristic, lambdas, metrics = build_hubl_fields(
        dataset,
        "hubl_horizon",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.02,
    )
    np.testing.assert_allclose(heuristic, dataset.raw_mc_next)
    assert lambdas.shape == dataset.noisy_rewards.shape
    # Episode 0 is a length-four timeout whose last transition is dropped.
    # Its retained suffix horizons are [3, 2, 1], so accumulated stochastic
    # variance decreases and heuristic trust increases toward the boundary.
    first = lambdas[dataset.episode_ids == 0]
    assert np.all(np.diff(first) > 0.0)
    assert metrics["lambda_mean"] == pytest.approx(float(lambdas.mean()))
    assert metrics["effective_horizon_max"] > metrics["effective_horizon_mean"]

    _, zero_scale, _ = build_hubl_fields(
        dataset,
        "hubl_horizon",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.0,
    )
    np.testing.assert_allclose(zero_scale, np.ones_like(zero_scale))


@pytest.mark.parametrize(
    "variant",
    ("hubl_horizon", "hubl_horizon_shuffled", "hubl_horizon_reverse"),
)
def test_horizon_zero_scale_is_constant_for_main_and_controls(
    synthetic_hdf5, variant
):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    _, lambdas, metrics = build_hubl_fields(
        dataset,
        variant,
        0.7,
        0.5,
        {},
        horizon_noise_scale=0.0,
        horizon_control_seed=31,
    )
    np.testing.assert_array_equal(lambdas, np.full_like(lambdas, 0.7))
    assert metrics["lambda_multiset_exactly_preserved"] is True


def test_horizon_shuffled_preserves_exact_multiset_and_is_reproducible(
    synthetic_hdf5,
):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    _, base, _ = build_hubl_fields(
        dataset, "hubl_horizon", 1.0, 0.5, {}, horizon_noise_scale=0.02
    )
    _, shuffled_a, metrics_a = build_hubl_fields(
        dataset,
        "hubl_horizon_shuffled",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.02,
        horizon_control_seed=31,
    )
    _, shuffled_b, metrics_b = build_hubl_fields(
        dataset,
        "hubl_horizon_shuffled",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.02,
        horizon_control_seed=31,
    )
    _, shuffled_other_seed, _ = build_hubl_fields(
        dataset,
        "hubl_horizon_shuffled",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.02,
        horizon_control_seed=32,
    )
    np.testing.assert_array_equal(np.sort(shuffled_a), np.sort(base))
    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert not np.array_equal(shuffled_a, base)
    assert not np.array_equal(shuffled_a, shuffled_other_seed)
    assert metrics_a["horizon_control_seed"] == 31
    assert metrics_a["horizon_control_seed_used"] is True
    assert metrics_a["lambda_multiset_exactly_preserved"] is True
    assert metrics_a["per_episode_lambda_multisets_exactly_preserved"] is None
    assert (
        metrics_a["lambda_multiset_sha256_before_control"]
        == metrics_a["lambda_multiset_sha256_after_control"]
    )
    assert (
        metrics_a["lambda_sequence_sha256_after_control"]
        == metrics_b["lambda_sequence_sha256_after_control"]
    )


def test_horizon_reverse_flips_direction_within_each_episode_only(
    synthetic_hdf5,
):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    heuristic, base, _ = build_hubl_fields(
        dataset, "hubl_horizon", 1.0, 0.5, {}, horizon_noise_scale=0.02
    )
    reversed_heuristic, reversed_lambdas, metrics = build_hubl_fields(
        dataset,
        "hubl_horizon_reverse",
        1.0,
        0.5,
        {},
        horizon_noise_scale=0.02,
        horizon_control_seed=91,
    )
    np.testing.assert_array_equal(reversed_heuristic, heuristic)
    for episode_id in np.unique(dataset.episode_ids):
        mask = dataset.episode_ids == episode_id
        np.testing.assert_array_equal(reversed_lambdas[mask], base[mask][::-1])
        np.testing.assert_array_equal(
            np.sort(reversed_lambdas[mask]), np.sort(base[mask])
        )
    first = reversed_lambdas[dataset.episode_ids == 0]
    assert np.all(np.diff(first) < 0.0)
    assert metrics["horizon_control_seed"] == 91
    assert metrics["horizon_control_seed_used"] is False
    assert metrics["lambda_multiset_exactly_preserved"] is True
    assert metrics["per_episode_lambda_multisets_exactly_preserved"] is True


def test_rank_horizon_is_exact_product_of_quality_and_reliability(
    synthetic_hdf5,
):
    dataset = prepare_dataset(
        synthetic_hdf5,
        discount=0.99,
        noise_seed=2,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    _, quality, _ = build_hubl_fields(
        dataset, "hubl_rank", 1.0, 0.5, {}
    )
    _, reliability, _ = build_hubl_fields(
        dataset, "hubl_horizon", 0.7, 0.5, {}, horizon_noise_scale=0.02
    )
    heuristic, product, metrics = build_hubl_fields(
        dataset,
        "hubl_rank_horizon",
        0.7,
        0.5,
        {},
        horizon_noise_scale=0.02,
    )
    np.testing.assert_array_equal(heuristic, dataset.raw_mc_next)
    np.testing.assert_array_equal(product, quality * reliability)
    assert metrics["quality_gate_mean"] == pytest.approx(float(quality.mean()))
    assert metrics["quality_gate_min"] == pytest.approx(float(quality.min()))
    assert metrics["quality_gate_max"] == pytest.approx(float(quality.max()))
    assert metrics["reliability_gate_mean"] == pytest.approx(
        float(reliability.mean())
    )
    assert metrics["reliability_gate_min"] == pytest.approx(
        float(reliability.min())
    )
    assert metrics["reliability_gate_max"] == pytest.approx(
        float(reliability.max())
    )
    assert metrics["product_lambda_mean"] == pytest.approx(float(product.mean()))
    assert metrics["product_lambda_min"] == pytest.approx(float(product.min()))
    assert metrics["product_lambda_max"] == pytest.approx(float(product.max()))
    assert metrics["lambda_mean"] == metrics["product_lambda_mean"]


def test_episode_slice_builder_rejects_noncontiguous_episode_ids():
    with pytest.raises(ValueError, match="nondecreasing"):
        _contiguous_episode_slices(np.asarray([0, 0, 1, 1, 0], dtype=np.int64))


def test_action_residual_episodic_timeout_includes_dropped_final_raw_step(
    tmp_path,
):
    path = tmp_path / "paired_episodic.hdf5"
    _write_paired_episodic_dataset(
        path,
        ([1.0, 4.0, 9.0, 16.0], [1.0, 1.0]),
        ("timeout", "terminal"),
    )
    dataset = _prepare(path, discount=0.5)
    raw_mean = (1.0 + 4.0 + 9.0 + 16.0 + 1.0 + 1.0) / 6.0
    # Episode zero has four raw steps but only three learner transitions.
    # Its last retained transition must still see raw step three's energy.
    expected = np.asarray(
        [
            (4.0 + 0.25 * 9.0 + 0.25**2 * 16.0) / raw_mean,
            (9.0 + 0.25 * 16.0) / raw_mean,
            16.0 / raw_mean,
        ],
        dtype=np.float32,
    )
    first_episode = dataset.action_residual_next_suffix[dataset.episode_ids == 0]
    np.testing.assert_allclose(first_episode, expected, rtol=1e-6, atol=1e-6)
    assert dataset.remaining_steps[dataset.episode_ids == 0][-1] == 1
    assert dataset.raw_mc_next[dataset.episode_ids == 0][-1] > 0.0
    metadata = dataset.action_residual_metadata
    assert metadata["raw_transition_count"] == 6
    assert metadata["residual_energy_raw_mean"] == pytest.approx(raw_mean)
    assert metadata["residual_energy_raw_std"] > 0.0
    assert metadata["next_suffix_D_min"] == pytest.approx(
        float(dataset.action_residual_next_suffix.min())
    )
    assert metadata["next_suffix_D_max"] == pytest.approx(
        float(dataset.action_residual_next_suffix.max())
    )
    assert len(metadata["residual_energy_raw_sequence_sha256"]) == 64
    assert len(metadata["next_suffix_D_sequence_sha256"]) == 64
    assert "dropped timeout" in metadata["alignment_semantics"]


def test_action_residual_flat_timeout_uses_exact_raw_mc_next_alignment(tmp_path):
    import h5py

    path = tmp_path / "paired_flat.hdf5"
    energies = np.asarray([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle["observations"] = np.arange(10, dtype=np.float32).reshape(5, 2)
        handle["actions"] = np.sqrt(energies)[:, None]
        handle["clean_policy_actions"] = np.zeros((5, 1), dtype=np.float32)
        handle["rewards"] = np.asarray([1, 1, 2, 2, 2], dtype=np.float32)
        handle["terminals"] = np.asarray([0, 1, 0, 0, 0], dtype=np.bool_)
        handle["timeouts"] = np.asarray([0, 0, 0, 0, 1], dtype=np.bool_)
    dataset = _prepare(path, discount=0.5)
    # True terminal index one is retained and has an empty next suffix.
    assert dataset.action_residual_next_suffix[1] == pytest.approx(0.0)
    assert dataset.raw_mc_next[1] == pytest.approx(0.0)
    # Timeout index four is dropped, but its energy/reward form the next suffix
    # of retained raw index three.
    assert dataset.action_residual_next_suffix[-1] == pytest.approx(25.0 / 11.0)
    assert dataset.raw_mc_next[-1] > 0.0
    np.testing.assert_array_equal(dataset.episode_ids, [0, 0, 1, 1])


def test_constant_residual_energy_reduces_exactly_to_horizon_proxy(tmp_path):
    path = tmp_path / "constant_energy.hdf5"
    _write_paired_episodic_dataset(
        path,
        ([4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0]),
        ("timeout", "terminal"),
    )
    dataset = _prepare(path, discount=0.9)
    _, horizon, _ = build_hubl_fields(
        dataset, "hubl_horizon", 0.8, 0.5, {}, horizon_noise_scale=0.07
    )
    heuristic, residual, metrics = build_hubl_fields(
        dataset,
        "hubl_action_residual",
        0.8,
        0.5,
        {},
        horizon_noise_scale=0.07,
    )
    np.testing.assert_array_equal(heuristic, dataset.raw_mc_next)
    np.testing.assert_allclose(dataset.action_residual_next_suffix, (
        1.0 - np.power(0.9**2, dataset.remaining_steps)
    ) / (1.0 - 0.9**2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(residual, horizon, rtol=1e-6, atol=1e-6)
    assert metrics["lambda_mean"] == pytest.approx(float(residual.mean()))
    assert metrics["lambda_min"] == pytest.approx(float(residual.min()))
    assert metrics["lambda_max"] == pytest.approx(float(residual.max()))
    assert "e_k/e_bar" in metrics["action_residual_formula"]


def test_action_residual_shuffle_preserves_exact_lambda_multiset(tmp_path):
    path = tmp_path / "variable_energy.hdf5"
    _write_paired_episodic_dataset(
        path,
        ([1.0, 4.0, 9.0, 16.0], [2.0, 3.0, 5.0, 8.0]),
        ("timeout", "terminal"),
    )
    dataset = _prepare(path, discount=0.9)
    _, base, _ = build_hubl_fields(
        dataset, "hubl_action_residual", 1.0, 0.5, {},
        horizon_noise_scale=0.03, horizon_control_seed=47
    )
    _, shuffled_a, metrics_a = build_hubl_fields(
        dataset, "hubl_action_residual_shuffled", 1.0, 0.5, {},
        horizon_noise_scale=0.03, horizon_control_seed=47
    )
    _, shuffled_b, metrics_b = build_hubl_fields(
        dataset, "hubl_action_residual_shuffled", 1.0, 0.5, {},
        horizon_noise_scale=0.03, horizon_control_seed=47
    )
    np.testing.assert_array_equal(np.sort(shuffled_a), np.sort(base))
    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert not np.array_equal(shuffled_a, base)
    assert metrics_a["lambda_multiset_exactly_preserved"] is True
    assert (
        metrics_a["lambda_multiset_sha256_before_control"]
        == metrics_a["lambda_multiset_sha256_after_control"]
    )
    assert (
        metrics_a["lambda_sequence_sha256_after_control"]
        == metrics_b["lambda_sequence_sha256_after_control"]
    )


def test_action_residual_requires_clean_policy_actions(synthetic_hdf5):
    dataset = _prepare(synthetic_hdf5, discount=0.99)
    with pytest.raises(KeyError, match="requires HDF5 field clean_policy_actions"):
        build_hubl_fields(
            dataset,
            "hubl_action_residual",
            1.0,
            0.5,
            {},
            horizon_noise_scale=0.02,
        )


@pytest.mark.parametrize(
    "variant", ("hubl_action_residual", "hubl_action_residual_shuffled")
)
def test_iql_parser_accepts_action_residual_variants(variant):
    args = build_iql_parser().parse_args(
        [
            "--dataset", "dataset.hdf5",
            "--output-dir", "output",
            "--variant", variant,
        ]
    )
    assert args.variant == variant


@pytest.fixture
def synthetic_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        for episode_id, length in enumerate((4, 5, 6)):
            group = handle.create_group(f"episode_{episode_id}")
            observations = np.arange((length + 1) * 2, dtype=np.float32).reshape(length + 1, 2)
            group.create_dataset("observations", data=observations + episode_id)
            group.create_dataset("actions", data=np.zeros((length, 1), dtype=np.float32))
            group.create_dataset("rewards", data=np.full(length, episode_id + 1, dtype=np.float32))
            terminals = np.zeros(length, dtype=np.bool_)
            truncations = np.zeros(length, dtype=np.bool_)
            if episode_id % 2:
                terminals[-1] = True
            else:
                truncations[-1] = True
            group.create_dataset("terminations", data=terminals)
            group.create_dataset("truncations", data=truncations)
    return path
