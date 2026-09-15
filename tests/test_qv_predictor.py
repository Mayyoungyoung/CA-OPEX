import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qv_predictor import (
    PredictorConfig,
    crossfit_predict,
    discounted_cumsum,
    prepare_flat_dataset,
    save_predictions,
)


def _write_flat(path: Path, episode_specs):
    h5py = pytest.importorskip("h5py")
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    for spec in episode_specs:
        length = int(spec["actions"].shape[0])
        observations.append(np.asarray(spec["observations"], dtype=np.float32))
        actions.append(np.asarray(spec["actions"], dtype=np.float32))
        rewards.append(np.asarray(spec["rewards"], dtype=np.float32))
        terminal = np.zeros(length, dtype=np.bool_)
        timeout = np.zeros(length, dtype=np.bool_)
        if spec.get("terminal", False):
            terminal[-1] = True
        if spec.get("timeout", False):
            timeout[-1] = True
        terminals.append(terminal)
        timeouts.append(timeout)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=np.concatenate(observations))
        handle.create_dataset("actions", data=np.concatenate(actions))
        handle.create_dataset("rewards", data=np.concatenate(rewards))
        handle.create_dataset("terminals", data=np.concatenate(terminals))
        handle.create_dataset("timeouts", data=np.concatenate(timeouts))


def _mixed_boundary_dataset(path: Path) -> Path:
    specs = []
    offset = 0
    for episode_id, (length, terminal, timeout) in enumerate(
        ((3, True, False), (4, False, True), (3, True, False), (4, False, True))
    ):
        observations = np.stack(
            [
                np.arange(offset, offset + length, dtype=np.float32),
                np.full(length, episode_id, dtype=np.float32),
            ],
            axis=1,
        )
        actions = np.full((length, 1), episode_id - 1.5, dtype=np.float32)
        rewards = np.arange(1, length + 1, dtype=np.float32) + episode_id
        specs.append(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminal": terminal,
                "timeout": timeout,
            }
        )
        offset += length
    _write_flat(path, specs)
    return path


def _action_quality_dataset(path: Path, episodes: int = 60, length: int = 4) -> Path:
    specs = []
    for episode_id in range(episodes):
        quality = 1.0 if episode_id % 2 == 0 else -1.0
        time_feature = np.arange(length, dtype=np.float32) / float(length - 1)
        observations = np.stack(
            (time_feature, np.ones(length, dtype=np.float32)), axis=1
        )
        actions = np.full((length, 1), quality, dtype=np.float32)
        # State alone cannot identify quality; action makes the return linear.
        rewards = np.full(length, 2.0 * quality, dtype=np.float32)
        specs.append(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminal": True,
            }
        )
    _write_flat(path, specs)
    return path


def _ridge_config(**overrides):
    values = dict(
        discount=0.9,
        noise_seed=19,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
        folds=3,
        fold_seed=23,
        model_seed=29,
        ensemble_size=1,
        backend="ridge",
        ridge_l2=1e-6,
    )
    values.update(overrides)
    return PredictorConfig(**values)


def test_discounted_return_and_exact_kept_order(tmp_path):
    path = _mixed_boundary_dataset(tmp_path / "mixed.hdf5")
    first = prepare_flat_dataset(
        path,
        discount=0.5,
        noise_seed=7,
        iid_noise_scale=0.4,
        episode_noise_scale=0.3,
    )
    second = prepare_flat_dataset(
        path,
        discount=0.5,
        noise_seed=7,
        iid_noise_scale=0.4,
        episode_noise_scale=0.3,
    )
    # True terminals keep the final row; timeout episodes drop it.
    np.testing.assert_array_equal(
        first.source_indices,
        np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        first.episode_ids,
        np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64),
    )
    np.testing.assert_allclose(first.noisy_rewards, second.noisy_rewards, rtol=0, atol=0)
    np.testing.assert_allclose(first.episode_biases, second.episode_biases, rtol=0, atol=0)
    assert first.kept_order_sha256 == second.kept_order_sha256
    # Episode zero rewards are [1,2,3], hence G=[2.75,3.5,3].
    np.testing.assert_allclose(first.clean_return_current[:3], [2.75, 3.5, 3.0])
    np.testing.assert_allclose(first.clean_return_next[:3], [3.5, 3.0, 0.0])
    np.testing.assert_allclose(discounted_cumsum(np.asarray([1, 2, 3]), 0.5), [2.75, 3.5, 3])


def test_crossfit_shapes_archive_and_fold_provenance(tmp_path):
    path = _action_quality_dataset(tmp_path / "quality.hdf5", episodes=18)
    config = _ridge_config(ensemble_size=3)
    dataset = prepare_flat_dataset(
        path,
        discount=config.discount,
        noise_seed=config.noise_seed,
        iid_noise_scale=config.iid_noise_scale,
        episode_noise_scale=config.episode_noise_scale,
    )
    arrays, metadata = crossfit_predict(dataset, config)
    transitions = dataset.observations.shape[0]
    episodes = len(dataset.episodes)
    for key in (
        "v_current",
        "q_current",
        "v_next",
        "v_current_std",
        "q_current_std",
        "v_next_std",
        "crossfit_next_mean",
        "crossfit_next_std",
        "kept_fold_ids",
        "episode_ids",
        "kept_source_indices",
    ):
        assert arrays[key].shape == (transitions,)
    for key in (
        "crossfit_mean",
        "crossfit_std",
        "controllable_score",
        "controllable_std",
        "fold_ids",
    ):
        assert arrays[key].shape == (episodes,)
    assert metadata["fold_integrity"] is True
    for fold in metadata["fold_provenance"]:
        assert set(fold["train_episode_ids"]).isdisjoint(fold["heldout_episode_ids"])
    np.testing.assert_array_equal(arrays["kept_fold_ids"], arrays["fold_ids"][arrays["episode_ids"]])
    np.testing.assert_allclose(arrays["crossfit_next_mean"], arrays["v_next"])
    assert arrays["prediction_units"].item() == "raw"
    embedded = json.loads(arrays["config_json"].item())
    assert embedded["kept_order_sha256"] == dataset.kept_order_sha256

    output = tmp_path / "predictions.npz"
    manifest = save_predictions(output, arrays, metadata)
    assert output.exists() and output.with_suffix(".json").exists()
    assert len(manifest["archive_sha256"]) == 64
    with np.load(output) as archive:
        assert archive["crossfit_next_mean"].shape == (transitions,)


def test_heldout_reward_perturbation_cannot_change_its_predictions(tmp_path):
    original_path = _action_quality_dataset(tmp_path / "original.hdf5", episodes=24)
    config = _ridge_config()
    original = prepare_flat_dataset(
        original_path,
        discount=config.discount,
        noise_seed=config.noise_seed,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    original_arrays, original_metadata = crossfit_predict(original, config)
    heldout_episodes = np.flatnonzero(original_arrays["fold_ids"] == 0)

    h5py = pytest.importorskip("h5py")
    changed_path = tmp_path / "changed.hdf5"
    changed_path.write_bytes(original_path.read_bytes())
    with h5py.File(changed_path, "r+") as handle:
        rewards = np.asarray(handle["rewards"])
        # Use source rows belonging to fold zero.  Features, boundaries and
        # ordering stay fixed; only held-out labels are adversarially changed.
        source_rows = original.source_indices[
            np.isin(original.episode_ids, heldout_episodes)
        ]
        rewards[source_rows] += 10000.0
        handle["rewards"][:] = rewards
    changed = prepare_flat_dataset(
        changed_path,
        discount=config.discount,
        noise_seed=config.noise_seed,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    changed_arrays, changed_metadata = crossfit_predict(changed, config)
    heldout_mask = np.isin(original.episode_ids, heldout_episodes)
    for key in ("v_current", "q_current", "v_next", "crossfit_next_mean"):
        np.testing.assert_allclose(
            original_arrays[key][heldout_mask],
            changed_arrays[key][heldout_mask],
            rtol=0,
            atol=0,
        )
    np.testing.assert_allclose(
        original_arrays["controllable_score"][heldout_episodes],
        changed_arrays["controllable_score"][heldout_episodes],
        rtol=0,
        atol=0,
    )
    assert original_metadata["fold_integrity"]
    assert changed_metadata["fold_integrity"]


def test_q_minus_v_recovers_action_quality_out_of_fold(tmp_path):
    path = _action_quality_dataset(tmp_path / "quality.hdf5")
    config = _ridge_config()
    dataset = prepare_flat_dataset(
        path,
        discount=config.discount,
        noise_seed=config.noise_seed,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
    )
    arrays, _ = crossfit_predict(dataset, config)
    labels = np.where(np.arange(len(dataset.episodes)) % 2 == 0, 1.0, -1.0)
    good = arrays["controllable_score"][labels > 0]
    bad = arrays["controllable_score"][labels < 0]
    assert float(good.mean()) > float(bad.mean()) + 2.0
    assert float(np.corrcoef(arrays["controllable_score"], labels)[0, 1]) > 0.95
    transition_advantage = arrays["q_current"] - arrays["v_current"]
    transition_quality = dataset.actions[:, 0]
    assert float(np.corrcoef(transition_advantage, transition_quality)[0, 1]) > 0.9
