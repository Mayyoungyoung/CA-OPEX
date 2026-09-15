import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_uniform_channel import (
    UnidentifiableChannelError,
    calibrate,
    estimate_beta_mle,
    load_pair_archive,
    parametric_bootstrap,
)
from generate_channel_pairs import generate_pairs, save_pair_archive


@pytest.fixture
def command_hdf5(tmp_path):
    path = tmp_path / "commands.hdf5"
    commands = np.random.default_rng(101).uniform(-1.0, 1.0, size=(6000, 3)).astype(
        np.float32
    )
    with h5py.File(path, "w") as handle:
        handle["clean_policy_actions"] = commands
        # These fields must never be copied into the pair archive.
        handle["observations"] = np.ones((6000, 4), dtype=np.float32)
        handle["rewards"] = np.ones(6000, dtype=np.float32)
        handle["next_observations"] = np.ones((6000, 4), dtype=np.float32)
    return path


def test_pair_generation_is_reproducible_and_archive_is_state_free(
    command_hdf5, tmp_path
):
    first = generate_pairs(
        command_hdf5,
        pair_count=512,
        beta=0.8,
        seed=47,
        action_low=-1.0,
        action_high=1.0,
    )
    second = generate_pairs(
        command_hdf5,
        pair_count=512,
        beta=0.8,
        seed=47,
        action_low=-1.0,
        action_high=1.0,
    )
    other = generate_pairs(
        command_hdf5,
        pair_count=512,
        beta=0.8,
        seed=48,
        action_low=-1.0,
        action_high=1.0,
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert not np.array_equal(first[0], other[0])
    assert not np.array_equal(first[1], other[1])
    assert np.all(first[1] >= -1.0) and np.all(first[1] <= 1.0)

    archive_path = tmp_path / "pairs.npz"
    save_pair_archive(archive_path, *first)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"commands", "executed", "provenance_json"}
        provenance = json.loads(str(archive["provenance_json"].item()))
    assert provenance["contains_state_or_transition_fields"] is False
    assert all(
        forbidden not in provenance
        for forbidden in ("observations", "rewards", "next_observations", "states")
    )
    loaded_commands, loaded_executed, loaded_provenance = load_pair_archive(
        archive_path
    )
    np.testing.assert_array_equal(loaded_commands, first[0])
    np.testing.assert_array_equal(loaded_executed, first[1])
    assert loaded_provenance == first[2]


@pytest.mark.parametrize("beta", [0.05, 0.25, 0.8, 1.5, 3.0])
def test_censored_mle_recovers_multiple_noise_scales(beta):
    rng = np.random.default_rng(7000 + int(beta * 100))
    commands = rng.uniform(-1.0, 1.0, size=(5000, 3)).astype(np.float32)
    executed = np.clip(
        commands + rng.uniform(-beta, beta, size=commands.shape).astype(np.float32),
        -1.0,
        1.0,
    )
    result = estimate_beta_mle(
        commands, executed, action_low=-1.0, action_high=1.0
    )
    assert result["beta_mle"] == pytest.approx(beta, rel=0.025, abs=2e-4)
    assert result["interior_scalar_count"] > 0
    assert result["censored_scalar_count"] > 0
    assert 0.0 < result["censored_fraction"] < 1.0
    assert "point masses" in result["likelihood_reference_measure"]


def test_small_sample_bootstrap_is_finite_and_auditable():
    rng = np.random.default_rng(9)
    commands = rng.uniform(-1.0, 1.0, size=(32, 2)).astype(np.float32)
    beta = 0.6
    executed = np.clip(
        commands + rng.uniform(-beta, beta, size=commands.shape).astype(np.float32),
        -1.0,
        1.0,
    )
    estimate = estimate_beta_mle(
        commands, executed, action_low=-1.0, action_high=1.0
    )
    uncertainty = parametric_bootstrap(
        commands,
        estimate["beta_mle"],
        action_low=-1.0,
        action_high=1.0,
        boundary_tolerance=1e-7,
        replicates=40,
        seed=123,
        confidence_level=0.9,
    )
    assert uncertainty["replicates_finite"] > 0
    lower, upper = uncertainty["beta_interval"]
    assert 0.0 <= lower <= upper
    assert len(uncertainty["beta_estimates_sha256"]) == 64
    assert len(uncertainty["beta_estimates"]) == uncertainty["replicates_finite"]


def test_all_clipped_nontrivial_pairs_report_unbounded_likelihood():
    commands = np.asarray([[0.0], [0.2], [-0.3]], dtype=np.float32)
    executed = np.asarray([[1.0], [1.0], [-1.0]], dtype=np.float32)
    with pytest.raises(UnidentifiableChannelError, match="no finite beta"):
        estimate_beta_mle(
            commands, executed, action_low=-1.0, action_high=1.0
        )


def test_archive_hash_detects_pair_tampering(command_hdf5, tmp_path):
    commands, executed, provenance = generate_pairs(
        command_hdf5,
        pair_count=64,
        beta=0.5,
        seed=3,
        action_low=-1.0,
        action_high=1.0,
    )
    executed = executed.copy()
    executed[0, 0] *= -1.0
    path = tmp_path / "tampered.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            commands=commands,
            executed=executed,
            provenance_json=np.asarray(json.dumps(provenance)),
        )
    with pytest.raises(ValueError, match="executed hash"):
        load_pair_archive(path)


def test_both_clis_support_nonwriting_dry_run(command_hdf5, tmp_path):
    code_dir = Path(__file__).resolve().parents[1]
    pair_path = tmp_path / "cli_pairs.npz"
    absent_generation_output = tmp_path / "generation_dry_run.npz"
    generated = subprocess.run(
        [
            sys.executable,
            str(code_dir / "generate_channel_pairs.py"),
            "--source", str(command_hdf5),
            "--output", str(absent_generation_output),
            "--pairs", "128",
            "--beta", "0.7",
            "--seed", "55",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(generated.stdout)["status"] == "dry_run"
    assert not absent_generation_output.exists()

    pair_payload = generate_pairs(
        command_hdf5,
        pair_count=128,
        beta=0.7,
        seed=55,
        action_low=-1.0,
        action_high=1.0,
    )
    save_pair_archive(pair_path, *pair_payload)
    absent_calibration_output = tmp_path / "calibration_dry_run.json"
    calibrated = subprocess.run(
        [
            sys.executable,
            str(code_dir / "calibrate_uniform_channel.py"),
            "--pairs", str(pair_path),
            "--output", str(absent_calibration_output),
            "--bootstrap-replicates", "20",
            "--bootstrap-seed", "56",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_result = json.loads(calibrated.stdout)
    assert dry_result["status"] == "dry_run"
    assert dry_result["pair_count"] == 128
    assert not absent_calibration_output.exists()


def test_calibration_json_contains_estimate_ci_and_input_hashes(
    command_hdf5, tmp_path
):
    archive_path = tmp_path / "pairs.npz"
    save_pair_archive(
        archive_path,
        *generate_pairs(
            command_hdf5,
            pair_count=256,
            beta=0.9,
            seed=88,
            action_low=-1.0,
            action_high=1.0,
        ),
    )
    result = calibrate(
        archive_path,
        boundary_tolerance=1e-7,
        bootstrap_replicates=25,
        bootstrap_seed=89,
        confidence_level=0.95,
    )
    assert result["estimator_uses_provenance_beta"] is False
    assert result["estimate"]["beta_mle"] == pytest.approx(0.9, rel=0.08)
    assert len(result["pair_archive_sha256"]) == 64
    assert len(result["commands_sha256"]) == 64
    assert len(result["executed_sha256"]) == 64
    assert len(result["uncertainty"]["beta_interval"]) == 2
