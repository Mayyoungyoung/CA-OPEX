"""Fail-closed tests for the Walker2d beta-severity grid aggregator.

Fixtures synthesize two beta rows against three synthetic checkpoints, then
exercise the happy path, missing-arm, calibration-hash, duplicate-unit,
overlapping-seed-block, and overwrite guards.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import aggregate_ca_opex_walker_betagrid as betagrid

EPISODES = 50


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def channel_record(
    *,
    control_id: str,
    beta: float,
    beta_hat: float | None,
    calibration_sha: str | None,
    checkpoint_sha: str,
    env_start: int,
    noise_start: int,
    grad_start: int,
    returns,
) -> dict:
    contract = betagrid.CHANNEL_CONTRACTS[control_id]
    return {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "environment": betagrid.ENVIRONMENT,
        "method_id": contract["method_id"],
        "base_checkpoint": {"sha256": checkpoint_sha, "step": 25_000},
        "controller": {
            "baseline_transform": contract["baseline_transform"],
            "K": contract["K"],
            "gradient_steps": contract["T"],
            "step_size": contract["step_size"],
            "delta_max": contract["delta_max"],
            "model_beta": beta_hat if contract["calibrated"] else 0.0,
        },
        "calibration": (
            {"calibration_sha256": calibration_sha}
            if contract["calibrated"]
            else {"calibration_sha256": None}
        ),
        "channel": {"environment_rollout_beta": beta},
        "arms": {
            "adapted": {
                "environment_seeds": list(range(env_start, env_start + EPISODES)),
                "action_noise_seeds": list(range(noise_start, noise_start + EPISODES)),
                "gradient_noise_seeds": list(range(grad_start, grad_start + EPISODES)),
                "returns": returns,
                "lengths": [100] * EPISODES,
            }
        },
    }


def command_record(
    *,
    arm_id: str,
    beta: float,
    checkpoint_sha: str,
    env_start: int,
    noise_start: int,
    returns,
) -> dict:
    return {
        "status": "complete",
        "environment": betagrid.ENVIRONMENT,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 25_000,
        "command_scale": 1.0,
        "command_transform": betagrid.COMMAND_ARMS[arm_id],
        "command_transform_beta": beta,
        "persistent_action_noise": {
            "action_noise_beta": beta,
            "environment_seeds": list(range(env_start, env_start + EPISODES)),
            "action_noise_seeds": list(range(noise_start, noise_start + EPISODES)),
            "returns": returns,
            "lengths": [100] * EPISODES,
        },
    }


@pytest.fixture()
def grid_root(tmp_path, monkeypatch):
    # Reduce the frozen grid to two beta rows for fixture size; the checkpoint
    # table is replaced with synthetic hashes.
    reduced_units = {
        0.5: (27_011, 28_011, 51_400, 61_400, 71_400),
        1.1: (27_013, 28_013, 51_600, 61_600, 71_600),
    }
    checkpoints = {
        seed: {"path": f"ckpt_seed{seed}.pt", "sha256": f"{seed:064d}"}
        for seed in (1, 2, 10)
    }
    monkeypatch.setattr(betagrid, "BETA_UNITS", reduced_units)
    monkeypatch.setattr(betagrid, "CHECKPOINTS", checkpoints)

    root = tmp_path / "betagrid"
    rng = np.random.default_rng(23)
    for beta, (pairs_seed, _boot, env_start, noise_start, grad_start) in (
        reduced_units.items()
    ):
        beta_hat = beta - 0.00012
        calibration_path = (
            root / f"calibration/beta{beta:g}_n512_seed{pairs_seed}_calibration.json"
        )
        write_json(calibration_path, {"estimate": {"beta_mle": beta_hat}})
        calibration_sha = sha256_file(calibration_path)
        for seed, checkpoint in checkpoints.items():
            unit_dir = root / f"beta{beta:g}_seed{seed}"
            complete = 900.0 + 10.0 * seed + rng.normal(0, 1, EPISODES)
            write_json(
                unit_dir / "controls/complete.json",
                channel_record(
                    control_id="complete",
                    beta=beta,
                    beta_hat=beta_hat,
                    calibration_sha=calibration_sha,
                    checkpoint_sha=checkpoint["sha256"],
                    env_start=env_start,
                    noise_start=noise_start,
                    grad_start=grad_start,
                    returns=complete.tolist(),
                ),
            )
            write_json(
                unit_dir / "controls/nominal_k8t2.json",
                channel_record(
                    control_id="nominal_k8t2",
                    beta=beta,
                    beta_hat=None,
                    calibration_sha=None,
                    checkpoint_sha=checkpoint["sha256"],
                    env_start=env_start,
                    noise_start=noise_start,
                    grad_start=grad_start,
                    returns=(complete - 150.0).tolist(),
                ),
            )
            for arm_id, shift in (("inverse_only", 90.0), ("identity_command", 400.0)):
                write_json(
                    unit_dir / "controls" / f"{arm_id}.json",
                    command_record(
                        arm_id=arm_id,
                        beta=beta,
                        checkpoint_sha=checkpoint["sha256"],
                        env_start=env_start,
                        noise_start=noise_start,
                        returns=(complete - shift).tolist(),
                    ),
                )
    return root


def test_betagrid_aggregate_completes(grid_root, tmp_path):
    output = tmp_path / "out/aggregate.json"
    report = betagrid.aggregate(grid_root, output)
    assert report["status"] == "complete"
    assert report["betas"] == [0.5, 1.1]
    per_beta = report["per_beta"]
    assert set(per_beta) == {"0.5", "1.1"}
    for entry in per_beta.values():
        assert set(entry["per_checkpoint"]) == {"1", "2", "10"}
        comparison = entry["comparison_cross_checkpoint"]["complete_minus_nominal_k8t2"]
        assert comparison["n_checkpoints"] == 3
        assert comparison["positive_checkpoint_count"] == 3
        factor = 100.0 / (betagrid.REFERENCE_MAX - betagrid.REFERENCE_MIN)
        assert comparison["mean"] == pytest.approx(150.0 * factor, abs=1e-9)
        per_seed = entry["per_checkpoint"]["1"]
        assert per_seed["complete_minus_inverse_only"]["mean"] == pytest.approx(
            90.0 * factor, abs=1e-9
        )
    assert output.exists()


def test_betagrid_missing_arm_fails(grid_root, tmp_path):
    (grid_root / "beta0.5_seed2/controls/inverse_only.json").unlink()
    with pytest.raises(betagrid.AggregationError, match="inverse_only"):
        betagrid.aggregate(grid_root, tmp_path / "out2/aggregate.json")


def test_betagrid_calibration_hash_mismatch_fails(grid_root, tmp_path):
    calibration = grid_root / "calibration/beta1.1_n512_seed27013_calibration.json"
    payload = json.loads(calibration.read_text())
    payload["estimate"]["beta_mle"] = payload["estimate"]["beta_mle"] - 0.5
    calibration.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(betagrid.AggregationError, match="calibration SHA mismatch"):
        betagrid.aggregate(grid_root, tmp_path / "out3/aggregate.json")


def test_betagrid_duplicate_unit_directory_fails(grid_root, tmp_path):
    (grid_root / "beta0.5_seed2_duplicate").mkdir()
    with pytest.raises(betagrid.AggregationError, match="duplicate seed"):
        betagrid.aggregate(grid_root, tmp_path / "out_duplicate/aggregate.json")


def test_betagrid_overlapping_seed_block_fails(grid_root, tmp_path):
    path = grid_root / "beta0.5_seed2/controls/complete.json"
    payload = json.loads(path.read_text())
    payload["arms"]["adapted"]["environment_seeds"] = list(range(51_600, 51_650))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(betagrid.AggregationError, match="does not match block"):
        betagrid.aggregate(grid_root, tmp_path / "out_overlap/aggregate.json")


def test_betagrid_refuses_overwrite(grid_root, tmp_path):
    output = tmp_path / "out/aggregate.json"
    betagrid.aggregate(grid_root, output)
    with pytest.raises(betagrid.AggregationError, match="refusing to overwrite"):
        betagrid.aggregate(grid_root, output)
