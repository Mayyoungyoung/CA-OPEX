"""Fail-closed tests for the Hopper-v4 cross-task CA-OPEX aggregator.

Fixtures synthesize one valid per-seed run root (base config/checkpoint plus
five control records) against the transformed Hopper constants, then exercise
the missing-arm, contract-mismatch, reserved-block, and overwrite guards.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import aggregate_ca_opex_hopper_crosstask as aggregate

EPISODES = 50
ENV_BLOCK = list(range(41_300, 41_350))
NOISE_BLOCK = list(range(51_300, 51_350))
GRAD_BLOCK = list(range(61_300, 61_350))
CALIBRATION_SHA = "c" * 64
DRIVER = CODE_DIR / "run_ca_opex_hopper_crosstask.sh"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_calibration_sha(monkeypatch, value: str = CALIBRATION_SHA):
    monkeypatch.setattr(aggregate, "EXPECTED_CALIBRATION_SHA256", value)
    return value


def live_implementation_map() -> dict:
    return {
        raw_key: sha256_file(CODE_DIR / filename)
        for raw_key, filename in aggregate.IMPLEMENTATION_FILES.items()
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def base_config(training_seed: int) -> dict:
    return {
        "arguments": {
            "train_seed": training_seed,
            "variant": "hubl_constant",
            "action_pairing": "executed_executed",
            "updates": 25_000,
            "env_name": aggregate.ENVIRONMENT,
            "heuristic_discount": 0.391843318939209,
        },
        "action_channels": {"action_pairing": "executed_executed"},
    }


def channel_raw(
    *,
    control_id: str,
    checkpoint_sha: str,
    training_seed: int,
    returns,
    calibration_sha: str = CALIBRATION_SHA,
    env_block=None,
    noise_block=None,
    grad_block=None,
) -> dict:
    contract = aggregate.CHANNEL_CONTRACTS[control_id]
    calibrated = bool(contract["calibrated"])
    env_block = env_block or ENV_BLOCK
    noise_block = noise_block or NOISE_BLOCK
    grad_block = grad_block or GRAD_BLOCK
    model_beta = aggregate.CALIBRATED_BETA if calibrated else 0.0
    return {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "environment": aggregate.ENVIRONMENT,
        "method_id": contract["method_id"],
        "training_seed": training_seed,
        "base_checkpoint": {"sha256": checkpoint_sha, "step": 25_000},
        "controller": {
            "baseline_transform": contract["baseline_transform"],
            "K": contract["K"],
            "execution_noise_samples": contract["K"],
            "gradient_steps": contract["T"],
            "gradient_steps_per_action": contract["T"],
            "q_reducer": "mean_q1",
            "critic": "frozen_q1",
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "step_size": contract["step_size"],
            "delta_max": contract["delta_max"],
            "model_beta": model_beta,
            "calibration_mode": (
                aggregate.CALIBRATION_SOURCE
                if calibrated
                else aggregate.NOMINAL_SOURCE
            ),
        },
        "calibration": (
            {
                "calibration_sha256": calibration_sha,
                "calibration_path": getattr(
                    aggregate, "CALIBRATION_PATH_FIXTURE", "/tmp/synthetic_calibration.json"
                ),
                "source": aggregate.CALIBRATION_SOURCE,
                "pair_count": 512,
                "beta": aggregate.CALIBRATED_BETA,
            }
            if calibrated
            else {
                "calibration_sha256": None,
                "source": aggregate.NOMINAL_SOURCE,
                "beta": 0.0,
            }
        ),
        "normalization": {
            "reference_min_score": aggregate.REFERENCE_MIN,
            "reference_max_score": aggregate.REFERENCE_MAX,
        },
        "channel": {
            "gradient_model": "iid_uniform_additive_then_clip",
            "environment_rollout_beta": aggregate.ROLLOUT_BETA,
            "model_or_calibration_beta": model_beta,
        },
        "evaluation_protocol": {
            "environment": aggregate.ENVIRONMENT,
            "episode_count_per_arm": EPISODES,
            "paired_environment_and_action_noise_seeds": True,
            "environment_seed_start": env_block[0],
            "action_noise_seed_start": noise_block[0],
            "gradient_noise_seed_start": grad_block[0],
        },
        "implementation": live_implementation_map(),
        "arms": {
            "adapted": {
                "environment_seeds": env_block,
                "action_noise_seeds": noise_block,
                "gradient_noise_seeds": grad_block,
                "returns": returns,
                "lengths": [100] * EPISODES,
                "action_noise_beta": aggregate.ROLLOUT_BETA,
                "q1_rows_per_action": contract["K"] * contract["T"],
                "q1_rows_total": 100 * EPISODES * contract["K"] * contract["T"],
                "q1_gradient_calls": 100 * EPISODES * contract["T"],
                "q1_rows_by_episode": [
                    100 * contract["K"] * contract["T"]
                ]
                * EPISODES,
            },
            "baseline_only": {
                "environment_seeds": env_block,
                "action_noise_seeds": noise_block,
                "returns": [value - 500.0 for value in returns],
                "lengths": [100] * EPISODES,
                "action_noise_beta": aggregate.ROLLOUT_BETA,
            },
        },
        "cost": {
            "environment_steps": 100 * EPISODES,
            "base_actor_rows": 100 * EPISODES,
            "q1_forward_rows": 100 * EPISODES * contract["K"] * contract["T"],
            "q1_backward_rows": 100 * EPISODES * contract["K"] * contract["T"],
            "q1_backward_calls": 100 * EPISODES * contract["T"],
            "adapted_q1_rows_per_environment_step": contract["K"] * contract["T"],
            "adapted_backward_calls": 100 * EPISODES * contract["T"],
            "wall_time_seconds": 60.0,
        },
    }


def inverse_raw(*, checkpoint_sha: str, training_seed: int, returns) -> dict:
    return {
        "status": "complete",
        "environment": aggregate.ENVIRONMENT,
        "training_seed": training_seed,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 25_000,
        "command_scale": 1.0,
        "command_transform": "uniform_mean_inverse",
        "command_transform_beta": aggregate.ROLLOUT_BETA,
        "persistent_action_noise": {
            "action_noise_distribution": "iid_uniform_minus_beta_plus_beta_per_step",
            "command_transform": "uniform_mean_inverse",
            "command_scale": 1.0,
            "command_transform_beta": aggregate.ROLLOUT_BETA,
            "action_noise_beta": aggregate.ROLLOUT_BETA,
            "environment_seeds": ENV_BLOCK,
            "action_noise_seeds": NOISE_BLOCK,
            "returns": returns,
            "lengths": [100] * EPISODES,
        },
        "wall_time_seconds": 45.0,
    }


@pytest.fixture()
def run_root(tmp_path, monkeypatch):
    calibration = tmp_path / "synthetic_calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    calibration_sha = sha256_file(calibration)
    monkeypatch.setattr(aggregate, "EXPECTED_CALIBRATION_SHA256", calibration_sha)
    aggregate.CALIBRATION_SHA_FIXTURE = calibration_sha
    aggregate.CALIBRATION_PATH_FIXTURE = str(calibration)

    root = tmp_path / "hopper_seed2"
    checkpoint = root / "base/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    checkpoint_sha = sha256_file(checkpoint)
    write_json(root / "base/config.json", base_config(2))

    rng = np.random.default_rng(11)
    complete = 900.0 + rng.normal(0, 1, EPISODES)
    controls = {
        "complete": channel_raw(
            control_id="complete",
            checkpoint_sha=checkpoint_sha,
            training_seed=2,
            returns=complete.tolist(),
            calibration_sha=calibration_sha,
        ),
        "calibrated_identity": channel_raw(
            control_id="calibrated_identity",
            checkpoint_sha=checkpoint_sha,
            training_seed=2,
            returns=(complete - 120.0).tolist(),
            calibration_sha=calibration_sha,
        ),
        "nominal_k8t2": channel_raw(
            control_id="nominal_k8t2",
            checkpoint_sha=checkpoint_sha,
            training_seed=2,
            returns=(complete - 260.0).tolist(),
        ),
        "original_opex": channel_raw(
            control_id="original_opex",
            checkpoint_sha=checkpoint_sha,
            training_seed=2,
            returns=(complete - 300.0).tolist(),
        ),
    }
    for control_id, payload in controls.items():
        write_json(root / "controls" / f"{control_id}.json", payload)
    write_json(
        root / "controls/inverse_only.json",
        inverse_raw(
            checkpoint_sha=checkpoint_sha,
            training_seed=2,
            returns=(complete - 180.0).tolist(),
        ),
    )
    return root


def test_hopper_aggregate_completes(run_root, tmp_path):
    output = tmp_path / "out/aggregate.json"
    report = aggregate.aggregate(run_root, output)
    assert report["status"] == "complete"
    assert report["environment"] == "Hopper-v4"
    assert report["training_seed"] == 2
    assert report["schema_version"] == "ca-opex-hopper-crosstask-aggregate-v1"
    by_id = {row["comparison_id"]: row for row in report["paired_comparisons"]}
    assert set(by_id) == {
        "complete_minus_calibrated_identity",
        "complete_minus_nominal_k8t2",
        "complete_minus_original_opex",
        "complete_minus_inverse_only",
    }
    delta = by_id["complete_minus_nominal_k8t2"]
    factor = 100.0 / (aggregate.REFERENCE_MAX - aggregate.REFERENCE_MIN)
    expected = 260.0 * factor
    assert delta["mean"] == pytest.approx(expected, abs=1e-9)


def test_hopper_missing_arm_fails(run_root, tmp_path):
    (run_root / "controls/nominal_k8t2.json").unlink()
    with pytest.raises(aggregate.AggregationError, match="nominal_k8t2"):
        aggregate.aggregate(run_root, tmp_path / "out2/aggregate.json")


def test_hopper_reserved_walker_block_rejected(run_root, tmp_path):
    """A Hopper block reusing the Walker seed-3 scale-up range must be rejected."""
    walker_block = list(range(131_400, 131_450))
    for control_path in sorted((run_root / "controls").glob("*.json")):
        payload = json.loads(control_path.read_text())
        arms = (
            payload["arms"]
            if "arms" in payload
            else {"persistent": payload["persistent_action_noise"]}
        )
        for arm in arms.values():
            if "environment_seeds" in arm:
                arm["environment_seeds"] = walker_block
        if "evaluation_protocol" in payload:
            payload["evaluation_protocol"]["environment_seed_start"] = walker_block[0]
        write_json(control_path, payload)
    with pytest.raises(aggregate.AggregationError, match="overlaps reserved"):
        aggregate.aggregate(run_root, tmp_path / "out3/aggregate.json")


def test_hopper_dev_training_seed_rejected(run_root, tmp_path):
    write_json(run_root / "base/config.json", base_config(0))
    with pytest.raises(aggregate.AggregationError, match="reserved for development"):
        aggregate.aggregate(run_root, tmp_path / "out4/aggregate.json")
    write_json(run_root / "base/config.json", base_config(2))


def test_hopper_refuses_overwrite(run_root, tmp_path):
    output = tmp_path / "out/aggregate.json"
    aggregate.aggregate(run_root, output)
    with pytest.raises(aggregate.AggregationError, match="refusing to overwrite"):
        aggregate.aggregate(run_root, output)


def test_hopper_driver_has_no_unfilled_placeholders():
    if not DRIVER.is_file():
        pytest.skip("driver not present")
    text = DRIVER.read_text(encoding="utf-8")
    assert "__HOPPER_DATASET_SHA256__" not in text
    assert "__HOPPER_CALIBRATION_SHA256__" not in text


def test_hopper_driver_passes_bash_syntax_check():
    if not DRIVER.is_file():
        pytest.skip("driver not present")
    probe = subprocess.run(
        ["bash", "-c", "true"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("no working local bash (e.g. broken WSL stub)")
    result = subprocess.run(
        ["bash", "-n", str(DRIVER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
