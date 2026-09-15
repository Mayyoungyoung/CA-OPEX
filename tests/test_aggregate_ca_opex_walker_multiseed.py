"""Fail-closed tests for the five-training-seed Walker2d aggregator.

The fixtures synthesize minimal-but-valid raw records for the formal block
(39300..), the supplemental block (79300..), and three scale-up run roots, then
exercise the missing-arm, hash-mismatch, duplicate-seed, reserved-block, and
overwrite guards.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import aggregate_ca_opex_walker_multiseed as multiseed


EPISODES = 50
FORMAL_ENV = list(range(39_300, 39_350))
FORMAL_NOISE = list(range(49_300, 49_350))
FORMAL_GRAD = list(range(69_300, 69_350))
SUPP_ENV = list(range(79_300, 79_350))
SUPP_NOISE = list(range(89_300, 89_350))
SUPP_GRAD = list(range(99_300, 99_350))
CALIBRATION_SHA = multiseed.EXPECTED_CALIBRATION_SHA256
CALIBRATED_BETA = multiseed.CALIBRATED_BETA


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def channel_record(
    *,
    control_id: str,
    checkpoint_sha: str,
    env_seeds,
    noise_seeds,
    grad_seeds,
    returns,
    baseline_returns,
    wall_time: float = 60.0,
) -> dict:
    contract = multiseed.CHANNEL_CONTRACTS[control_id]
    calibrated = bool(contract["calibrated"])
    base_returns = baseline_returns
    return {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "environment": multiseed.ENVIRONMENT,
        "method_id": contract["method_id"],
        "base_checkpoint": {"sha256": checkpoint_sha, "step": 25_000},
        "controller": {
            "baseline_transform": contract["baseline_transform"],
            "K": contract["K"],
            "gradient_steps": contract["T"],
            "step_size": contract["step_size"],
            "delta_max": contract["delta_max"],
            "model_beta": CALIBRATED_BETA if calibrated else 0.0,
        },
        "calibration": (
            {
                "calibration_sha256": CALIBRATION_SHA,
                "source": multiseed.CALIBRATION_SOURCE,
                "beta": CALIBRATED_BETA,
            }
            if calibrated
            else {
                "calibration_sha256": None,
                "source": multiseed.NOMINAL_SOURCE,
                "beta": 0.0,
            }
        ),
        "evaluation_protocol": {
            "paired_environment_and_action_noise_seeds": True,
        },
        "arms": {
            "adapted": {
                "environment_seeds": env_seeds,
                "action_noise_seeds": noise_seeds,
                "gradient_noise_seeds": grad_seeds,
                "returns": returns,
                "lengths": [100] * EPISODES,
                "action_noise_beta": multiseed.ROLLOUT_BETA,
            },
            "baseline_only": {
                "environment_seeds": env_seeds,
                "action_noise_seeds": noise_seeds,
                "returns": base_returns,
                "lengths": [100] * EPISODES,
                "action_noise_beta": multiseed.ROLLOUT_BETA,
            },
        },
        "cost": {"wall_time_seconds": wall_time},
    }


def inverse_record(
    *,
    checkpoint_sha: str,
    env_seeds,
    noise_seeds,
    returns,
    wall_time: float = 40.0,
) -> dict:
    return {
        "status": "complete",
        "environment": multiseed.ENVIRONMENT,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 25_000,
        "command_transform": "uniform_mean_inverse",
        "command_transform_beta": multiseed.ROLLOUT_BETA,
        "persistent_action_noise": {
            "environment_seeds": env_seeds,
            "action_noise_seeds": noise_seeds,
            "returns": returns,
            "lengths": [100] * EPISODES,
        },
        "wall_time_seconds": wall_time,
    }


def build_tree(root: Path, seed_blocks: dict[int, tuple[int, int, int]] | None = None):
    """Create the full synthetic results tree and return pinned-hash tables."""

    seed_blocks = seed_blocks or {
        2: (131_300, 231_300, 331_300),
        3: (131_400, 231_400, 331_400),
        4: (131_500, 231_500, 331_500),
    }
    rng = np.random.default_rng(7)
    checkpoint_shas = {
        1: "a" * 64,
        10: "b" * 64,
    }
    formal_hashes = {}
    supplemental_hashes = {}
    for seed in (1, 10):
        checkpoint = checkpoint_shas[seed]
        complete = 1200.0 + 100.0 * (seed % 7) + rng.normal(0, 1, EPISODES)
        identity = complete - 200.0 + rng.normal(0, 1, EPISODES)
        opex = complete - 300.0 + rng.normal(0, 1, EPISODES)
        formal_hashes[("complete", seed)] = write_json(
            root / f"inverse_residual_confirm/external_controls/ca_opex_inverse_seed{seed}_beta125_50.json",
            channel_record(
                control_id="complete",
                checkpoint_sha=checkpoint,
                env_seeds=FORMAL_ENV,
                noise_seeds=FORMAL_NOISE,
                grad_seeds=FORMAL_GRAD,
                returns=complete.tolist(),
                baseline_returns=(complete - 150.0).tolist(),
            ),
        )
        formal_hashes[("calibrated_identity", seed)] = write_json(
            root / f"inverse_residual_confirm/external_controls/ca_opex_identity_seed{seed}_beta125_50.json",
            channel_record(
                control_id="calibrated_identity",
                checkpoint_sha=checkpoint,
                env_seeds=FORMAL_ENV,
                noise_seeds=FORMAL_NOISE,
                grad_seeds=FORMAL_GRAD,
                returns=identity.tolist(),
                baseline_returns=(complete - 400.0).tolist(),
            ),
        )
        formal_hashes[("original_opex", seed)] = write_json(
            root / f"inverse_residual_confirm/external_controls/opex_original_seed{seed}_beta125_50.json",
            channel_record(
                control_id="original_opex",
                checkpoint_sha=checkpoint,
                env_seeds=FORMAL_ENV,
                noise_seeds=FORMAL_NOISE,
                grad_seeds=FORMAL_GRAD,
                returns=opex.tolist(),
                baseline_returns=(complete - 400.0).tolist(),
            ),
        )
        write_json(
            root / f"inverse_residual_confirm/base_hubl_executed_25k_seed{seed}/summary.json",
            {"wall_time_seconds": 250.0 + seed},
        )
        supp_complete = 1250.0 + rng.normal(0, 1, EPISODES)
        nominal = supp_complete - 250.0 + rng.normal(0, 1, EPISODES)
        supplemental_hashes[("complete_supplemental", seed)] = write_json(
            root / f"equal_compute_nominal_control/holdout/complete_calibrated_inverse_eta_0.1_seed{seed}_fresh50.json",
            channel_record(
                control_id="complete",
                checkpoint_sha=checkpoint,
                env_seeds=SUPP_ENV,
                noise_seeds=SUPP_NOISE,
                grad_seeds=SUPP_GRAD,
                returns=supp_complete.tolist(),
                baseline_returns=(supp_complete - 300.0).tolist(),
            ),
        )
        supplemental_hashes[("nominal_k8t2", seed)] = write_json(
            root / f"equal_compute_nominal_control/holdout/nominal_tuned_seed{seed}_fresh50.json",
            channel_record(
                control_id="nominal_k8t2",
                checkpoint_sha=checkpoint,
                env_seeds=SUPP_ENV,
                noise_seeds=SUPP_NOISE,
                grad_seeds=SUPP_GRAD,
                returns=nominal.tolist(),
                baseline_returns=(supp_complete - 500.0).tolist(),
            ),
        )
    for seed, (env_start, noise_start, grad_start) in seed_blocks.items():
        run_root = root / f"scaleup/walker_seed{seed}"
        checkpoint = run_root / "base/latest.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_sha = sha256_file(checkpoint)
        env = list(range(env_start, env_start + EPISODES))
        noise = list(range(noise_start, noise_start + EPISODES))
        grad = list(range(grad_start, grad_start + EPISODES))
        complete = 1300.0 + 10.0 * seed + rng.normal(0, 1, EPISODES)
        control_files = {
            "complete": channel_record(
                control_id="complete",
                checkpoint_sha=checkpoint_sha,
                env_seeds=env,
                noise_seeds=noise,
                grad_seeds=grad,
                returns=complete.tolist(),
                baseline_returns=(complete - 100.0).tolist(),
            ),
            "calibrated_identity": channel_record(
                control_id="calibrated_identity",
                checkpoint_sha=checkpoint_sha,
                env_seeds=env,
                noise_seeds=noise,
                grad_seeds=grad,
                returns=(complete - 180.0).tolist(),
                baseline_returns=(complete - 450.0).tolist(),
            ),
            "nominal_k8t2": channel_record(
                control_id="nominal_k8t2",
                checkpoint_sha=checkpoint_sha,
                env_seeds=env,
                noise_seeds=noise,
                grad_seeds=grad,
                returns=(complete - 260.0).tolist(),
                baseline_returns=(complete - 450.0).tolist(),
            ),
            "original_opex": channel_record(
                control_id="original_opex",
                checkpoint_sha=checkpoint_sha,
                env_seeds=env,
                noise_seeds=noise,
                grad_seeds=grad,
                returns=(complete - 340.0).tolist(),
                baseline_returns=(complete - 450.0).tolist(),
            ),
        }
        inventory = []
        for control_id, payload in control_files.items():
            sha = write_json(run_root / "controls" / f"{control_id}.json", payload)
            inventory.append(
                {"role": f"control:{control_id}", "sha256": sha, "path": ""}
            )
        inverse_sha = write_json(
            run_root / "controls/inverse_only.json",
            inverse_record(
                checkpoint_sha=checkpoint_sha,
                env_seeds=env,
                noise_seeds=noise,
                returns=(complete - 220.0).tolist(),
            ),
        )
        inventory.append(
            {"role": "control:inverse_only", "sha256": inverse_sha, "path": ""}
        )
        write_json(
            run_root / "base/summary.json",
            {"wall_time_seconds": 260.0 + seed},
        )
        write_json(
            run_root / "aggregate.json",
            {
                "schema_version": multiseed.SCALEUP_AGGREGATE_SCHEMA,
                "status": "complete",
                "training_seed": seed,
                "checkpoint": {"sha256": checkpoint_sha},
                "evaluation_protocol": {
                    "environment_seeds": env,
                    "action_noise_seeds": noise,
                    "channel_gradient_noise_seeds": grad,
                },
                "input_files": inventory,
            },
        )
    return formal_hashes, supplemental_hashes, checkpoint_shas


@pytest.fixture()
def patched(tmp_path, monkeypatch):
    root = tmp_path / "results"
    formal_hashes, supplemental_hashes, checkpoint_shas = build_tree(root)
    monkeypatch.setattr(multiseed, "FORMAL_INPUTS", {
        (control, seed): (
            f"inverse_residual_confirm/external_controls/"
            + {
                "complete": f"ca_opex_inverse_seed{seed}_beta125_50.json",
                "calibrated_identity": f"ca_opex_identity_seed{seed}_beta125_50.json",
                "original_opex": f"opex_original_seed{seed}_beta125_50.json",
            }[control],
            sha,
        )
        for (control, seed), sha in formal_hashes.items()
    })
    monkeypatch.setattr(multiseed, "SUPPLEMENTAL_INPUTS", {
        (control, seed): (
            f"equal_compute_nominal_control/holdout/"
            + {
                "complete_supplemental": f"complete_calibrated_inverse_eta_0.1_seed{seed}_fresh50.json",
                "nominal_k8t2": f"nominal_tuned_seed{seed}_fresh50.json",
            }[control],
            sha,
        )
        for (control, seed), sha in supplemental_hashes.items()
    })
    monkeypatch.setattr(
        multiseed,
        "PINNED_BASE_CHECKPOINTS",
        {seed: sha for seed, sha in checkpoint_shas.items() if seed in (1, 10)},
    )
    return root


def test_multiseed_aggregate_completes_and_statistics_match(patched, tmp_path):
    output = tmp_path / "multiseed/aggregate.json"
    report = multiseed.aggregate(patched, output)
    assert report["status"] == "complete"
    assert report["training_seeds"] == [1, 2, 3, 4, 10]
    assert len(report["paired_comparisons"]) == 5
    by_id = {entry["comparison_id"]: entry for entry in report["paired_comparisons"]}
    nominal = by_id["complete_minus_nominal_k8t2"]
    seeds_present = {row["training_seed"] for row in nominal["per_training_seed"]}
    assert seeds_present == {1, 2, 3, 4, 10}
    for row in nominal["per_training_seed"]:
        if row["training_seed"] in (1, 10):
            assert row["left_source_block"] == (
                "post_confirmation_supplemental_block_79300"
            )
    identity = by_id["complete_minus_calibrated_identity"]
    cross = identity["cross_seed_normalized"]
    assert cross["n_training_seeds"] == 5
    assert cross["positive_training_seed_count"] == 5
    assert cross["t95_low"] <= cross["mean"] <= cross["t95_high"]
    controllers = {
        entry["control_id"]: entry for entry in report["per_seed_controller_means"]
    }
    assert set(controllers["complete"]["per_seed"]) == {"1", "2", "3", "4", "10"}
    assert report["wall_times"]["total_base_training_seconds"] > 0
    assert output.exists()


def test_multiseed_missing_control_file_fails(patched, tmp_path, monkeypatch):
    missing = patched / "scaleup/walker_seed3/controls/original_opex.json"
    original = missing.read_text()
    missing.unlink()
    with pytest.raises(multiseed.AggregationError, match="original_opex"):
        multiseed.aggregate(patched, tmp_path / "out1/aggregate.json")
    missing.write_text(original, encoding="utf-8")


def test_multiseed_tampered_formal_hash_fails(patched, tmp_path):
    target = patched / (
        "inverse_residual_confirm/external_controls/ca_opex_inverse_seed1_beta125_50.json"
    )
    payload = json.loads(target.read_text())
    payload["arms"]["adapted"]["returns"][0] += 1.0
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(multiseed.AggregationError, match="SHA mismatch"):
        multiseed.aggregate(patched, tmp_path / "out2/aggregate.json")


def test_multiseed_duplicate_training_seed_fails(patched, tmp_path, monkeypatch):
    monkeypatch.setattr(multiseed, "TRAINING_SEEDS", (1, 2, 3, 3, 10))
    monkeypatch.setattr(multiseed, "SCALEUP_SEEDS", (2, 3, 3))
    with pytest.raises(multiseed.AggregationError, match="duplicate training seeds"):
        multiseed.aggregate(patched, tmp_path / "out3/aggregate.json")


def test_multiseed_reserved_block_reuse_fails(patched, tmp_path):
    seed3_aggregate = patched / "scaleup/walker_seed3/aggregate.json"
    payload = json.loads(seed3_aggregate.read_text())
    payload["evaluation_protocol"]["environment_seeds"] = list(FORMAL_ENV)
    seed3_aggregate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(multiseed.AggregationError, match="reserved seed block"):
        multiseed.aggregate(patched, tmp_path / "out4/aggregate.json")


def test_multiseed_overlapping_scaleup_blocks_fail(patched, tmp_path):
    run_root = patched / "scaleup/walker_seed4"
    for control_path in sorted((run_root / "controls").glob("*.json")):
        control = json.loads(control_path.read_text())
        if "arms" in control:
            for arm in control["arms"].values():
                arm["environment_seeds"] = list(range(131_300, 131_350))
        else:
            control["persistent_action_noise"]["environment_seeds"] = list(
                range(131_300, 131_350)
            )
        control_path.write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")
    seed4_aggregate = run_root / "aggregate.json"
    payload = json.loads(seed4_aggregate.read_text())
    payload["evaluation_protocol"]["environment_seeds"] = list(
        range(131_300, 131_350)
    )
    for entry in payload["input_files"]:
        role = entry["role"]
        control_id = role.split(":", 1)[1] if role.startswith("control:") else None
        if control_id is not None:
            entry["sha256"] = sha256_file(
                run_root / "controls" / f"{control_id}.json"
            )
    seed4_aggregate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(multiseed.AggregationError, match="share environment block"):
        multiseed.aggregate(patched, tmp_path / "out5/aggregate.json")


def test_multiseed_refuses_overwrite(patched, tmp_path):
    output = tmp_path / "multiseed/aggregate.json"
    multiseed.aggregate(patched, output)
    with pytest.raises(multiseed.AggregationError, match="refusing to overwrite"):
        multiseed.aggregate(patched, output)


def test_cross_seed_statistics_match_manual_computation():
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    stats = multiseed._cross_seed_stats(values)
    array = np.asarray(values)
    assert stats["mean"] == pytest.approx(array.mean())
    assert stats["sample_std_ddof1"] == pytest.approx(array.std(ddof=1))
    expected_se = array.std(ddof=1) / math.sqrt(5)
    critical = 2.7764451051977987
    assert stats["t95_low"] == pytest.approx(array.mean() - critical * expected_se)
    assert stats["positive_training_seed_count"] == 5
