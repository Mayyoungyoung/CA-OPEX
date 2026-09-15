import hashlib
import json
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import aggregate_ca_opex_walker_scaleup as aggregate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def live_implementation_map():
    return {
        raw_key: sha256(CODE_DIR / filename)
        for raw_key, filename in aggregate.IMPLEMENTATION_FILES.items()
    }


def channel_raw(
    *,
    control_id: str,
    checkpoint_sha: str,
    checkpoint_path: str,
    training_seed: int,
    returns,
    env_seeds,
    noise_seeds,
    gradient_seeds,
    calibration_path: Path,
    calibration_sha: str,
):
    spec = aggregate.CHANNEL_CONTRACTS[control_id]
    calibrated = spec["calibrated"]
    beta = aggregate.CALIBRATED_BETA if calibrated else 0.0
    lengths = [2 + (index % 3) for index in range(aggregate.EPISODES)]
    environment_steps = sum(lengths)
    rows_per_action = spec["K"] * spec["T"]
    q_rows = environment_steps * rows_per_action
    backward_calls = environment_steps * spec["T"]
    return {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "environment": aggregate.ENVIRONMENT,
        "method_id": spec["method_id"],
        "training_seed": training_seed,
        "base_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "step": 25_000,
        },
        "normalization": {
            "reference_min_score": aggregate.REFERENCE_MIN,
            "reference_max_score": aggregate.REFERENCE_MAX,
        },
        "calibration": (
            {
                "source": aggregate.CALIBRATION_SOURCE,
                "calibration_path": str(calibration_path.resolve()),
                "calibration_sha256": calibration_sha,
                "pair_count": 512,
                "beta": aggregate.CALIBRATED_BETA,
            }
            if calibrated
            else {
                "source": aggregate.NOMINAL_SOURCE,
                "calibration_sha256": None,
                "pair_count": None,
                "beta": 0.0,
            }
        ),
        "implementation": live_implementation_map(),
        "channel": {
            "gradient_model": "iid_uniform_additive_then_clip",
            "environment_rollout_beta": aggregate.ROLLOUT_BETA,
            "model_or_calibration_beta": beta,
        },
        "controller": {
            "baseline_transform": spec["baseline_transform"],
            "K": spec["K"],
            "execution_noise_samples": spec["K"],
            "gradient_steps": spec["T"],
            "gradient_steps_per_action": spec["T"],
            "step_size": spec["step_size"],
            "delta_max": spec["delta_max"],
            "model_beta": beta,
            "calibration_mode": (
                aggregate.CALIBRATION_SOURCE if calibrated else aggregate.NOMINAL_SOURCE
            ),
            "q_reducer": "mean_q1",
            "critic": "frozen_q1",
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
        },
        "evaluation_protocol": {
            "environment": aggregate.ENVIRONMENT,
            "episode_count_per_arm": aggregate.EPISODES,
            "paired_environment_and_action_noise_seeds": True,
            "environment_seed_start": env_seeds[0],
            "action_noise_seed_start": noise_seeds[0],
            "gradient_noise_seed_start": gradient_seeds[0],
        },
        "arms": {
            "adapted": {
                "environment_seeds": env_seeds,
                "action_noise_seeds": noise_seeds,
                "gradient_noise_seeds": gradient_seeds,
                "action_noise_beta": aggregate.ROLLOUT_BETA,
                "returns": list(returns),
                "lengths": lengths,
                # Intentionally false: aggregation must derive scores from returns.
                "return_mean": -999_999.0,
                "normalized_score_mean": -999_999.0,
                "q1_rows_per_action": rows_per_action,
                "q1_rows_by_episode": [length * rows_per_action for length in lengths],
                "q1_rows_total": q_rows,
                "q1_gradient_calls": backward_calls,
            }
        },
        "cost": {
            "environment_steps": environment_steps,
            "base_actor_rows": environment_steps,
            "q1_forward_rows": q_rows,
            "q1_backward_rows": q_rows,
            "q1_backward_calls": backward_calls,
            "adapted_q1_rows_per_environment_step": rows_per_action,
            "adapted_backward_calls": backward_calls,
            "wall_time_seconds": 1.25,
        },
    }


def inverse_raw(
    *, checkpoint_sha, checkpoint_path, training_seed, returns, env_seeds, noise_seeds
):
    lengths = [3 + (index % 2) for index in range(aggregate.EPISODES)]
    return {
        "status": "complete",
        "environment": aggregate.ENVIRONMENT,
        "training_seed": training_seed,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": 25_000,
        "command_scale": 1.0,
        "command_transform": "uniform_mean_inverse",
        "command_transform_beta": aggregate.ROLLOUT_BETA,
        "persistent_action_noise": {
            "action_noise_distribution": "iid_uniform_minus_beta_plus_beta_per_step",
            "action_noise_beta": aggregate.ROLLOUT_BETA,
            "command_scale": 1.0,
            "command_transform": "uniform_mean_inverse",
            "command_transform_beta": aggregate.ROLLOUT_BETA,
            "environment_seeds": env_seeds,
            "action_noise_seeds": noise_seeds,
            "returns": list(returns),
            "lengths": lengths,
            "return_mean": -999_999.0,
            "normalized_score_mean": -999_999.0,
        },
        "wall_time_seconds": 2.5,
    }


def make_run(
    tmp_path: Path,
    monkeypatch,
    *,
    training_seed: int = 7,
    environment_seed_start: int = 101_000,
    action_noise_seed_start: int = 201_000,
    gradient_noise_seed_start: int = 301_000,
) -> Path:
    run_root = tmp_path / "walker_seed7"
    checkpoint = run_root / "base/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic-frozen-checkpoint")
    checkpoint_sha = sha256(checkpoint)
    config = {
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
    write_json(run_root / "base/config.json", config)
    calibration_path = run_root / "inputs/channel_calibration.json"
    write_json(calibration_path, {"synthetic_calibration": True})
    calibration_sha = sha256(calibration_path)
    monkeypatch.setattr(
        aggregate,
        "EXPECTED_CALIBRATION_SHA256",
        calibration_sha,
    )
    env_seeds = list(
        range(environment_seed_start, environment_seed_start + aggregate.EPISODES)
    )
    noise_seeds = list(
        range(action_noise_seed_start, action_noise_seed_start + aggregate.EPISODES)
    )
    gradient_seeds = list(
        range(gradient_noise_seed_start, gradient_noise_seed_start + aggregate.EPISODES)
    )
    offsets = {
        "complete": 40.0,
        "calibrated_identity": 30.0,
        "nominal_k8t2": 20.0,
        "original_opex": 10.0,
    }
    for control_id, offset in offsets.items():
        returns = [offset + index for index in range(aggregate.EPISODES)]
        payload = channel_raw(
            control_id=control_id,
            checkpoint_sha=checkpoint_sha,
            checkpoint_path=str(checkpoint.resolve()),
            training_seed=training_seed,
            returns=returns,
            env_seeds=env_seeds,
            noise_seeds=noise_seeds,
            gradient_seeds=gradient_seeds,
            calibration_path=calibration_path,
            calibration_sha=calibration_sha,
        )
        write_json(run_root / aggregate.CONTROL_PATHS[control_id], payload)
    write_json(
        run_root / aggregate.CONTROL_PATHS["inverse_only"],
        inverse_raw(
            checkpoint_sha=checkpoint_sha,
            checkpoint_path=str(checkpoint.resolve()),
            training_seed=training_seed,
            returns=[5.0 + index for index in range(aggregate.EPISODES)],
            env_seeds=env_seeds,
            noise_seeds=noise_seeds,
        ),
    )
    return run_root


def test_frozen_calibration_sha_contract_is_exact():
    assert aggregate.EXPECTED_CALIBRATION_SHA256 == (
        "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354"
    )


def test_success_recomputes_scores_statistics_costs_and_hashes(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    output = tmp_path / "aggregate.json"
    report = aggregate.aggregate(run_root, output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["status"] == "complete"
    assert report["training_seed"] == 7
    assert report["normalization"]["cached_raw_means_used"] is False
    scores = {item["control_id"]: item for item in report["absolute_scores"]}
    assert scores["complete"]["raw_score_mean_recomputed"] == pytest.approx(64.5)
    assert scores["complete"]["raw_score_mean_recomputed"] != -999_999.0
    comparisons = {item["comparison_id"]: item for item in report["paired_comparisons"]}
    expected_normalized = (
        10.0 * 100.0 / (aggregate.REFERENCE_MAX - aggregate.REFERENCE_MIN)
    )
    assert comparisons["complete_minus_calibrated_identity"]["mean"] == pytest.approx(
        expected_normalized
    )
    assert (
        comparisons["complete_minus_calibrated_identity"]["positive_episode_count"]
        == 50
    )
    assert (
        comparisons["complete_minus_calibrated_identity"]["bootstrap_replicates"]
        == 20_000
    )
    costs = {item["control_id"]: item for item in report["per_decision_and_total_cost"]}
    assert costs["complete"]["q1_forward_rows_per_decision"] == 16
    assert costs["complete"]["q1_backward_calls_per_decision"] == 2
    assert costs["original_opex"]["q1_forward_rows_per_decision"] == 1
    assert costs["inverse_only"]["q1_forward_rows_per_decision"] == 0
    roles = {item["role"] for item in report["input_files"]}
    assert "channel_calibration" in roles
    assert "scaleup_driver" in roles
    assert {
        f"implementation:{raw_key}" for raw_key in aggregate.IMPLEMENTATION_FILES
    }.issubset(roles)
    for item in report["input_files"]:
        assert sha256(Path(item["path"])) == item["sha256"]


def test_refuses_to_overwrite_output_without_touching_it(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    output = tmp_path / "aggregate.json"
    output.write_text("sentinel", encoding="utf-8")
    with pytest.raises(aggregate.AggregationError, match="refusing to overwrite"):
        aggregate.aggregate(run_root, output)
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_rejects_elementwise_rollout_seed_mismatch(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["inverse_only"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["persistent_action_noise"]["action_noise_seeds"][17] += 10
    write_json(path, payload)
    with pytest.raises(
        aggregate.AggregationError, match="consecutive seed block|differ elementwise"
    ):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_controller_contract_mismatch(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["original_opex"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["controller"]["K"] = 8
    write_json(path, payload)
    with pytest.raises(
        aggregate.AggregationError, match=r"controller.K contract mismatch"
    ):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_checkpoint_hash_mismatch(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    (run_root / "base/latest.pt").write_bytes(b"different-checkpoint")
    with pytest.raises(aggregate.AggregationError, match="checkpoint SHA"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_training_seed_mismatch(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["calibrated_identity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training_seed"] = 999
    write_json(path, payload)
    with pytest.raises(
        aggregate.AggregationError, match="does not match base/config.json"
    ):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


@pytest.mark.parametrize("training_seed", [0, 1, 10])
def test_rejects_reserved_training_seeds(tmp_path, monkeypatch, training_seed):
    run_root = make_run(tmp_path, monkeypatch, training_seed=training_seed)
    with pytest.raises(
        aggregate.AggregationError, match="training seed .* is reserved"
    ):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


@pytest.mark.parametrize(
    ("seed_arguments", "namespace"),
    [
        ({"environment_seed_start": 39_251}, "environment"),
        ({"environment_seed_start": 79_349}, "environment"),
        ({"action_noise_seed_start": 49_251}, "action_noise"),
        ({"action_noise_seed_start": 89_349}, "action_noise"),
        ({"gradient_noise_seed_start": 69_251}, "gradient_noise"),
        ({"gradient_noise_seed_start": 99_349}, "gradient_noise"),
    ],
)
def test_rejects_any_overlap_with_reserved_seed_blocks(
    tmp_path, monkeypatch, seed_arguments, namespace
):
    run_root = make_run(tmp_path, monkeypatch, **seed_arguments)
    with pytest.raises(
        aggregate.AggregationError,
        match=rf"{namespace} 50-case block overlaps reserved",
    ):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_raw_calibration_sha_not_frozen(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["complete"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["calibration_sha256"] = "b" * 64
    write_json(path, payload)
    with pytest.raises(aggregate.AggregationError, match="frozen calibration"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_calibration_path_with_changed_file(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    calibration_path = run_root / "inputs/channel_calibration.json"
    calibration_path.write_bytes(b"changed calibration bytes")
    with pytest.raises(aggregate.AggregationError, match="calibration file"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_missing_calibration_path(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["calibrated_identity"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration"]["calibration_path"] = str(tmp_path / "missing.json")
    write_json(path, payload)
    with pytest.raises(aggregate.AggregationError, match="not an existing file"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_different_channel_implementation_maps(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    path = run_root / aggregate.CONTROL_PATHS["nominal_k8t2"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["implementation"]["evaluate_sha256"] = "b" * 64
    write_json(path, payload)
    with pytest.raises(aggregate.AggregationError, match="maps are not identical"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")


def test_rejects_identical_but_stale_implementation_maps(tmp_path, monkeypatch):
    run_root = make_run(tmp_path, monkeypatch)
    for control_id in aggregate.CHANNEL_CONTRACTS:
        path = run_root / aggregate.CONTROL_PATHS[control_id]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["implementation"]["evaluate_sha256"] = "b" * 64
        write_json(path, payload)
    with pytest.raises(aggregate.AggregationError, match="live code hashes"):
        aggregate.aggregate(run_root, tmp_path / "aggregate.json")
