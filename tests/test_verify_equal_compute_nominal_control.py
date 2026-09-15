import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import verify_equal_compute_nominal_control as verifier


PROTOCOL_PATH = CODE_DIR / "equal_compute_nominal_control_protocol.json"
RUNNER_PATH = CODE_DIR / "run_equal_compute_nominal_control.py"
FORMAL_MANIFEST_PATH = CODE_DIR / "inverse_residual_frozen_confirmation_manifest.json"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def compact_spec(*, calibrated=False):
    payload = protocol()
    checkpoint = payload["holdout"]["base_checkpoints"][0]
    controller_id = (
        "calibrated_identity_eta_0.3" if calibrated else "nominal_tuned"
    )
    spec = verifier.make_spec(
        payload,
        stage="holdout",
        controller_id=controller_id,
        checkpoint=checkpoint,
        eta=0.3 if calibrated else 0.1,
    )
    spec.update(
        episodes=3,
        environment_seed=101,
        action_noise_seed=201,
        gradient_noise_seed=301,
    )
    return payload, spec


def raw_arm(payload, spec, *, adapted, returns, lengths, wall_time):
    returns_array = np.asarray(returns, dtype=np.float64)
    reference_min = payload["environment"]["reference_min_score"]
    reference_max = payload["environment"]["reference_max_score"]
    normalized = 100.0 * (returns_array - reference_min) / (
        reference_max - reference_min
    )
    steps = sum(lengths)
    rows_per_action = spec["K"] * spec["T"] if adapted else 0
    q_rows = [length * rows_per_action for length in lengths]
    backward_calls = steps * spec["T"] if adapted else 0
    calibrated_gradient = adapted and spec["calibrated"]
    return {
        "arm": (
            "channel_opex"
            if adapted
            else f"{spec['baseline_transform']}_anchor_baseline"
        ),
        "environment_seeds": list(
            range(spec["environment_seed"], spec["environment_seed"] + 3)
        ),
        "action_noise_seeds": list(
            range(spec["action_noise_seed"], spec["action_noise_seed"] + 3)
        ),
        "gradient_noise_seeds": list(
            range(spec["gradient_noise_seed"], spec["gradient_noise_seed"] + 3)
        ),
        "gradient_noise_stream_used": calibrated_gradient,
        "gradient_noise_sampling_frequency": (
            "per_gradient_step"
            if calibrated_gradient
            else "none_beta_zero_or_baseline_arm"
        ),
        "gradient_noise_draw_calls": backward_calls if calibrated_gradient else 0,
        "action_noise_beta": payload["environment"]["rollout_action_noise_beta"],
        "returns": returns_array.tolist(),
        "lengths": lengths,
        "return_mean": float(returns_array.mean()),
        "return_std": float(returns_array.std(ddof=0)),
        "normalized_score_mean": float(normalized.mean()),
        "normalized_score_std": float(normalized.std(ddof=0)),
        "q1_rows_per_action": rows_per_action,
        "q1_rows_by_episode": q_rows,
        "q1_rows_total": sum(q_rows),
        "q1_gradient_calls": backward_calls,
        "wall_time_seconds": wall_time,
    }


def raw_record(*, calibrated=False):
    payload, spec = compact_spec(calibrated=calibrated)
    baseline = raw_arm(
        payload,
        spec,
        adapted=False,
        returns=[10.0, 20.0, 30.0],
        lengths=[2, 3, 4],
        wall_time=1.25,
    )
    adapted = raw_arm(
        payload,
        spec,
        adapted=True,
        returns=[13.0, 22.0, 35.0],
        lengths=[3, 3, 5],
        wall_time=2.5,
    )
    differences = (
        np.asarray(adapted["returns"], dtype=np.float64)
        - np.asarray(baseline["returns"], dtype=np.float64)
    )
    paired_wall = 4.0
    calibration = (
        {
            "source": "censored_uniform_plus_clip_pair_calibration",
            "calibration_path": payload["channel_calibration"]["path"],
            "calibration_sha256": payload["channel_calibration"]["sha256"],
            "pair_count": payload["channel_calibration"]["pair_count"],
            "beta": payload["channel_calibration"]["estimated_beta"],
        }
        if calibrated
        else {
            "source": "cli_known_beta_without_pair_calibration",
            "beta": 0.0,
            "pair_count": None,
            "calibration_path": None,
            "calibration_sha256": None,
        }
    )
    cost = {
        "cost_scope": "adapted_arm_deployment_controller_only",
        "environment_steps": 11,
        "q1_forward_rows": 176,
        "q1_backward_rows": 176,
        "q1_backward_calls": 22,
        "base_actor_rows": 11,
        "baseline_environment_steps": 9,
        "baseline_base_actor_rows": 9,
        "paired_evaluation_environment_steps": 20,
        "baseline_q1_rows_total": 0,
        "adapted_q1_rows_total": 176,
        "adapted_q1_rows_per_environment_step": 16,
        "adapted_backward_calls": 22,
        "wall_time_seconds": 2.5,
        "baseline_wall_time_seconds": 1.25,
        "paired_evaluation_wall_time_seconds": paired_wall,
    }
    raw = {
        "raw_schema": verifier.RAW_SCHEMA,
        "status": "complete",
        "method_id": f"channel_aware_opex_{spec['baseline_transform']}_anchor",
        "method_scope": (
            "stronger channel-aware extension of OPEX; not an unchanged "
            "reproduction of the original paper"
        ),
        "environment": payload["environment"]["name"],
        "base_checkpoint": {
            "path": spec["checkpoint"]["path"],
            "sha256": spec["checkpoint"]["sha256"],
        },
        "calibration": calibration,
        "channel": {
            "gradient_model": "iid_uniform_additive_then_clip",
            "model_or_calibration_beta": spec["model_beta"],
            "environment_rollout_beta": payload["environment"][
                "rollout_action_noise_beta"
            ],
            "beta_mismatch": spec["model_beta"]
            != payload["environment"]["rollout_action_noise_beta"],
        },
        "controller": verifier.expected_controller(spec),
        "evaluation_protocol": verifier.expected_evaluation_protocol(payload, spec),
        "normalization": {
            "reference_min_score": payload["environment"]["reference_min_score"],
            "reference_max_score": payload["environment"]["reference_max_score"],
        },
        "arms": {"baseline_only": baseline, "adapted": adapted},
        "paired": {
            "adapted_minus_baseline_returns": differences.tolist(),
            "return_difference_mean": float(differences.mean()),
            "return_difference_std": float(differences.std(ddof=0)),
            "positive_episode_count": int((differences > 0).sum()),
            "episode_count": 3,
        },
        "cost": cost,
        "implementation": {
            verifier.IMPLEMENTATION_KEY_MAP[filename]: digest
            for filename, digest in payload["implementation"].items()
        },
        "wall_time_seconds": paired_wall,
    }
    return payload, spec, raw


def test_launch_frozen_hash_constants_match_only_static_files():
    assert verifier.sha256_file(PROTOCOL_PATH) == verifier.EXPECTED_PROTOCOL_SHA256
    assert verifier.sha256_file(RUNNER_PATH) == verifier.EXPECTED_RUNNER_SHA256
    assert (
        verifier.sha256_file(FORMAL_MANIFEST_PATH)
        == verifier.EXPECTED_FORMAL_MANIFEST_SHA256
    )
    verifier.validate_protocol(protocol())


@pytest.mark.parametrize("calibrated", [False, True])
def test_synthetic_raw_recomputes_scores_and_all_costs(calibrated):
    payload, spec, raw = raw_record(calibrated=calibrated)
    verified = verifier.validate_raw(payload, spec, raw)
    assert verified["adapted"]["environment_steps"] == 11
    assert verified["cost"]["q1_forward_rows"] == 176
    assert verified["cost"]["q1_backward_rows"] == 176
    assert verified["cost"]["q1_backward_calls"] == 22
    expected_delta = np.mean([3.0, 2.0, 5.0]) * 100.0 / (4592.3 - 1.629008)
    assert verified["within_controller_normalized_difference"] == pytest.approx(
        expected_delta
    )


def test_raw_normalized_score_cannot_drive_selection_if_not_return_derived():
    payload, spec, raw = raw_record(calibrated=False)
    raw["arms"]["adapted"]["normalized_score_mean"] += 100.0
    with pytest.raises(verifier.VerificationError, match="normalized_score_mean"):
        verifier.validate_raw(payload, spec, raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["channel"].update(environment_rollout_beta=0.0), "channel map"),
        (lambda raw: raw["controller"].update(critic="q2"), "controller map"),
        (lambda raw: raw["cost"].update(q1_forward_rows=175), "q1_forward_rows"),
        (
            lambda raw: raw["arms"]["adapted"]["lengths"].__setitem__(0, 0),
            "lengths",
        ),
    ],
)
def test_raw_channel_controller_length_and_cost_tampering_fails(mutation, message):
    payload, spec, raw = raw_record(calibrated=True)
    mutation(raw)
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.validate_raw(payload, spec, raw)


def test_strict_json_reader_rejects_duplicate_keys_and_nan(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="duplicate JSON key"):
        verifier.read_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="non-standard JSON constant"):
        verifier.read_json(nonfinite)


def test_selection_uses_recomputed_score_tie_rule_and_one_endpoint():
    def candidate(eta, score):
        return {
            "spec": {"eta": eta},
            "verified": {"adapted": {"normalized_mean": score}},
        }

    selected = verifier.choose(
        [candidate(0.3, 5.0), candidate(0.1, 5.0), candidate(0.03, 4.0)]
    )
    assert selected["spec"]["eta"] == 0.1
    assert verifier.required_endpoint_extension(0.01) == 0.003
    assert verifier.required_endpoint_extension(1.0) == 3.0
    assert verifier.required_endpoint_extension(0.1) is None


def test_paired_statistics_use_ddof1_t_and_deterministic_pcg64():
    values = np.asarray([1.0, 2.0, 3.0, 8.0], dtype=np.float64)
    first = verifier.paired_statistics(
        values, bootstrap_seed=20261915, replicates=20000
    )
    second = verifier.paired_statistics(
        values, bootstrap_seed=20261915, replicates=20000
    )
    assert first == second
    assert first["sample_std_ddof1"] == pytest.approx(values.std(ddof=1))
    assert first["mean"] == pytest.approx(values.mean())
    assert first["positive_episode_count"] == 4
    assert first["negative_episode_count"] == 0
    assert math.isfinite(first["t95_low"])
    assert math.isfinite(first["bootstrap95_high"])


def test_create_only_writer_refuses_existing_target(tmp_path):
    target = tmp_path / "verified.json"
    verifier.write_new(target, "{}\n")
    with pytest.raises(verifier.VerificationError, match="refusing to overwrite"):
        verifier.write_new(target, "{}\n")


def test_input_inventory_deduplicates_files_and_accumulates_roles(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"immutable")
    expected = verifier.sha256_file(source)
    inventory = verifier.InputInventory()
    inventory.add(source, "first", expected_sha256=expected)
    inventory.add(source, "second", expected_sha256=expected)
    assert inventory.records() == [
        {
            "path": str(source.resolve()),
            "sha256": expected,
            "size_bytes": 9,
            "roles": ["first", "second"],
        }
    ]
