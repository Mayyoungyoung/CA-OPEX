import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
RESULTS_DIR = next(
    (
        candidate
        for candidate in (CODE_DIR / "results", CODE_DIR.parent / "results")
        if candidate.is_dir()
    ),
    CODE_DIR / "results",
)

import run_equal_compute_nominal_control as audit


PROTOCOL_PATH = CODE_DIR / "equal_compute_nominal_control_protocol.json"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_frozen_protocol_has_exact_requested_blocks_and_honest_label():
    payload = protocol()
    audit.validate_protocol(payload)
    assert payload["evidence_label"] == (
        "post_confirmation_fresh_rollout_mechanism_audit"
    )
    assert payload["chronology"]["formal_results_seen_before_protocol_freeze"]
    assert not payload["chronology"]["globally_blind_confirmation_claim"]
    assert payload["development"]["step_size_grid"] == [
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    ]
    assert payload["development"]["endpoint_extension"] == {
        "if_lower_endpoint_selected": 0.003,
        "if_upper_endpoint_selected": 3.0,
        "run_exactly_one_outward_point_only_if_endpoint_wins_initial_grid": True,
    }
    assert [
        item["training_seed"] for item in payload["holdout"]["base_checkpoints"]
    ] == [1, 10]


def test_real_calibration_shape_matches_protocol():
    payload = protocol()
    calibration = json.loads(
        (
            RESULTS_DIR
            / "channel_calibration/beta125_n512_seed27001_calibration.json"
        ).read_text(encoding="utf-8")
    )
    assert "beta" not in calibration
    audit.same_float(
        calibration["estimate"]["beta_mle"],
        payload["channel_calibration"]["estimated_beta"],
        "calibration estimate.beta_mle",
    )
    assert calibration["pair_count"] == payload["channel_calibration"]["pair_count"]


def test_nominal_and_calibrated_commands_match_compute_but_not_q_model():
    payload = protocol()
    checkpoint = payload["holdout"]["base_checkpoints"][0]
    nominal = audit.make_spec(
        payload,
        stage="holdout",
        controller_id="nominal_tuned",
        checkpoint=checkpoint,
        eta=0.1,
    )
    calibrated = audit.make_spec(
        payload,
        stage="holdout",
        controller_id="calibrated_identity_eta_0.3",
        checkpoint=checkpoint,
        eta=0.3,
    )
    assert (nominal["K"], nominal["T"]) == (8, 2)
    assert (calibrated["K"], calibrated["T"]) == (8, 2)
    assert nominal["baseline_transform"] == calibrated["baseline_transform"] == "identity"
    assert nominal["model_beta"] == 0.0
    assert calibrated["model_beta"] == payload["channel_calibration"]["estimated_beta"]
    nominal_command = audit.command(payload, nominal)
    calibrated_command = audit.command(payload, calibrated)
    assert nominal_command[-2:] == ["--model-action-noise-beta", "0"]
    assert calibrated_command[-2:] == [
        "--channel-calibration",
        payload["channel_calibration"]["path"],
    ]


def test_real_raw_schema_and_implementation_alias_mapping_validate():
    payload = protocol()
    checkpoint = payload["holdout"]["base_checkpoints"][0]
    spec = audit.make_spec(
        payload,
        stage="holdout",
        controller_id="calibrated_identity_eta_0.3",
        checkpoint=checkpoint,
        eta=0.3,
    )
    spec.update(
        path=(
            RESULTS_DIR
            / "inverse_residual_confirm/external_controls/ca_opex_identity_seed1_beta125_50.json"
        ),
        env_seed=39300,
        action_seed=49300,
        gradient_seed=69300,
    )
    raw = audit.validate_raw(payload, spec)
    assert raw["status"] == "complete"
    assert raw["arms"]["adapted"]["q1_rows_per_action"] == 16


def test_selection_maximizes_score_and_exact_tie_uses_smaller_eta():
    def pair(eta, score):
        return (
            {"eta": eta},
            {"arms": {"adapted": {"normalized_score_mean": score}}},
        )

    selected, _ = audit.select([pair(0.3, 4.0), pair(0.1, 4.0), pair(0.03, 3.0)])
    assert selected["eta"] == 0.1


def test_paired_statistics_are_ddof1_and_bootstrap_is_deterministic():
    values = np.asarray([1.0, 2.0, 3.0, 8.0], dtype=np.float64)
    first = audit.paired_stats(values, bootstrap_seed=1234, replicates=20000)
    second = audit.paired_stats(values, bootstrap_seed=1234, replicates=20000)
    assert first == second
    assert first["sample_std_ddof1"] == pytest.approx(values.std(ddof=1))
    assert first["mean"] == pytest.approx(3.5)
    assert first["bootstrap_replicates"] == 20000
    assert math.isfinite(first["t95_low"])
    assert math.isfinite(first["t95_high"])


def test_atomic_writer_refuses_overwrite(tmp_path):
    target = tmp_path / "result.json"
    audit.write_new(target, "{}\n")
    with pytest.raises(audit.AuditError, match="refusing to overwrite"):
        audit.write_new(target, "{}\n")
