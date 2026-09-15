"""Independently verify the post-confirmation equal-compute controller audit.

This verifier deliberately does not import ``run_equal_compute_nominal_control``.
It reads the launch-frozen protocol and runner as data, checks their pre-launch
hashes, validates every raw evaluator record, reconstructs development
selection and the conditional endpoint rule, and recomputes all holdout scores,
costs, paired intervals, and bootstrap intervals from raw returns.

The v2 outputs are written to a new directory with create-only semantics.  Use
``--check-only`` for a read-only validation pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import scipy
from scipy.stats import t as student_t


EXPECTED_PROTOCOL_SHA256 = (
    "fc8fb274f283cd32ef6d45d8a944bff9cdb6757f3927816af9b8a9f17d7cf034"
)
EXPECTED_RUNNER_SHA256 = (
    "c5cd5a3631ffc38e8806c16471f77ab48c00894aff266a44e9e9ee6490a564c1"
)
EXPECTED_FORMAL_MANIFEST_SHA256 = (
    "57ac5bf657ffbb7185e703949d3b5cbfad4f8061fe3d218ae755ba41aa704807"
)
PROTOCOL_SCHEMA = "equal-compute-nominal-control-protocol-v1"
RAW_SCHEMA = "channel_opex_v1"
V1_SELECTION_SCHEMA = "equal-compute-nominal-control-selection-v1"
V1_AGGREGATE_SCHEMA = "equal-compute-nominal-control-aggregate-v1"
V2_SCHEMA = "equal-compute-nominal-control-independent-verification-v2"
EVIDENCE = "post_confirmation_fresh_rollout_mechanism_audit"

IMPLEMENTATION_KEY_MAP = {
    "evaluate_channel_opex.py": "evaluate_sha256",
    "inverse_residual_core.py": "inverse_residual_core_sha256",
    "td3bc_core.py": "td3bc_core_sha256",
    "train_td3bc.py": "train_td3bc_sha256",
    "evaluation_controls.py": "evaluation_controls_sha256",
    "train_inverse_residual_adapter.py": "train_inverse_residual_adapter_sha256",
}


class VerificationError(RuntimeError):
    """Raised when an immutable-input or raw-result invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise VerificationError(f"non-standard JSON constant {token!r} in {path}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    assert_json_finite(value, str(path))
    return value


def assert_json_finite(value: object, label: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON number at {label}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            assert_json_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_json_finite(child, f"{label}[{index}]")


def exact_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool==int numeric coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(
            exact_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_new(path: Path, text: str) -> None:
    """Atomically publish one create-only UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise VerificationError(f"refusing to overwrite output: {path}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise VerificationError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def same_float(
    actual: object,
    expected: float,
    label: str,
    *,
    absolute_tolerance: float = 1e-9,
) -> None:
    if isinstance(actual, bool):
        raise VerificationError(f"{label} is boolean, not numeric")
    try:
        observed = float(actual)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} is not numeric") from exc
    require(math.isfinite(observed), f"{label} is not finite")
    require(
        math.isclose(
            observed,
            float(expected),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ),
        f"{label} mismatch: {observed!r} != {expected!r}",
    )


def numeric_vector(value: object, label: str, count: int) -> np.ndarray:
    require(isinstance(value, list), f"{label} is not a list")
    require(len(value) == count, f"{label} has {len(value)} items, expected {count}")
    require(
        all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value),
        f"{label} contains a non-numeric value",
    )
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} contains a non-numeric value") from exc
    require(result.shape == (count,), f"{label} is not one-dimensional")
    require(bool(np.isfinite(result).all()), f"{label} contains a non-finite value")
    return result


def integer_vector(value: object, label: str, expected: Sequence[int]) -> List[int]:
    require(isinstance(value, list), f"{label} is not a list")
    require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{label} contains a non-integer",
    )
    observed = list(value)
    require(observed == list(expected), f"{label} differs from the frozen sequence")
    return observed


class InputInventory:
    """Deduplicated inventory of every file read by the independent verifier."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, Dict[str, Any]] = {}

    def add(
        self,
        path: Path,
        role: str,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        resolved = path.resolve()
        require(resolved.is_file(), f"required input is missing: {resolved}")
        key = str(resolved)
        digest = sha256_file(resolved)
        if expected_sha256 is not None:
            require(
                digest == expected_sha256,
                f"input hash mismatch for {resolved}: {digest} != {expected_sha256}",
            )
        record = self._records.get(key)
        if record is None:
            record = {
                "path": key,
                "sha256": digest,
                "size_bytes": resolved.stat().st_size,
                "roles": [],
            }
            self._records[key] = record
        else:
            require(record["sha256"] == digest, f"input changed while verifying: {resolved}")
        if role not in record["roles"]:
            record["roles"].append(role)
            record["roles"].sort()
        return digest

    def records(self) -> List[Dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]

    def verify_unchanged(self) -> None:
        for key, record in self._records.items():
            path = Path(key)
            require(path.is_file(), f"input disappeared while verifying: {path}")
            require(
                sha256_file(path) == record["sha256"]
                and path.stat().st_size == record["size_bytes"],
                f"input changed while verifying: {path}",
            )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "protocol schema mismatch")
    require(protocol.get("evidence_label") == EVIDENCE, "protocol evidence label mismatch")
    chronology = protocol.get("chronology", {})
    require(
        chronology.get("created_after_formal_confirmation_results_were_read") is True,
        "protocol chronology omits post-confirmation creation",
    )
    require(
        chronology.get("formal_results_seen_before_protocol_freeze") is True,
        "protocol chronology omits observed formal results",
    )
    require(
        chronology.get("globally_blind_confirmation_claim") is False,
        "protocol makes an invalid blindness claim",
    )
    environment = protocol.get("environment", {})
    require(environment.get("name") == "Walker2d-v4", "environment mismatch")
    same_float(environment.get("rollout_action_noise_beta"), 1.25, "rollout beta")
    same_float(environment.get("reference_min_score"), 1.629008, "reference minimum")
    same_float(environment.get("reference_max_score"), 4592.3, "reference maximum")

    development = protocol.get("development", {})
    require(
        development.get("controller")
        == {
            "semantic_alias": "equal_compute_nominal_q_identity_development",
            "baseline_transform": "identity",
            "model_action_noise_beta": 0.0,
            "gradient_noise_samples": 8,
            "gradient_steps": 2,
            "delta_max": 2.0,
            "q_reducer": "mean_q1",
            "critic": "frozen_q1",
        },
        "development controller contract mismatch",
    )
    require(
        [float(value) for value in development.get("step_size_grid", [])]
        == [0.01, 0.03, 0.1, 0.3, 1.0],
        "development eta grid mismatch",
    )
    require(
        (
            development.get("episode_count_per_arm"),
            development.get("environment_seed_start"),
            development.get("action_noise_seed_start"),
            development.get("gradient_noise_seed_start"),
        )
        == (10, 28300, 38300, 58300),
        "development rollout block mismatch",
    )
    extension = development.get("endpoint_extension", {})
    require(
        extension
        == {
            "if_lower_endpoint_selected": 0.003,
            "if_upper_endpoint_selected": 3.0,
            "run_exactly_one_outward_point_only_if_endpoint_wins_initial_grid": True,
        },
        "development endpoint-extension rule mismatch",
    )
    holdout = protocol.get("holdout", {})
    require(
        (
            holdout.get("episode_count_per_arm"),
            holdout.get("environment_seed_start"),
            holdout.get("action_noise_seed_start"),
            holdout.get("gradient_noise_seed_start"),
        )
        == (50, 79300, 89300, 99300),
        "holdout rollout block mismatch",
    )
    require(
        [item.get("training_seed") for item in holdout.get("base_checkpoints", [])]
        == [1, 10],
        "holdout checkpoint seeds must be [1, 10]",
    )
    expected_ids = {
        "nominal_tuned",
        "nominal_matched_eta_0.3",
        "calibrated_identity_eta_0.3",
        "complete_calibrated_inverse_eta_0.1",
    }
    controllers = holdout.get("controllers", {})
    require(set(controllers) == expected_ids, "holdout controller inventory mismatch")
    for controller_id, controller in controllers.items():
        require(
            int(controller.get("gradient_noise_samples", -1)) == 8
            and int(controller.get("gradient_steps", -1)) == 2,
            f"{controller_id} does not use K=8,T=2",
        )
    require(
        controllers["nominal_tuned"]
        == {
            "semantic_alias": "equal_compute_nominal_q_identity_tuned",
            "baseline_transform": "identity",
            "model_source": "known_zero_beta_without_pair_calibration",
            "model_action_noise_beta": 0.0,
            "step_size_source": "frozen_development_selection",
            "gradient_noise_samples": 8,
            "gradient_steps": 2,
            "delta_max": 2.0,
        },
        "nominal_tuned contract mismatch",
    )
    require(
        controllers["nominal_matched_eta_0.3"]
        == {
            "semantic_alias": "equal_compute_nominal_q_identity_eta_0p3",
            "baseline_transform": "identity",
            "model_source": "known_zero_beta_without_pair_calibration",
            "model_action_noise_beta": 0.0,
            "step_size": 0.3,
            "skip_if_identical_to_nominal_tuned": True,
            "gradient_noise_samples": 8,
            "gradient_steps": 2,
            "delta_max": 2.0,
        },
        "nominal matched-eta contract mismatch",
    )
    require(
        controllers["calibrated_identity_eta_0.3"]
        == {
            "semantic_alias": "equal_compute_calibrated_q_identity_eta_0p3",
            "baseline_transform": "identity",
            "model_source": "pair_calibration",
            "step_size": 0.3,
            "gradient_noise_samples": 8,
            "gradient_steps": 2,
            "delta_max": 2.0,
        },
        "calibrated identity contract mismatch",
    )
    require(
        controllers["complete_calibrated_inverse_eta_0.1"]
        == {
            "semantic_alias": "complete_ca_opex_calibrated_inverse_eta_0p1",
            "baseline_transform": "inverse",
            "model_source": "pair_calibration",
            "step_size": 0.1,
            "gradient_noise_samples": 8,
            "gradient_steps": 2,
            "delta_max": 0.25,
        },
        "complete inverse controller contract mismatch",
    )
    statistics = protocol.get("statistics", {})
    require(statistics.get("sample_standard_deviation_ddof") == 1, "ddof mismatch")
    bootstrap = statistics.get("bootstrap", {})
    require(
        bootstrap.get("replicates") == 20000
        and bootstrap.get("bit_generator") == "numpy_PCG64"
        and bootstrap.get("seed_root") == 20260915,
        "bootstrap contract mismatch",
    )
    expected_comparisons = [
        ("complete_minus_nominal_tuned", "complete_calibrated_inverse_eta_0.1", "nominal_tuned"),
        ("complete_minus_nominal_matched_eta_0.3", "complete_calibrated_inverse_eta_0.1", "nominal_matched_eta_0.3"),
        ("complete_minus_calibrated_identity_eta_0.3", "complete_calibrated_inverse_eta_0.1", "calibrated_identity_eta_0.3"),
        ("calibrated_identity_minus_nominal_tuned", "calibrated_identity_eta_0.3", "nominal_tuned"),
        ("calibrated_identity_minus_nominal_matched_eta_0.3", "calibrated_identity_eta_0.3", "nominal_matched_eta_0.3"),
    ]
    observed_comparisons = [
        (item.get("id"), item.get("left"), item.get("right"))
        for item in statistics.get("comparisons_in_order", [])
    ]
    require(observed_comparisons == expected_comparisons, "comparison endpoint/order contract mismatch")

    # The raw arrays are checked against these blocks later.  This static set
    # check additionally demonstrates that the post-confirmation holdout is
    # disjoint from both development and the original formal rollout block.
    seed_blocks = {
        "development_environment": set(range(28300, 28310)),
        "development_action": set(range(38300, 38310)),
        "development_gradient": set(range(58300, 58310)),
        "formal_environment": set(range(39300, 39350)),
        "formal_action": set(range(49300, 49350)),
        "formal_gradient": set(range(69300, 69350)),
        "holdout_environment": set(range(79300, 79350)),
        "holdout_action": set(range(89300, 89350)),
        "holdout_gradient": set(range(99300, 99350)),
    }
    require(
        seed_blocks["holdout_environment"].isdisjoint(seed_blocks["development_environment"])
        and seed_blocks["holdout_environment"].isdisjoint(seed_blocks["formal_environment"])
        and seed_blocks["holdout_action"].isdisjoint(seed_blocks["development_action"])
        and seed_blocks["holdout_action"].isdisjoint(seed_blocks["formal_action"])
        and seed_blocks["holdout_gradient"].isdisjoint(seed_blocks["development_gradient"])
        and seed_blocks["holdout_gradient"].isdisjoint(seed_blocks["formal_gradient"]),
        "fresh holdout seed blocks overlap development/formal blocks",
    )


def eta_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def make_spec(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    controller_id: str,
    checkpoint: Mapping[str, Any],
    eta: float,
) -> Dict[str, Any]:
    if stage == "development":
        controller = protocol["development"]["controller"]
        block = protocol["development"]
        filename = f"nominal_q_identity_eta_{eta_token(eta)}_seed0_dev10.json"
        calibrated = False
        semantic_alias = controller["semantic_alias"]
    elif stage == "holdout":
        controller = protocol["holdout"]["controllers"][controller_id]
        block = protocol["holdout"]
        filename = f"{controller_id}_seed{checkpoint['training_seed']}_fresh50.json"
        calibrated = controller["model_source"] == "pair_calibration"
        semantic_alias = controller["semantic_alias"]
    else:
        raise VerificationError(f"unsupported stage: {stage}")
    directory = protocol["artifacts"][
        "development_directory" if stage == "development" else "holdout_directory"
    ]
    return {
        "stage": stage,
        "controller_id": controller_id,
        "semantic_alias": semantic_alias,
        "checkpoint": dict(checkpoint),
        "eta": float(eta),
        "baseline_transform": controller["baseline_transform"],
        "model_beta": float(
            protocol["channel_calibration"]["estimated_beta"] if calibrated else 0.0
        ),
        "calibrated": calibrated,
        "K": int(controller["gradient_noise_samples"]),
        "T": int(controller["gradient_steps"]),
        "delta": float(controller["delta_max"]),
        "episodes": int(block["episode_count_per_arm"]),
        "environment_seed": int(block["environment_seed_start"]),
        "action_noise_seed": int(block["action_noise_seed_start"]),
        "gradient_noise_seed": int(block["gradient_noise_seed_start"]),
        "path": Path(protocol["paths"]["output_root"]) / directory / filename,
    }


def expected_controller(spec: Mapping[str, Any]) -> Dict[str, Any]:
    calibrated = bool(spec["calibrated"])
    return {
        "baseline_transform": spec["baseline_transform"],
        "step_size": spec["eta"],
        "gradient_steps": spec["T"],
        "K": spec["K"],
        "execution_noise_samples": spec["K"],
        "model_beta": spec["model_beta"],
        "calibration_mode": (
            "censored_uniform_plus_clip_pair_calibration"
            if calibrated
            else "cli_known_beta_without_pair_calibration"
        ),
        "gradient_noise_seed": spec["gradient_noise_seed"],
        "q_reducer": "mean_q1",
        "delta_max": spec["delta"],
        "critic": "frozen_q1",
        "gradient_objective": (
            "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
            "uniform_noise_per_gradient_step"
            if calibrated
            else "q1_of_action_bounded_command_with_deterministic_zero_channel_noise"
        ),
        "gradient_steps_per_action": spec["T"],
        "actor_parameter_updates": 0,
        "critic_parameter_updates": 0,
    }


def expected_evaluation_protocol(
    protocol: Mapping[str, Any], spec: Mapping[str, Any]
) -> Dict[str, Any]:
    calibrated = bool(spec["calibrated"])
    return {
        "environment": protocol["environment"]["name"],
        "episode_count_per_arm": spec["episodes"],
        "paired_environment_and_action_noise_seeds": True,
        "environment_rng_namespace": "gymnasium_env_reset",
        "actuator_noise_rng_namespace": "numpy_generator_per_episode",
        "gradient_noise_rng_namespace": (
            "torch_generator_per_episode_fresh_antithetic_per_gradient_step"
            if calibrated
            else "none_beta_zero_deterministic_objective"
        ),
        "gradient_noise_sampling_frequency": (
            "per_gradient_step" if calibrated else "none_beta_zero"
        ),
        "gradient_noise_antithetic": calibrated,
        "gradient_noise_seed_start": spec["gradient_noise_seed"],
        "environment_seed_start": spec["environment_seed"],
        "action_noise_seed_start": spec["action_noise_seed"],
        "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling": True,
        "selection_rule": "all requested episodes retained",
    }


def validate_arm(
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    arm_name: str,
    arm: Mapping[str, Any],
) -> Dict[str, Any]:
    episodes = int(spec["episodes"])
    expected_environment = list(
        range(spec["environment_seed"], spec["environment_seed"] + episodes)
    )
    expected_action = list(
        range(spec["action_noise_seed"], spec["action_noise_seed"] + episodes)
    )
    expected_gradient = list(
        range(spec["gradient_noise_seed"], spec["gradient_noise_seed"] + episodes)
    )
    integer_vector(arm.get("environment_seeds"), f"{arm_name}.environment_seeds", expected_environment)
    integer_vector(arm.get("action_noise_seeds"), f"{arm_name}.action_noise_seeds", expected_action)
    integer_vector(arm.get("gradient_noise_seeds"), f"{arm_name}.gradient_noise_seeds", expected_gradient)
    returns = numeric_vector(arm.get("returns"), f"{arm_name}.returns", episodes)
    lengths_raw = arm.get("lengths")
    require(isinstance(lengths_raw, list), f"{arm_name}.lengths is not a list")
    require(
        len(lengths_raw) == episodes
        and all(
            isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= 1000
            for item in lengths_raw
        ),
        f"{arm_name}.lengths is invalid",
    )
    lengths = [int(item) for item in lengths_raw]
    steps = int(sum(lengths))
    adapted = arm_name == "adapted"
    calibrated_gradient = adapted and bool(spec["calibrated"])
    rows_per_action = int(spec["K"] * spec["T"] if adapted else 0)
    q_rows_by_episode = [length * rows_per_action for length in lengths]
    q_rows_total = int(sum(q_rows_by_episode))
    backward_calls = int(steps * spec["T"] if adapted else 0)

    require(
        arm.get("arm")
        == (
            "channel_opex"
            if adapted
            else f"{spec['baseline_transform']}_anchor_baseline"
        ),
        f"{arm_name}.arm label mismatch",
    )
    same_float(
        arm.get("action_noise_beta"),
        protocol["environment"]["rollout_action_noise_beta"],
        f"{arm_name}.action_noise_beta",
    )
    raw_mean = float(returns.mean())
    raw_std = float(returns.std(ddof=0))
    reference_min = float(protocol["environment"]["reference_min_score"])
    reference_max = float(protocol["environment"]["reference_max_score"])
    normalized = 100.0 * (returns - reference_min) / (reference_max - reference_min)
    normalized_mean = float(normalized.mean())
    normalized_std = float(normalized.std(ddof=0))
    same_float(arm.get("return_mean"), raw_mean, f"{arm_name}.return_mean")
    same_float(arm.get("return_std"), raw_std, f"{arm_name}.return_std")
    same_float(
        arm.get("normalized_score_mean"),
        normalized_mean,
        f"{arm_name}.normalized_score_mean",
    )
    same_float(
        arm.get("normalized_score_std"),
        normalized_std,
        f"{arm_name}.normalized_score_std",
    )
    require(
        arm.get("gradient_noise_stream_used") is calibrated_gradient,
        f"{arm_name}.gradient_noise_stream_used mismatch",
    )
    require(
        arm.get("gradient_noise_sampling_frequency")
        == ("per_gradient_step" if calibrated_gradient else "none_beta_zero_or_baseline_arm"),
        f"{arm_name}.gradient_noise_sampling_frequency mismatch",
    )
    require(
        arm.get("gradient_noise_draw_calls")
        == (backward_calls if calibrated_gradient else 0),
        f"{arm_name}.gradient_noise_draw_calls mismatch",
    )
    require(arm.get("q1_rows_per_action") == rows_per_action, f"{arm_name}.q1_rows_per_action mismatch")
    require(arm.get("q1_rows_by_episode") == q_rows_by_episode, f"{arm_name}.q1_rows_by_episode mismatch")
    require(arm.get("q1_rows_total") == q_rows_total, f"{arm_name}.q1_rows_total mismatch")
    require(arm.get("q1_gradient_calls") == backward_calls, f"{arm_name}.q1_gradient_calls mismatch")
    try:
        arm_wall_time = float(arm.get("wall_time_seconds"))
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{arm_name}.wall_time_seconds is not numeric") from exc
    require(math.isfinite(arm_wall_time) and arm_wall_time >= 0.0, f"{arm_name}.wall time is invalid")
    return {
        "returns": returns,
        "lengths": lengths,
        "environment_steps": steps,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "normalized_mean": normalized_mean,
        "normalized_std": normalized_std,
        "q1_rows_per_action": rows_per_action,
        "q1_rows_total": q_rows_total,
        "q1_backward_calls": backward_calls,
        "wall_time_seconds": arm_wall_time,
    }


def validate_raw(
    protocol: Mapping[str, Any], spec: Mapping[str, Any], raw: Mapping[str, Any]
) -> Dict[str, Any]:
    """Strictly validate one evaluator JSON and return raw-derived quantities."""

    require(raw.get("raw_schema") == RAW_SCHEMA, "raw schema mismatch")
    require(raw.get("status") == "complete", "raw status is not complete")
    require(raw.get("environment") == protocol["environment"]["name"], "raw environment mismatch")
    expected_method = f"channel_aware_opex_{spec['baseline_transform']}_anchor"
    require(raw.get("method_id") == expected_method, "raw method_id mismatch")
    require(
        raw.get("method_scope")
        == "stronger channel-aware extension of OPEX; not an unchanged reproduction of the original paper",
        "raw method_scope mismatch",
    )
    base = raw.get("base_checkpoint", {})
    require(base.get("sha256") == spec["checkpoint"]["sha256"], "base checkpoint SHA mismatch")
    require(base.get("path") == spec["checkpoint"]["path"], "base checkpoint path mismatch")

    expected_implementation = {
        IMPLEMENTATION_KEY_MAP[filename]: digest
        for filename, digest in protocol["implementation"].items()
    }
    require(
        exact_json_equal(raw.get("implementation"), expected_implementation),
        "raw evaluator implementation map mismatch",
    )
    require(
        exact_json_equal(raw.get("controller"), expected_controller(spec)),
        "raw controller map mismatch",
    )
    require(
        exact_json_equal(
            raw.get("evaluation_protocol"),
            expected_evaluation_protocol(protocol, spec),
        ),
        "raw evaluation protocol map mismatch",
    )
    expected_normalization = {
        "reference_min_score": protocol["environment"]["reference_min_score"],
        "reference_max_score": protocol["environment"]["reference_max_score"],
    }
    require(
        exact_json_equal(raw.get("normalization"), expected_normalization),
        "raw normalization mismatch",
    )
    expected_channel = {
        "gradient_model": "iid_uniform_additive_then_clip",
        "model_or_calibration_beta": spec["model_beta"],
        "environment_rollout_beta": protocol["environment"]["rollout_action_noise_beta"],
        "beta_mismatch": spec["model_beta"] != protocol["environment"]["rollout_action_noise_beta"],
    }
    require(exact_json_equal(raw.get("channel"), expected_channel), "raw channel map mismatch")

    calibration = raw.get("calibration", {})
    if spec["calibrated"]:
        require(
            calibration.get("source") == "censored_uniform_plus_clip_pair_calibration"
            and calibration.get("calibration_path") == protocol["channel_calibration"]["path"]
            and calibration.get("calibration_sha256") == protocol["channel_calibration"]["sha256"]
            and calibration.get("pair_count") == protocol["channel_calibration"]["pair_count"],
            "raw calibrated-channel provenance mismatch",
        )
        same_float(calibration.get("beta"), spec["model_beta"], "raw calibration beta")
    else:
        require(
            exact_json_equal(
                calibration,
                {
                "source": "cli_known_beta_without_pair_calibration",
                "beta": 0.0,
                "pair_count": None,
                "calibration_path": None,
                "calibration_sha256": None,
                },
            ),
            "raw nominal-channel provenance mismatch",
        )

    arms = raw.get("arms")
    require(isinstance(arms, dict) and set(arms) == {"baseline_only", "adapted"}, "raw arm inventory mismatch")
    baseline = validate_arm(protocol, spec, "baseline_only", arms["baseline_only"])
    adapted = validate_arm(protocol, spec, "adapted", arms["adapted"])

    differences = adapted["returns"] - baseline["returns"]
    paired = raw.get("paired", {})
    recorded = numeric_vector(
        paired.get("adapted_minus_baseline_returns"),
        "paired.adapted_minus_baseline_returns",
        spec["episodes"],
    )
    require(np.array_equal(recorded, differences), "raw paired differences are not return-derived")
    same_float(paired.get("return_difference_mean"), float(differences.mean()), "paired.return_difference_mean")
    same_float(paired.get("return_difference_std"), float(differences.std(ddof=0)), "paired.return_difference_std")
    require(paired.get("positive_episode_count") == int((differences > 0).sum()), "paired positive count mismatch")
    require(paired.get("episode_count") == int(spec["episodes"]), "paired episode count mismatch")

    cost = raw.get("cost", {})
    expected_cost = {
        "cost_scope": "adapted_arm_deployment_controller_only",
        "environment_steps": adapted["environment_steps"],
        "q1_forward_rows": adapted["q1_rows_total"],
        "q1_backward_rows": adapted["q1_rows_total"],
        "q1_backward_calls": adapted["q1_backward_calls"],
        "base_actor_rows": adapted["environment_steps"],
        "baseline_environment_steps": baseline["environment_steps"],
        "baseline_base_actor_rows": baseline["environment_steps"],
        "paired_evaluation_environment_steps": (
            baseline["environment_steps"] + adapted["environment_steps"]
        ),
        "baseline_q1_rows_total": 0,
        "adapted_q1_rows_total": adapted["q1_rows_total"],
        "adapted_q1_rows_per_environment_step": spec["K"] * spec["T"],
        "adapted_backward_calls": adapted["q1_backward_calls"],
    }
    for key, expected in expected_cost.items():
        require(
            exact_json_equal(cost.get(key), expected),
            f"cost.{key} mismatch",
        )
    same_float(cost.get("wall_time_seconds"), adapted["wall_time_seconds"], "cost.wall_time_seconds")
    same_float(cost.get("baseline_wall_time_seconds"), baseline["wall_time_seconds"], "cost.baseline_wall_time_seconds")
    paired_wall = float(cost.get("paired_evaluation_wall_time_seconds"))
    require(math.isfinite(paired_wall) and paired_wall >= 0.0, "paired wall time invalid")
    require(
        paired_wall + 1e-9 >= baseline["wall_time_seconds"] + adapted["wall_time_seconds"],
        "paired wall time is smaller than its two arms",
    )
    same_float(raw.get("wall_time_seconds"), paired_wall, "top-level wall_time_seconds")

    return {
        "raw": raw,
        "baseline": baseline,
        "adapted": adapted,
        "within_controller_normalized_difference": float(
            (adapted["returns"] - baseline["returns"]).mean()
            * 100.0
            / (
                protocol["environment"]["reference_max_score"]
                - protocol["environment"]["reference_min_score"]
            )
        ),
        "cost": {
            **expected_cost,
            "adapted_wall_time_seconds": adapted["wall_time_seconds"],
            "baseline_wall_time_seconds": baseline["wall_time_seconds"],
            "paired_evaluation_wall_time_seconds": paired_wall,
        },
    }


def choose(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    require(bool(candidates), "selection requires at least one candidate")
    return sorted(
        candidates,
        key=lambda item: (-float(item["verified"]["adapted"]["normalized_mean"]), float(item["spec"]["eta"])),
    )[0]


def required_endpoint_extension(initial_eta: float) -> float | None:
    if float(initial_eta) == 0.01:
        return 0.003
    if float(initial_eta) == 1.0:
        return 3.0
    return None


def paired_statistics(
    values: np.ndarray, *, bootstrap_seed: int, replicates: int
) -> Dict[str, Any]:
    require(values.ndim == 1 and values.size > 1, "paired statistics require at least two values")
    require(bool(np.isfinite(values).all()), "paired statistics received non-finite values")
    count = int(values.size)
    mean = float(values.mean())
    sample_std = float(values.std(ddof=1))
    standard_error = sample_std / math.sqrt(count)
    critical = float(student_t.ppf(0.975, count - 1))
    generator = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    bootstrap_means = values[
        generator.integers(0, count, size=(int(replicates), count))
    ].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    return {
        "n_episode_pairs": count,
        "mean": mean,
        "sample_std_ddof1": sample_std,
        "standard_error": standard_error,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap95_low": float(low),
        "bootstrap95_high": float(high),
        "positive_episode_count": int((values > 0).sum()),
        "negative_episode_count": int((values < 0).sum()),
        "zero_episode_count": int((values == 0).sum()),
    }


def load_and_validate_raw(
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    inventory: InputInventory,
) -> Dict[str, Any]:
    path = Path(spec["path"])
    require(not path.with_name(path.name + ".tmp").exists(), f"partial raw artifact exists: {path}.tmp")
    inventory.add(
        path,
        f"{spec['stage']}_raw:{spec['controller_id']}:seed{spec['checkpoint']['training_seed']}",
    )
    return validate_raw(protocol, spec, read_json(path))


def validate_static_inputs(
    protocol_path: Path,
    runner_path: Path,
    inventory: InputInventory,
    *,
    expected_protocol_sha256: str,
    expected_runner_sha256: str,
) -> Dict[str, Any]:
    protocol_sha = inventory.add(
        protocol_path,
        "launch_frozen_protocol",
        expected_sha256=expected_protocol_sha256,
    )
    runner_sha = inventory.add(
        runner_path,
        "launch_frozen_runner_source_not_imported",
        expected_sha256=expected_runner_sha256,
    )
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    code_dir = Path(protocol["paths"]["code_dir"])
    require(
        set(protocol["implementation"]) == set(IMPLEMENTATION_KEY_MAP),
        "protocol implementation inventory mismatch",
    )
    for filename, digest in protocol["implementation"].items():
        inventory.add(code_dir / filename, f"raw_evaluator_dependency:{filename}", expected_sha256=digest)
    formal_manifest_path = code_dir / "inverse_residual_frozen_confirmation_manifest.json"
    formal_manifest_sha = inventory.add(
        formal_manifest_path,
        "canonical_formal_manifest_for_seed_freshness",
        expected_sha256=EXPECTED_FORMAL_MANIFEST_SHA256,
    )
    formal_manifest = read_json(formal_manifest_path)
    formal_seed_sets = {"environment": set(), "action": set(), "gradient": set()}
    formal_opex_specs = [
        item
        for item in formal_manifest.get("external_evaluations", [])
        if item.get("raw_schema") == RAW_SCHEMA
    ]
    require(len(formal_opex_specs) == 6, "canonical formal manifest lacks six OPEX controls")
    for item in formal_opex_specs:
        evaluations = item.get("evaluations", [])
        require(len(evaluations) == 1, "formal OPEX control does not have one evaluation")
        block = evaluations[0].get("expected_protocol", {})
        count = int(block.get("episode_count", -1))
        require(count == 50, "formal OPEX episode count mismatch")
        formal_seed_sets["environment"].update(
            range(int(block["environment_seed_start"]), int(block["environment_seed_start"]) + count)
        )
        formal_seed_sets["action"].update(
            range(int(block["action_noise_seed_start"]), int(block["action_noise_seed_start"]) + count)
        )
        formal_seed_sets["gradient"].update(
            range(int(block["gradient_noise_seed_start"]), int(block["gradient_noise_seed_start"]) + count)
        )
    holdout = protocol["holdout"]
    holdout_seed_sets = {
        "environment": set(range(holdout["environment_seed_start"], holdout["environment_seed_start"] + holdout["episode_count_per_arm"])),
        "action": set(range(holdout["action_noise_seed_start"], holdout["action_noise_seed_start"] + holdout["episode_count_per_arm"])),
        "gradient": set(range(holdout["gradient_noise_seed_start"], holdout["gradient_noise_seed_start"] + holdout["episode_count_per_arm"])),
    }
    require(
        all(holdout_seed_sets[key].isdisjoint(formal_seed_sets[key]) for key in holdout_seed_sets),
        "holdout seed arrays overlap the canonical formal manifest",
    )
    checkpoints: Iterable[Mapping[str, Any]] = [protocol["development"]["base_checkpoint"], *protocol["holdout"]["base_checkpoints"]]
    for checkpoint in checkpoints:
        inventory.add(
            Path(checkpoint["path"]),
            f"base_checkpoint:seed{checkpoint['training_seed']}",
            expected_sha256=checkpoint["sha256"],
        )
    calibration_spec = protocol["channel_calibration"]
    calibration_path = Path(calibration_spec["path"])
    inventory.add(
        calibration_path,
        "channel_calibration",
        expected_sha256=calibration_spec["sha256"],
    )
    calibration = read_json(calibration_path)
    require(calibration.get("pair_count") == calibration_spec["pair_count"], "calibration pair count mismatch")
    estimate = calibration.get("estimate", {})
    same_float(estimate.get("beta_mle"), calibration_spec["estimated_beta"], "calibration beta MLE")
    return {
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "runner_sha256": runner_sha,
        "formal_manifest_sha256": formal_manifest_sha,
    }


def verify_development(
    protocol: Mapping[str, Any], inventory: InputInventory
) -> Dict[str, Any]:
    checkpoint = protocol["development"]["base_checkpoint"]
    candidates: List[Dict[str, Any]] = []
    for eta in protocol["development"]["step_size_grid"]:
        spec = make_spec(
            protocol,
            stage="development",
            controller_id="nominal_dev",
            checkpoint=checkpoint,
            eta=float(eta),
        )
        candidates.append({"spec": spec, "verified": load_and_validate_raw(protocol, spec, inventory)})
    initial = choose(candidates)
    extension = required_endpoint_extension(float(initial["spec"]["eta"]))
    if extension is not None:
        spec = make_spec(
            protocol,
            stage="development",
            controller_id="nominal_dev",
            checkpoint=checkpoint,
            eta=extension,
        )
        candidates.append({"spec": spec, "verified": load_and_validate_raw(protocol, spec, inventory)})
    expected_paths = {Path(item["spec"]["path"]).resolve() for item in candidates}
    development_directory = Path(protocol["paths"]["output_root"]) / protocol["artifacts"]["development_directory"]
    observed_paths = {path.resolve() for path in development_directory.glob("*.json")}
    require(observed_paths == expected_paths, "development raw JSON inventory differs from endpoint rule")
    selected = choose(candidates)
    rows = [
        {
            "step_size": float(item["spec"]["eta"]),
            "raw_score_mean_recomputed": item["verified"]["adapted"]["raw_mean"],
            "normalized_score_mean_recomputed": item["verified"]["adapted"]["normalized_mean"],
            "environment_steps": item["verified"]["adapted"]["environment_steps"],
            "q1_forward_rows": item["verified"]["cost"]["q1_forward_rows"],
            "q1_backward_rows": item["verified"]["cost"]["q1_backward_rows"],
            "q1_backward_calls": item["verified"]["cost"]["q1_backward_calls"],
            "path": str(Path(item["spec"]["path"]).resolve()),
            "sha256": sha256_file(Path(item["spec"]["path"])),
        }
        for item in sorted(candidates, key=lambda item: item["spec"]["eta"])
    ]
    return {
        "initial_selected_step_size": float(initial["spec"]["eta"]),
        "endpoint_extension_step_size": extension,
        "selected_step_size": float(selected["spec"]["eta"]),
        "endpoint_limited_after_single_extension": bool(
            extension is not None and float(selected["spec"]["eta"]) == extension
        ),
        "selection_metric": "raw-derived adapted normalized score over all ten development episodes",
        "tie_rule": "smaller_step_size",
        "candidates": rows,
    }


def holdout_specs(
    protocol: Mapping[str, Any], selected_eta: float
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    result: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for checkpoint in protocol["holdout"]["base_checkpoints"]:
        seed = int(checkpoint["training_seed"])
        specs = [
            make_spec(protocol, stage="holdout", controller_id="nominal_tuned", checkpoint=checkpoint, eta=selected_eta),
            make_spec(protocol, stage="holdout", controller_id="calibrated_identity_eta_0.3", checkpoint=checkpoint, eta=0.3),
            make_spec(protocol, stage="holdout", controller_id="complete_calibrated_inverse_eta_0.1", checkpoint=checkpoint, eta=0.1),
        ]
        if selected_eta != 0.3:
            specs.insert(
                1,
                make_spec(protocol, stage="holdout", controller_id="nominal_matched_eta_0.3", checkpoint=checkpoint, eta=0.3),
            )
        result[seed] = {spec["controller_id"]: spec for spec in specs}
    return result


def verify_holdout(
    protocol: Mapping[str, Any],
    selected_eta: float,
    inventory: InputInventory,
) -> Dict[str, Any]:
    specs_by_seed = holdout_specs(protocol, selected_eta)
    verified: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for seed, specs in specs_by_seed.items():
        verified[seed] = {
            controller_id: load_and_validate_raw(protocol, spec, inventory)
            for controller_id, spec in specs.items()
        }
    expected_paths = {
        Path(spec["path"]).resolve()
        for specs in specs_by_seed.values()
        for spec in specs.values()
    }
    holdout_directory = Path(protocol["paths"]["output_root"]) / protocol["artifacts"]["holdout_directory"]
    observed_paths = {path.resolve() for path in holdout_directory.glob("*.json")}
    require(observed_paths == expected_paths, "holdout raw JSON inventory mismatch")

    baseline_checks = []
    for seed, by_controller in verified.items():
        identity_ids = [
            controller_id
            for controller_id, spec in specs_by_seed[seed].items()
            if spec["baseline_transform"] == "identity"
        ]
        reference_id = identity_ids[0]
        reference = by_controller[reference_id]["raw"]["arms"]["baseline_only"]
        for controller_id in identity_ids[1:]:
            comparison = by_controller[controller_id]["raw"]["arms"]["baseline_only"]
            for field in ("environment_seeds", "action_noise_seeds", "gradient_noise_seeds", "returns", "lengths"):
                require(
                    reference[field] == comparison[field],
                    f"identity baseline differs for seed {seed}, field {field}",
                )
        baseline_checks.append(
            {
                "training_seed": seed,
                "status": "verified_elementwise_identical",
                "controllers": identity_ids,
                "fields": ["environment_seeds", "action_noise_seeds", "gradient_noise_seeds", "returns", "lengths"],
                "episode_count": len(reference["returns"]),
            }
        )

    controller_scores: List[Dict[str, Any]] = []
    for seed, specs in specs_by_seed.items():
        for controller_id, spec in specs.items():
            item = verified[seed][controller_id]
            controller_scores.append(
                {
                    "training_seed": seed,
                    "controller_id": controller_id,
                    "semantic_alias": spec["semantic_alias"],
                    "baseline_transform": spec["baseline_transform"],
                    "model_beta": spec["model_beta"],
                    "step_size": spec["eta"],
                    "K": spec["K"],
                    "T": spec["T"],
                    "delta_max": spec["delta"],
                    "episode_count_per_arm": spec["episodes"],
                    "baseline_raw_score_mean": item["baseline"]["raw_mean"],
                    "baseline_normalized_score_mean": item["baseline"]["normalized_mean"],
                    "adapted_raw_score_mean": item["adapted"]["raw_mean"],
                    "adapted_normalized_score_mean": item["adapted"]["normalized_mean"],
                    "adapted_minus_baseline_normalized_mean": item["within_controller_normalized_difference"],
                    "baseline_environment_steps": item["baseline"]["environment_steps"],
                    "adapted_environment_steps": item["adapted"]["environment_steps"],
                    "base_actor_rows_per_adapted_step": 1,
                    "q1_forward_rows_per_adapted_step": spec["K"] * spec["T"],
                    "q1_backward_rows_per_adapted_step": spec["K"] * spec["T"],
                    "q1_backward_calls_per_adapted_step": spec["T"],
                    "base_actor_rows_total": item["cost"]["base_actor_rows"],
                    "q1_forward_rows_total": item["cost"]["q1_forward_rows"],
                    "q1_backward_rows_total": item["cost"]["q1_backward_rows"],
                    "q1_backward_calls_total": item["cost"]["q1_backward_calls"],
                    "adapted_wall_time_seconds": item["cost"]["adapted_wall_time_seconds"],
                    "paired_evaluation_wall_time_seconds": item["cost"]["paired_evaluation_wall_time_seconds"],
                    "path": str(Path(spec["path"]).resolve()),
                    "sha256": sha256_file(Path(spec["path"])),
                }
            )

    score_groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in controller_scores:
        score_groups.setdefault(row["controller_id"], []).append(row)
    cross_controller = []
    for controller_id, rows in score_groups.items():
        require([row["training_seed"] for row in rows] == [1, 10], f"{controller_id} checkpoint order mismatch")
        cross_controller.append(
            {
                "controller_id": controller_id,
                "training_seeds": [1, 10],
                "n_checkpoint_seeds": 2,
                "descriptive_baseline_raw_mean": float(np.mean([row["baseline_raw_score_mean"] for row in rows])),
                "descriptive_baseline_normalized_mean": float(np.mean([row["baseline_normalized_score_mean"] for row in rows])),
                "descriptive_adapted_raw_mean": float(np.mean([row["adapted_raw_score_mean"] for row in rows])),
                "descriptive_adapted_normalized_mean": float(np.mean([row["adapted_normalized_score_mean"] for row in rows])),
                "training_seed_interval": None,
            }
        )

    factor = 100.0 / (
        protocol["environment"]["reference_max_score"]
        - protocol["environment"]["reference_min_score"]
    )
    comparisons = []
    skipped = []
    for comparison_index, comparison in enumerate(protocol["statistics"]["comparisons_in_order"]):
        for seed in (1, 10):
            by_controller = verified[seed]
            if comparison["right"] not in by_controller:
                skipped.append(
                    {
                        "comparison_id": comparison["id"],
                        "training_seed": seed,
                        "reason": "nominal matched controller duplicates nominal tuned eta=0.3",
                    }
                )
                continue
            left = by_controller[comparison["left"]]["adapted"]["returns"]
            right = by_controller[comparison["right"]]["adapted"]["returns"]
            values = (left - right) * factor
            seed_value = (
                protocol["statistics"]["bootstrap"]["seed_root"]
                + 1000 * seed
                + comparison_index
            )
            comparisons.append(
                {
                    "comparison_id": comparison["id"],
                    "left": comparison["left"],
                    "right": comparison["right"],
                    "training_seed": seed,
                    **paired_statistics(
                        values,
                        bootstrap_seed=seed_value,
                        replicates=protocol["statistics"]["bootstrap"]["replicates"],
                    ),
                }
            )
    comparison_groups: Dict[str, List[Mapping[str, Any]]] = {}
    for item in comparisons:
        comparison_groups.setdefault(item["comparison_id"], []).append(item)
    cross_comparisons = [
        {
            "comparison_id": comparison_id,
            "training_seeds": [item["training_seed"] for item in items],
            "n_checkpoint_seeds": len(items),
            "per_checkpoint_normalized_means": [item["mean"] for item in items],
            "descriptive_normalized_mean": float(np.mean([item["mean"] for item in items])),
            "positive_checkpoint_count": int(sum(item["mean"] > 0.0 for item in items)),
            "training_seed_interval": None,
        }
        for comparison_id, items in comparison_groups.items()
    ]
    return {
        "specs_by_seed": specs_by_seed,
        "verified_by_seed": verified,
        "identity_baseline_checks": baseline_checks,
        "controller_scores": controller_scores,
        "cross_checkpoint_controller_scores": cross_controller,
        "paired_comparisons": comparisons,
        "skipped_comparisons": skipped,
        "cross_checkpoint_comparisons": cross_comparisons,
    }


def validate_v1_selection(
    protocol_path: Path,
    runner_sha256: str,
    protocol_sha256: str,
    protocol: Mapping[str, Any],
    development: Mapping[str, Any],
    inventory: InputInventory,
) -> Dict[str, Any]:
    path = Path(protocol["paths"]["output_root"]) / protocol["artifacts"]["selection_json"]
    inventory.add(path, "runner_v1_selection_crosscheck")
    actual = read_json(path)
    expected_candidates = [
        {
            "step_size": item["step_size"],
            "adapted_normalized_score_mean": item["normalized_score_mean_recomputed"],
            "path": item["path"],
            "sha256": item["sha256"],
        }
        for item in development["candidates"]
    ]
    expected = {
        "schema_version": V1_SELECTION_SCHEMA,
        "status": "complete",
        "evidence_label": EVIDENCE,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha256,
        "runner_sha256": runner_sha256,
        "initial_selected_step_size": development["initial_selected_step_size"],
        "endpoint_extension_step_size": development["endpoint_extension_step_size"],
        "selected_step_size": development["selected_step_size"],
        "candidates": expected_candidates,
        "formal_results_seen_before_protocol_freeze": True,
        "globally_blind_confirmation_claim": False,
    }
    require(actual == expected, "runner v1 selection differs from independent reconstruction")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "exact_match": True}


def expected_v1_report(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    runner_sha256: str,
    selection_sha256: str,
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_artifacts = []
    specs_by_seed = holdout["specs_by_seed"]
    for seed, specs in specs_by_seed.items():
        for controller_id, spec in specs.items():
            path = Path(spec["path"])
            raw_artifacts.append(
                {
                    "training_seed": seed,
                    "controller_id": controller_id,
                    "semantic_alias": protocol["holdout"]["controllers"][controller_id]["semantic_alias"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    grouped: Dict[str, List[float]] = {}
    for item in holdout["paired_comparisons"]:
        grouped.setdefault(item["comparison_id"], []).append(item["mean"])
    v1_comparisons = [
        {
            key: value
            for key, value in item.items()
            if key not in {"negative_episode_count", "zero_episode_count"}
        }
        for item in holdout["paired_comparisons"]
    ]
    return {
        "schema_version": V1_AGGREGATE_SCHEMA,
        "status": "complete",
        "evidence_label": EVIDENCE,
        "protocol_sha256": protocol_sha256,
        "selection_sha256": selection_sha256,
        "runner_sha256": runner_sha256,
        "selected_nominal_step_size": development["selected_step_size"],
        "paired_comparisons": v1_comparisons,
        "skipped_comparisons": holdout["skipped_comparisons"],
        "cross_checkpoint_descriptive_means": {
            key: float(np.mean(values)) for key, values in grouped.items()
        },
        "identity_baseline_elementwise_identical": True,
        "raw_artifacts": raw_artifacts,
        "cross_checkpoint_inference_guardrail": protocol["statistics"]["cross_checkpoint_summary"],
    }


def validate_v1_aggregate(
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    runner_sha256: str,
    selection: Mapping[str, Any],
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
    inventory: InputInventory,
) -> Dict[str, Any]:
    path = (
        Path(protocol["paths"]["output_root"])
        / protocol["artifacts"]["aggregate_directory"]
        / protocol["artifacts"]["aggregate_json"]
    )
    inventory.add(path, "runner_v1_aggregate_crosscheck")
    actual = read_json(path)
    expected = expected_v1_report(
        protocol,
        protocol_sha256,
        runner_sha256,
        selection["sha256"],
        development,
        holdout,
    )
    require(actual == expected, "runner v1 aggregate differs from independent reconstruction")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "exact_match": True}


def build_report(
    protocol_path: Path,
    runner_path: Path,
    verifier_path: Path,
    *,
    expected_protocol_sha256: str = EXPECTED_PROTOCOL_SHA256,
    expected_runner_sha256: str = EXPECTED_RUNNER_SHA256,
    require_v1_outputs: bool = True,
) -> Dict[str, Any]:
    inventory = InputInventory()
    static = validate_static_inputs(
        protocol_path,
        runner_path,
        inventory,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_runner_sha256=expected_runner_sha256,
    )
    verifier_sha = inventory.add(verifier_path, "independent_verifier_source")
    protocol = static["protocol"]
    development = verify_development(protocol, inventory)
    holdout = verify_holdout(protocol, development["selected_step_size"], inventory)
    prior_outputs: Dict[str, Any] = {"required": require_v1_outputs}
    if require_v1_outputs:
        selection = validate_v1_selection(
            protocol_path,
            static["runner_sha256"],
            static["protocol_sha256"],
            protocol,
            development,
            inventory,
        )
        aggregate = validate_v1_aggregate(
            protocol,
            static["protocol_sha256"],
            static["runner_sha256"],
            selection,
            development,
            holdout,
            inventory,
        )
        prior_outputs.update(selection=selection, aggregate=aggregate)

    # Rehash the two files whose late self-hashing was unsafe in v1.  The v2
    # verifier carries their startup digests and fails if either changes while
    # verification is in progress.
    require(sha256_file(protocol_path) == static["protocol_sha256"], "protocol changed during verification")
    require(sha256_file(runner_path) == static["runner_sha256"], "runner changed during verification")
    require(sha256_file(verifier_path) == verifier_sha, "verifier changed during verification")
    inventory.verify_unchanged()
    input_files = inventory.records()
    return {
        "schema_version": V2_SCHEMA,
        "status": "verified_complete",
        "evidence_label": EVIDENCE,
        "verification_independence": {
            "imports_active_runner": False,
            "recomputes_selection_from_raw_returns": True,
            "recomputes_scores_costs_and_statistics_from_raw": True,
            "writes_or_modifies_raw_inputs": False,
        },
        "launch_frozen_contract": {
            "protocol_expected_sha256": expected_protocol_sha256,
            "protocol_observed_sha256": static["protocol_sha256"],
            "runner_expected_sha256": expected_runner_sha256,
            "runner_observed_sha256": static["runner_sha256"],
            "canonical_formal_manifest_sha256": static["formal_manifest_sha256"],
            "protocol_unchanged_during_v2_verification": True,
            "runner_unchanged_during_v2_verification": True,
        },
        "development_selection": development,
        "identity_baseline_checks": holdout["identity_baseline_checks"],
        "controller_scores": holdout["controller_scores"],
        "cross_checkpoint_controller_scores": holdout["cross_checkpoint_controller_scores"],
        "paired_comparisons": holdout["paired_comparisons"],
        "skipped_comparisons": holdout["skipped_comparisons"],
        "cross_checkpoint_comparisons": holdout["cross_checkpoint_comparisons"],
        "compute_scope": {
            "equal_per_decision_neural_budget": True,
            "base_actor_rows_per_adapted_step": 1,
            "q1_forward_rows_per_adapted_step": 16,
            "q1_backward_rows_per_adapted_step": 16,
            "q1_backward_calls_per_adapted_step": 2,
            "exact_wall_time_or_flop_matching_claim": False,
            "note": (
                "All holdout controllers match K=8,T=2 neural rows/calls per "
                "decision. The inverse controller additionally performs 48 "
                "deterministic scalar bisection iterations; total episode cost "
                "also varies with achieved horizon."
            ),
        },
        "statistical_guardrails": {
            "episode_intervals_condition_on_each_fixed_base_checkpoint": True,
            "checkpoint_seed_count": 2,
            "training_seed_interval_or_p_value": None,
            "rollout_episodes_pooled_across_checkpoints": False,
            "multiple_comparison_correction": None,
            "post_confirmation_not_blind": True,
            "chronology_is_runner_order_plus_create_only_evidence_not_external_timestamp": True,
            "runtime_versions_not_embedded_in_v1_raw_records": True,
        },
        "runner_v1_crosscheck": prior_outputs,
        "verifier_source_sha256": verifier_sha,
        "verification_runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "evaluation_runtime_note": (
                "These are v2 verification-library versions. The v1 raw evaluator "
                "records do not embed the original Python/Torch/Gymnasium/MuJoCo/CUDA versions."
            ),
        },
        "input_file_count": len(input_files),
        "input_files": input_files,
    }


CSV_FIELDS = [
    "row_type",
    "training_seed",
    "controller_id",
    "comparison_id",
    "step_size",
    "baseline_raw_score_mean",
    "baseline_normalized_score_mean",
    "adapted_raw_score_mean",
    "adapted_normalized_score_mean",
    "adapted_minus_baseline_normalized_mean",
    "mean",
    "sample_std_ddof1",
    "t95_low",
    "t95_high",
    "bootstrap95_low",
    "bootstrap95_high",
    "positive_episode_count",
    "n_episode_pairs",
    "base_actor_rows_total",
    "q1_forward_rows_total",
    "q1_backward_rows_total",
    "q1_backward_calls_total",
    "adapted_environment_steps",
    "adapted_wall_time_seconds",
    "path",
    "sha256",
    "size_bytes",
    "roles",
]


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for item in report["controller_scores"]:
        writer.writerow({"row_type": "controller_score", **item})
    for item in report["paired_comparisons"]:
        writer.writerow({"row_type": "paired_comparison", **item})
    for item in report["input_files"]:
        writer.writerow(
            {
                "row_type": "input_file",
                **item,
                "roles": ";".join(item["roles"]),
            }
        )
    return buffer.getvalue()


def format_value(value: object, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def render_markdown(report: Mapping[str, Any]) -> str:
    launch = report["launch_frozen_contract"]
    lines = [
        "# Independent v2 verification: equal-compute nominal-Q audit",
        "",
        f"Status: `{report['status']}`. Evidence label: `{report['evidence_label']}`.",
        "",
        "This report was reconstructed from raw returns without importing the active v1 runner. "
        "It is a post-confirmation fresh-rollout mechanism audit, not blind confirmation.",
        "",
        "## Launch-frozen provenance",
        "",
        "| File | Expected SHA-256 | Observed SHA-256 |",
        "|---|---|---|",
        f"| Protocol | `{launch['protocol_expected_sha256']}` | `{launch['protocol_observed_sha256']}` |",
        f"| Runner | `{launch['runner_expected_sha256']}` | `{launch['runner_observed_sha256']}` |",
        "",
        "## Independently reconstructed development selection",
        "",
        "| eta | raw score | normalized score | env steps | Q1 rows | backward calls |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["development_selection"]["candidates"]:
        lines.append(
            "| "
            + " | ".join(
                format_value(value)
                for value in (
                    item["step_size"],
                    item["raw_score_mean_recomputed"],
                    item["normalized_score_mean_recomputed"],
                    item["environment_steps"],
                    item["q1_forward_rows"],
                    item["q1_backward_calls"],
                )
            )
            + " |"
        )
    lines += [
        "",
        f"Initial eta: `{report['development_selection']['initial_selected_step_size']}`; "
        f"endpoint extension: `{report['development_selection']['endpoint_extension_step_size']}`; "
        f"final eta: `{report['development_selection']['selected_step_size']}`.",
        "",
        "## Absolute holdout controller scores and recomputed cost",
        "",
        "| Controller | Checkpoint | eta | Baseline norm | Adapted norm | Delta | Adapted steps | Q1 rows | Backward calls | Adapted wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["controller_scores"]:
        lines.append(
            "| "
            + " | ".join(
                format_value(value)
                for value in (
                    item["controller_id"],
                    item["training_seed"],
                    item["step_size"],
                    item["baseline_normalized_score_mean"],
                    item["adapted_normalized_score_mean"],
                    item["adapted_minus_baseline_normalized_mean"],
                    item["adapted_environment_steps"],
                    item["q1_forward_rows_total"],
                    item["q1_backward_calls_total"],
                    item["adapted_wall_time_seconds"],
                )
            )
            + " |"
        )
    lines += [
        "",
        "All rows match one actor row, 16 Q1 forward/backward rows, and two "
        "backward calls per adapted decision. This is equal per-decision neural "
        "budget, not exact wall-time/FLOP equality: inverse anchoring adds 48 "
        "scalar bisection iterations, and total calls vary with episode horizon.",
        "",
        "## Paired holdout comparisons",
        "",
        "| Comparison | Checkpoint | mean normalized delta | t 95% | bootstrap 95% | positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["paired_comparisons"]:
        lines.append(
            f"| {format_value(item['comparison_id'])} | {item['training_seed']} | "
            f"{item['mean']:.6f} | [{item['t95_low']:.6f}, {item['t95_high']:.6f}] | "
            f"[{item['bootstrap95_low']:.6f}, {item['bootstrap95_high']:.6f}] | "
            f"{item['positive_episode_count']}/{item['n_episode_pairs']} |"
        )
    lines += [
        "",
        "Intervals use paired rollout cases within each fixed checkpoint. The two "
        "checkpoint means below are descriptive only; no training-seed interval, "
        "p-value, pooling to n=100, or multiple-comparison correction is claimed.",
        "",
        "## Cross-checkpoint descriptive comparisons",
        "",
        "| Comparison | Per-checkpoint means | Mean | Positive checkpoints |",
        "|---|---|---:|---:|",
    ]
    for item in report["cross_checkpoint_comparisons"]:
        values = ", ".join(f"{value:.6f}" for value in item["per_checkpoint_normalized_means"])
        lines.append(
            f"| {format_value(item['comparison_id'])} | {values} | "
            f"{item['descriptive_normalized_mean']:.6f} | "
            f"{item['positive_checkpoint_count']}/{item['n_checkpoint_seeds']} |"
        )
    lines += [
        "",
        "## Complete input hash inventory",
        "",
        "| Roles | Bytes | SHA-256 | Path |",
        "|---|---:|---|---|",
    ]
    for item in report["input_files"]:
        lines.append(
            f"| {format_value('; '.join(item['roles']))} | {item['size_bytes']} | "
            f"`{item['sha256']}` | `{format_value(item['path'])}` |"
        )
    return "\n".join(lines) + "\n"


def write_report_outputs(
    report: Mapping[str, Any], output_directory: Path, stem: str
) -> Dict[str, str]:
    targets = {
        "json": output_directory / f"{stem}.json",
        "csv": output_directory / f"{stem}.csv",
        "markdown": output_directory / f"{stem}.md",
    }
    for path in targets.values():
        require(
            not path.exists() and not path.with_name(path.name + ".tmp").exists(),
            f"refusing to overwrite output: {path}",
        )
    payloads = {
        "json": json_text(report),
        "csv": render_csv(report),
        "markdown": render_markdown(report),
    }
    for key, path in targets.items():
        write_new(path, payloads[key])
    return {key: str(path.resolve()) for key, path in targets.items()}


def main() -> int:
    source = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=source.with_name("equal_compute_nominal_control_protocol.json"),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=source.with_name("run_equal_compute_nominal_control.py"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stem",
        default="equal_compute_nominal_control_verified_v2",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(
            args.protocol.resolve(),
            args.runner.resolve(),
            source,
        )
        if args.check_only:
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "selected_nominal_step_size": report["development_selection"]["selected_step_size"],
                        "controller_score_rows": len(report["controller_scores"]),
                        "paired_comparisons": len(report["paired_comparisons"]),
                        "input_file_count": report["input_file_count"],
                        "outputs_written": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        output_directory = args.output_dir
        if output_directory is None:
            protocol = read_json(args.protocol.resolve())
            output_directory = (
                Path(protocol["paths"]["output_root"]) / "aggregate_v2"
            )
        outputs = write_report_outputs(report, output_directory.resolve(), args.stem)
        print(json.dumps({"status": report["status"], "outputs": outputs}, indent=2, sort_keys=True))
        return 0
    except VerificationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
