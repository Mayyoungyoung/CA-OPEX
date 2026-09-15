"""Fail-closed aggregation for one Hopper-v4 CA-OPEX cross-task seed.

The run directory must contain one base checkpoint/config pair and five raw
controller records.  Every score and paired statistic is reconstructed from
the per-episode return arrays; cached means in the raw JSON are deliberately
not used.  The output is a create-only JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "ca-opex-hopper-crosstask-aggregate-v1"
ENVIRONMENT = "Hopper-v4"
EPISODES = 50
ROLLOUT_BETA = 1.25
REFERENCE_MIN = -20.272305
REFERENCE_MAX = 3234.3
CALIBRATED_BETA = 1.2498948872089386
EXPECTED_CALIBRATION_SHA256 = "d7f273b39ae935bb31e62d12f41861ca216f89c8c5086899f355b6ebfd7941f0"
CALIBRATION_SOURCE = "censored_uniform_plus_clip_pair_calibration"
NOMINAL_SOURCE = "cli_known_beta_without_pair_calibration"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED_ROOT = 20_260_915
RESERVED_TRAINING_SEEDS = frozenset({0})
RESERVED_SEED_BLOCKS = {
    "environment": (
        (39_300, 39_349),
        (79_300, 79_349),
        (131_300, 131_349),
        (131_400, 131_449),
        (131_500, 131_549),
    ),
    "action_noise": (
        (49_300, 49_349),
        (89_300, 89_349),
        (231_300, 231_349),
        (231_400, 231_449),
        (231_500, 231_549),
    ),
    "gradient_noise": (
        (69_300, 69_349),
        (99_300, 99_349),
        (331_300, 331_349),
        (331_400, 331_449),
        (331_500, 331_549),
    ),
}

IMPLEMENTATION_FILES = {
    "evaluate_sha256": "evaluate_channel_opex.py",
    "inverse_residual_core_sha256": "inverse_residual_core.py",
    "td3bc_core_sha256": "td3bc_core.py",
    "train_td3bc_sha256": "train_td3bc.py",
    "evaluation_controls_sha256": "evaluation_controls.py",
    "train_inverse_residual_adapter_sha256": "train_inverse_residual_adapter.py",
}
SCALEUP_DRIVER = "run_ca_opex_hopper_crosstask.sh"

CONTROL_PATHS = {
    "complete": "controls/complete.json",
    "calibrated_identity": "controls/calibrated_identity.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
    "original_opex": "controls/original_opex.json",
    "inverse_only": "controls/inverse_only.json",
}

CHANNEL_CONTRACTS = {
    "complete": {
        "method_id": "channel_aware_opex_inverse_anchor",
        "calibrated": True,
        "baseline_transform": "inverse",
        "K": 8,
        "T": 2,
        "step_size": 0.1,
        "delta_max": 0.25,
    },
    "calibrated_identity": {
        "method_id": "channel_aware_opex_identity_anchor",
        "calibrated": True,
        "baseline_transform": "identity",
        "K": 8,
        "T": 2,
        "step_size": 0.3,
        "delta_max": 2.0,
    },
    "nominal_k8t2": {
        "method_id": "channel_aware_opex_identity_anchor",
        "calibrated": False,
        "baseline_transform": "identity",
        "K": 8,
        "T": 2,
        "step_size": 0.1,
        "delta_max": 2.0,
    },
    "original_opex": {
        "method_id": "original_structure_opex_t1",
        "calibrated": False,
        "baseline_transform": "identity",
        "K": 1,
        "T": 1,
        "step_size": 0.1,
        "delta_max": 2.0,
    },
}

COMPARISONS = (
    ("complete_minus_calibrated_identity", "calibrated_identity"),
    ("complete_minus_nominal_k8t2", "nominal_k8t2"),
    ("complete_minus_original_opex", "original_opex"),
    ("complete_minus_inverse_only", "inverse_only"),
)


class AggregationError(RuntimeError):
    """Raised when an input violates the scale-up contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AggregationError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_constant(token: str) -> None:
    raise AggregationError(f"non-finite JSON constant is forbidden: {token}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AggregationError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except AggregationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"JSON root is not an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{label} must be an object")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregationError(f"{label} must be an integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise AggregationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise AggregationError(f"{label} must be finite")
    return result


def _same_float(actual: object, expected: float, label: str) -> None:
    observed = _finite(actual, label)
    if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise AggregationError(f"{label} mismatch: {observed!r} != {expected!r}")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AggregationError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AggregationError(f"{label} is not a SHA-256 digest") from exc
    return value.lower()


def _int_vector(value: object, label: str, *, positive: bool = False) -> list[int]:
    if not isinstance(value, list) or len(value) != EPISODES:
        raise AggregationError(f"{label} must contain exactly {EPISODES} integers")
    result = [_integer(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if positive and any(item <= 0 for item in result):
        raise AggregationError(f"{label} values must be positive")
    return result


def _returns(value: object, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != EPISODES:
        raise AggregationError(f"{label} must contain exactly {EPISODES} returns")
    result = np.asarray(
        [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)],
        dtype=np.float64,
    )
    if result.shape != (EPISODES,) or not np.isfinite(result).all():
        raise AggregationError(f"{label} contains invalid returns")
    return result


def _require_consecutive(values: Sequence[int], start: int, label: str) -> None:
    expected = list(range(start, start + EPISODES))
    if list(values) != expected:
        raise AggregationError(f"{label} does not match its consecutive seed block")


def _optional_train_seed(raw: Mapping[str, Any], expected: int, label: str) -> None:
    candidates: list[tuple[str, object]] = []
    if "training_seed" in raw:
        candidates.append(("training_seed", raw["training_seed"]))
    base = raw.get("base_checkpoint")
    if isinstance(base, Mapping) and "training_seed" in base:
        candidates.append(("base_checkpoint.training_seed", base["training_seed"]))
    for suffix, value in candidates:
        if _integer(value, f"{label}.{suffix}") != expected:
            raise AggregationError(f"{label}.{suffix} does not match base/config.json")


def _validate_base_config(config: Mapping[str, Any]) -> int:
    arguments = _mapping(config.get("arguments"), "base.config.arguments")
    train_seed = _integer(
        arguments.get("train_seed"), "base.config.arguments.train_seed"
    )
    if train_seed < 0:
        raise AggregationError("base training seed must be non-negative")
    if train_seed in RESERVED_TRAINING_SEEDS:
        raise AggregationError(
            f"training seed {train_seed} is reserved for development"
        )
    expected = {
        "variant": "hubl_constant",
        "action_pairing": "executed_executed",
        "updates": 25_000,
        "env_name": ENVIRONMENT,
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise AggregationError(f"base.config.arguments.{key} contract mismatch")
    _same_float(
        arguments.get("heuristic_discount"),
        0.391843318939209,
        "base.config.arguments.heuristic_discount",
    )
    channels = config.get("action_channels")
    if (
        isinstance(channels, Mapping)
        and channels.get("action_pairing") != "executed_executed"
    ):
        raise AggregationError("base.config.action_channels.action_pairing mismatch")
    return train_seed


def _validate_channel_raw(
    control_id: str,
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    training_seed: int,
) -> Dict[str, Any]:
    contract = CHANNEL_CONTRACTS[control_id]
    label = f"controls.{control_id}"
    if raw.get("raw_schema") != "channel_opex_v1" or raw.get("status") != "complete":
        raise AggregationError(f"{label} is not a complete channel_opex_v1 record")
    if (
        raw.get("environment") != ENVIRONMENT
        or raw.get("method_id") != contract["method_id"]
    ):
        raise AggregationError(f"{label} environment/method contract mismatch")
    _optional_train_seed(raw, training_seed, label)

    base = _mapping(raw.get("base_checkpoint"), f"{label}.base_checkpoint")
    if (
        _digest(base.get("sha256"), f"{label}.base_checkpoint.sha256")
        != checkpoint_sha256
    ):
        raise AggregationError(f"{label} checkpoint SHA does not match base/latest.pt")
    if _integer(base.get("step"), f"{label}.base_checkpoint.step") != 25_000:
        raise AggregationError(f"{label} checkpoint step mismatch")

    normalization = _mapping(raw.get("normalization"), f"{label}.normalization")
    _same_float(
        normalization.get("reference_min_score"),
        REFERENCE_MIN,
        f"{label}.reference_min",
    )
    _same_float(
        normalization.get("reference_max_score"),
        REFERENCE_MAX,
        f"{label}.reference_max",
    )

    controller = _mapping(raw.get("controller"), f"{label}.controller")
    expected_controller = {
        "baseline_transform": contract["baseline_transform"],
        "K": contract["K"],
        "execution_noise_samples": contract["K"],
        "gradient_steps": contract["T"],
        "gradient_steps_per_action": contract["T"],
        "q_reducer": "mean_q1",
        "critic": "frozen_q1",
        "actor_parameter_updates": 0,
        "critic_parameter_updates": 0,
    }
    for key, expected in expected_controller.items():
        if controller.get(key) != expected:
            raise AggregationError(f"{label}.controller.{key} contract mismatch")
    _same_float(
        controller.get("step_size"), contract["step_size"], f"{label}.step_size"
    )
    _same_float(
        controller.get("delta_max"), contract["delta_max"], f"{label}.delta_max"
    )

    calibrated = bool(contract["calibrated"])
    expected_beta = CALIBRATED_BETA if calibrated else 0.0
    _same_float(controller.get("model_beta"), expected_beta, f"{label}.model_beta")
    calibration = _mapping(raw.get("calibration"), f"{label}.calibration")
    if calibrated:
        calibration_sha = _digest(
            calibration.get("calibration_sha256"), f"{label}.calibration.sha256"
        )
        if calibration_sha != EXPECTED_CALIBRATION_SHA256:
            raise AggregationError(
                f"{label} calibration SHA does not match the frozen calibration"
            )
        calibration_path_value = calibration.get("calibration_path")
        if not isinstance(calibration_path_value, str) or not calibration_path_value:
            raise AggregationError(
                f"{label}.calibration.calibration_path must be a path"
            )
        calibration_path = Path(calibration_path_value)
        if not calibration_path.is_file():
            raise AggregationError(
                f"{label} calibration path is not an existing file: {calibration_path}"
            )
        try:
            calibration_path = calibration_path.resolve(strict=True)
        except OSError as exc:
            raise AggregationError(
                f"cannot resolve {label} calibration path: {exc}"
            ) from exc
        if sha256_file(calibration_path) != EXPECTED_CALIBRATION_SHA256:
            raise AggregationError(
                f"{label} calibration file does not match its frozen SHA"
            )
        if calibration.get("source") != CALIBRATION_SOURCE:
            raise AggregationError(f"{label} calibration source mismatch")
        if (
            _integer(calibration.get("pair_count"), f"{label}.calibration.pair_count")
            != 512
        ):
            raise AggregationError(f"{label} calibration pair count mismatch")
        _same_float(
            calibration.get("beta"), CALIBRATED_BETA, f"{label}.calibration.beta"
        )
        if controller.get("calibration_mode") != CALIBRATION_SOURCE:
            raise AggregationError(f"{label} calibration mode mismatch")
    else:
        calibration_sha = None
        calibration_path = None
        if calibration.get("calibration_sha256") is not None:
            raise AggregationError(
                f"{label} nominal controller unexpectedly uses calibration"
            )
        if calibration.get("source") != NOMINAL_SOURCE:
            raise AggregationError(f"{label} nominal beta source mismatch")
        _same_float(calibration.get("beta"), 0.0, f"{label}.calibration.beta")
        if controller.get("calibration_mode") != NOMINAL_SOURCE:
            raise AggregationError(f"{label} nominal calibration mode mismatch")

    channel = _mapping(raw.get("channel"), f"{label}.channel")
    if channel.get("gradient_model") != "iid_uniform_additive_then_clip":
        raise AggregationError(f"{label} gradient channel mismatch")
    _same_float(
        channel.get("environment_rollout_beta"), ROLLOUT_BETA, f"{label}.rollout_beta"
    )
    _same_float(
        channel.get("model_or_calibration_beta"), expected_beta, f"{label}.channel_beta"
    )

    protocol = _mapping(raw.get("evaluation_protocol"), f"{label}.evaluation_protocol")
    if protocol.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} evaluation environment mismatch")
    if (
        _integer(protocol.get("episode_count_per_arm"), f"{label}.episode_count")
        != EPISODES
    ):
        raise AggregationError(f"{label} episode count mismatch")
    if protocol.get("paired_environment_and_action_noise_seeds") is not True:
        raise AggregationError(f"{label} does not declare paired rollout seeds")
    env_start = _integer(
        protocol.get("environment_seed_start"), f"{label}.env_seed_start"
    )
    noise_start = _integer(
        protocol.get("action_noise_seed_start"), f"{label}.noise_seed_start"
    )
    gradient_start = _integer(
        protocol.get("gradient_noise_seed_start"), f"{label}.gradient_seed_start"
    )

    arms = _mapping(raw.get("arms"), f"{label}.arms")
    arm = _mapping(arms.get("adapted"), f"{label}.arms.adapted")
    env_seeds = _int_vector(arm.get("environment_seeds"), f"{label}.environment_seeds")
    noise_seeds = _int_vector(
        arm.get("action_noise_seeds"), f"{label}.action_noise_seeds"
    )
    gradient_seeds = _int_vector(
        arm.get("gradient_noise_seeds"), f"{label}.gradient_noise_seeds"
    )
    _require_consecutive(env_seeds, env_start, f"{label}.environment_seeds")
    _require_consecutive(noise_seeds, noise_start, f"{label}.action_noise_seeds")
    _require_consecutive(
        gradient_seeds, gradient_start, f"{label}.gradient_noise_seeds"
    )
    returns = _returns(arm.get("returns"), f"{label}.returns")
    lengths = _int_vector(arm.get("lengths"), f"{label}.lengths", positive=True)
    _same_float(arm.get("action_noise_beta"), ROLLOUT_BETA, f"{label}.arm_action_beta")

    K = int(contract["K"])
    T = int(contract["T"])
    environment_steps = int(sum(lengths))
    q_rows = environment_steps * K * T
    backward_calls = environment_steps * T
    if _integer(arm.get("q1_rows_per_action"), f"{label}.q1_rows_per_action") != K * T:
        raise AggregationError(f"{label} per-action Q-row cost mismatch")
    if _integer(arm.get("q1_rows_total"), f"{label}.q1_rows_total") != q_rows:
        raise AggregationError(f"{label} total Q-row cost mismatch")
    if (
        _integer(arm.get("q1_gradient_calls"), f"{label}.q1_gradient_calls")
        != backward_calls
    ):
        raise AggregationError(f"{label} backward-call cost mismatch")
    rows_by_episode = arm.get("q1_rows_by_episode")
    if rows_by_episode is not None:
        expected_rows = [length * K * T for length in lengths]
        if rows_by_episode != expected_rows:
            raise AggregationError(f"{label}.q1_rows_by_episode mismatch")

    cost = _mapping(raw.get("cost"), f"{label}.cost")
    expected_cost = {
        "environment_steps": environment_steps,
        "base_actor_rows": environment_steps,
        "q1_forward_rows": q_rows,
        "q1_backward_rows": q_rows,
        "q1_backward_calls": backward_calls,
        "adapted_q1_rows_per_environment_step": K * T,
        "adapted_backward_calls": backward_calls,
    }
    for key, expected in expected_cost.items():
        if _integer(cost.get(key), f"{label}.cost.{key}") != expected:
            raise AggregationError(f"{label}.cost.{key} mismatch")
    wall_time = _finite(
        cost.get("wall_time_seconds"), f"{label}.cost.wall_time_seconds"
    )
    if wall_time < 0.0:
        raise AggregationError(f"{label} wall time must be non-negative")

    return {
        "returns": returns,
        "lengths": lengths,
        "environment_seeds": env_seeds,
        "action_noise_seeds": noise_seeds,
        "gradient_noise_seeds": gradient_seeds,
        "calibration_sha256": calibration_sha,
        "calibration_path": str(calibration_path)
        if calibration_path is not None
        else None,
        "cost": {
            "base_actor_rows_per_decision": 1,
            "q1_forward_rows_per_decision": K * T,
            "q1_backward_rows_per_decision": K * T,
            "q1_backward_calls_per_decision": T,
            "inverse_bisection_iterations_per_decision": 48
            if control_id == "complete"
            else 0,
            "environment_steps": environment_steps,
            "q1_forward_rows_total": q_rows,
            "q1_backward_rows_total": q_rows,
            "q1_backward_calls_total": backward_calls,
            "wall_time_seconds": wall_time,
            "wall_time_scope": "adapted_arm_deployment_controller_only",
        },
        "contract": {
            "baseline_transform": contract["baseline_transform"],
            "model_beta": expected_beta,
            "K": K,
            "T": T,
            "step_size": contract["step_size"],
            "delta_max": contract["delta_max"],
        },
    }


def _expected_implementation(code_dir: Path) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    for raw_key, filename in IMPLEMENTATION_FILES.items():
        path = code_dir / filename
        if not path.is_file():
            raise AggregationError(f"required implementation source is missing: {path}")
        expected[raw_key] = sha256_file(path)
    driver = code_dir / SCALEUP_DRIVER
    if not driver.is_file():
        raise AggregationError(f"scale-up driver is missing: {driver}")
    return expected


def _validate_implementation_maps(
    raw_payloads: Mapping[str, Mapping[str, Any]],
    *,
    expected: Mapping[str, str],
) -> None:
    reference: Mapping[str, Any] | None = None
    for control_id in CHANNEL_CONTRACTS:
        implementation = _mapping(
            raw_payloads[control_id].get("implementation"),
            f"controls.{control_id}.implementation",
        )
        if reference is None:
            reference = implementation
        elif implementation != reference:
            raise AggregationError(
                "four channel raw implementation maps are not identical"
            )
        if dict(implementation) != dict(expected):
            raise AggregationError(
                f"controls.{control_id}.implementation does not match live code hashes"
            )


def _validate_unreserved_seed_block(values: Sequence[int], namespace: str) -> None:
    if any(value < 0 for value in values):
        raise AggregationError(f"{namespace} seeds must be non-negative")
    observed = set(values)
    for low, high in RESERVED_SEED_BLOCKS[namespace]:
        if observed.intersection(range(low, high + 1)):
            raise AggregationError(
                f"{namespace} 50-case block overlaps reserved [{low}, {high}]"
            )


def _validate_inverse_raw(
    raw: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    training_seed: int,
) -> Dict[str, Any]:
    label = "controls.inverse_only"
    if raw.get("status") != "complete" or raw.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} is not a complete {ENVIRONMENT} record")
    _optional_train_seed(raw, training_seed, label)
    if (
        _digest(raw.get("checkpoint_sha256"), f"{label}.checkpoint_sha256")
        != checkpoint_sha256
    ):
        raise AggregationError(f"{label} checkpoint SHA does not match base/latest.pt")
    if _integer(raw.get("checkpoint_step"), f"{label}.checkpoint_step") != 25_000:
        raise AggregationError(f"{label} checkpoint step mismatch")
    _same_float(raw.get("command_scale"), 1.0, f"{label}.command_scale")
    if raw.get("command_transform") != "uniform_mean_inverse":
        raise AggregationError(f"{label}.command_transform mismatch")
    _same_float(
        raw.get("command_transform_beta"), ROLLOUT_BETA, f"{label}.transform_beta"
    )

    arm = _mapping(
        raw.get("persistent_action_noise"), f"{label}.persistent_action_noise"
    )
    if (
        arm.get("action_noise_distribution")
        != "iid_uniform_minus_beta_plus_beta_per_step"
    ):
        raise AggregationError(f"{label} action-noise distribution mismatch")
    if arm.get("command_transform") != "uniform_mean_inverse":
        raise AggregationError(f"{label} arm transform mismatch")
    _same_float(arm.get("command_scale"), 1.0, f"{label}.arm_command_scale")
    _same_float(
        arm.get("command_transform_beta"), ROLLOUT_BETA, f"{label}.arm_transform_beta"
    )
    _same_float(
        arm.get("action_noise_beta"), ROLLOUT_BETA, f"{label}.action_noise_beta"
    )
    env_seeds = _int_vector(arm.get("environment_seeds"), f"{label}.environment_seeds")
    noise_seeds = _int_vector(
        arm.get("action_noise_seeds"), f"{label}.action_noise_seeds"
    )
    returns = _returns(arm.get("returns"), f"{label}.returns")
    lengths = _int_vector(arm.get("lengths"), f"{label}.lengths", positive=True)
    _require_consecutive(env_seeds, env_seeds[0], f"{label}.environment_seeds")
    _require_consecutive(noise_seeds, noise_seeds[0], f"{label}.action_noise_seeds")
    environment_steps = int(sum(lengths))
    wall_time = _finite(raw.get("wall_time_seconds"), f"{label}.wall_time_seconds")
    if wall_time < 0.0:
        raise AggregationError(f"{label} wall time must be non-negative")
    return {
        "returns": returns,
        "lengths": lengths,
        "environment_seeds": env_seeds,
        "action_noise_seeds": noise_seeds,
        "gradient_noise_seeds": None,
        "calibration_sha256": None,
        "cost": {
            "base_actor_rows_per_decision": 1,
            "q1_forward_rows_per_decision": 0,
            "q1_backward_rows_per_decision": 0,
            "q1_backward_calls_per_decision": 0,
            "inverse_bisection_iterations_per_decision": 48,
            "environment_steps": environment_steps,
            "q1_forward_rows_total": 0,
            "q1_backward_rows_total": 0,
            "q1_backward_calls_total": 0,
            "wall_time_seconds": wall_time,
            "wall_time_scope": "combined_clean_and_persistent_evaluation",
        },
        "contract": {
            "baseline_transform": "uniform_mean_inverse",
            "model_beta": ROLLOUT_BETA,
            "K": 0,
            "T": 0,
            "step_size": 0.0,
            "delta_max": 0.0,
        },
    }


def _score_record(
    control_id: str,
    parsed: Mapping[str, Any],
    *,
    raw_path: Path,
    raw_sha256: str,
) -> Dict[str, Any]:
    values = np.asarray(parsed["returns"], dtype=np.float64)
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)
    normalized = (values - REFERENCE_MIN) * factor
    return {
        "control_id": control_id,
        "episode_count": EPISODES,
        "raw_score_mean_recomputed": float(values.mean()),
        "raw_score_std_ddof0_recomputed": float(values.std(ddof=0)),
        "normalized_score_mean_recomputed": float(normalized.mean()),
        "normalized_score_std_ddof0_recomputed": float(normalized.std(ddof=0)),
        "environment_steps": int(sum(parsed["lengths"])),
        "raw_path": str(raw_path),
        "raw_sha256": raw_sha256,
        "contract": dict(parsed["contract"]),
    }


def paired_stats(values: np.ndarray, *, bootstrap_seed: int) -> Dict[str, Any]:
    if values.shape != (EPISODES,) or not np.isfinite(values).all():
        raise AggregationError("paired statistic input must contain 50 finite values")
    mean = float(values.mean())
    sample_std = float(values.std(ddof=1))
    standard_error = sample_std / math.sqrt(EPISODES)
    critical = float(student_t.ppf(0.975, EPISODES - 1))
    generator = np.random.Generator(np.random.PCG64(bootstrap_seed))
    indices = generator.integers(0, EPISODES, size=(BOOTSTRAP_REPLICATES, EPISODES))
    bootstrap_means = values[indices].mean(axis=1)
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrap_means, [0.025, 0.975], method="linear"
    ).tolist()
    return {
        "n_episode_pairs": EPISODES,
        "mean": mean,
        "sample_std_ddof1": sample_std,
        "standard_error": standard_error,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_bit_generator": "PCG64",
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_quantile_method": "linear",
        "bootstrap95_low": float(bootstrap_low),
        "bootstrap95_high": float(bootstrap_high),
        "positive_episode_count": int((values > 0.0).sum()),
        "zero_episode_count": int((values == 0.0).sum()),
        "negative_episode_count": int((values < 0.0).sum()),
    }


def _require_new_output(path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        existing = path if path.exists() else temporary
        raise AggregationError(f"refusing to overwrite output: {existing}")


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AggregationError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def aggregate(run_root: Path, output: Path) -> Dict[str, Any]:
    run_root = run_root.resolve()
    output = output.resolve()
    _require_new_output(output)
    base_checkpoint = run_root / "base/latest.pt"
    base_config_path = run_root / "base/config.json"
    if not base_checkpoint.is_file() or not base_config_path.is_file():
        raise AggregationError(
            "run root must contain base/latest.pt and base/config.json"
        )
    checkpoint_sha = sha256_file(base_checkpoint)
    base_config = read_json(base_config_path)
    training_seed = _validate_base_config(base_config)

    raw_paths = {key: run_root / relative for key, relative in CONTROL_PATHS.items()}
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise AggregationError(f"run root is missing required controls: {missing}")
    raw_payloads = {key: read_json(path) for key, path in raw_paths.items()}
    raw_hashes = {key: sha256_file(path) for key, path in raw_paths.items()}
    code_dir = Path(__file__).resolve().parent
    expected_implementation = _expected_implementation(code_dir)
    _validate_implementation_maps(
        raw_payloads,
        expected=expected_implementation,
    )

    parsed: Dict[str, Dict[str, Any]] = {}
    for control_id in CHANNEL_CONTRACTS:
        parsed[control_id] = _validate_channel_raw(
            control_id,
            raw_payloads[control_id],
            checkpoint_sha256=checkpoint_sha,
            training_seed=training_seed,
        )
    parsed["inverse_only"] = _validate_inverse_raw(
        raw_payloads["inverse_only"],
        checkpoint_sha256=checkpoint_sha,
        training_seed=training_seed,
    )

    calibration_shas = {
        parsed["complete"]["calibration_sha256"],
        parsed["calibrated_identity"]["calibration_sha256"],
    }
    if len(calibration_shas) != 1 or None in calibration_shas:
        raise AggregationError(
            "calibrated controllers do not share one calibration SHA"
        )
    calibration_paths = {
        parsed["complete"]["calibration_path"],
        parsed["calibrated_identity"]["calibration_path"],
    }
    if len(calibration_paths) != 1 or None in calibration_paths:
        raise AggregationError(
            "calibrated controllers do not share one calibration file"
        )
    calibration_path = Path(next(iter(calibration_paths)))

    reference_env = parsed["complete"]["environment_seeds"]
    reference_noise = parsed["complete"]["action_noise_seeds"]
    for control_id, item in parsed.items():
        if item["environment_seeds"] != reference_env:
            raise AggregationError(f"{control_id} environment seeds differ elementwise")
        if item["action_noise_seeds"] != reference_noise:
            raise AggregationError(
                f"{control_id} action-noise seeds differ elementwise"
            )
    reference_gradient = parsed["complete"]["gradient_noise_seeds"]
    for control_id in CHANNEL_CONTRACTS:
        if parsed[control_id]["gradient_noise_seeds"] != reference_gradient:
            raise AggregationError(
                f"{control_id} gradient-noise seeds differ elementwise"
            )
    _validate_unreserved_seed_block(reference_env, "environment")
    _validate_unreserved_seed_block(reference_noise, "action_noise")
    _validate_unreserved_seed_block(reference_gradient, "gradient_noise")

    scores = [
        _score_record(
            control_id,
            parsed[control_id],
            raw_path=raw_paths[control_id],
            raw_sha256=raw_hashes[control_id],
        )
        for control_id in CONTROL_PATHS
    ]
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)
    complete_returns = np.asarray(parsed["complete"]["returns"], dtype=np.float64)
    comparisons = []
    for comparison_index, (comparison_id, right_id) in enumerate(COMPARISONS):
        right_returns = np.asarray(parsed[right_id]["returns"], dtype=np.float64)
        raw_deltas = complete_returns - right_returns
        normalized_deltas = raw_deltas * factor
        bootstrap_seed = BOOTSTRAP_SEED_ROOT + 1000 * training_seed + comparison_index
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "left": "complete",
                "right": right_id,
                "raw_delta_mean_recomputed": float(raw_deltas.mean()),
                "normalized_delta_orientation": "complete_minus_control",
                **paired_stats(normalized_deltas, bootstrap_seed=bootstrap_seed),
            }
        )

    input_files = [
        {
            "role": "base_checkpoint",
            "path": str(base_checkpoint),
            "sha256": checkpoint_sha,
            "size_bytes": base_checkpoint.stat().st_size,
        },
        {
            "role": "base_config",
            "path": str(base_config_path),
            "sha256": sha256_file(base_config_path),
            "size_bytes": base_config_path.stat().st_size,
        },
    ]
    input_files.extend(
        {
            "role": f"control:{control_id}",
            "path": str(raw_paths[control_id]),
            "sha256": raw_hashes[control_id],
            "size_bytes": raw_paths[control_id].stat().st_size,
        }
        for control_id in CONTROL_PATHS
    )
    live_calibration_sha = sha256_file(calibration_path)
    if live_calibration_sha != EXPECTED_CALIBRATION_SHA256:
        raise AggregationError(
            f"calibration source changed during aggregation: {calibration_path}"
        )
    input_files.append(
        {
            "role": "channel_calibration",
            "path": str(calibration_path),
            "sha256": live_calibration_sha,
            "size_bytes": calibration_path.stat().st_size,
        }
    )
    for raw_key, filename in IMPLEMENTATION_FILES.items():
        implementation_path = code_dir / filename
        live_sha = sha256_file(implementation_path)
        if live_sha != expected_implementation[raw_key]:
            raise AggregationError(
                f"implementation source changed during aggregation: {implementation_path}"
            )
        input_files.append(
            {
                "role": f"implementation:{raw_key}",
                "path": str(implementation_path),
                "sha256": live_sha,
                "size_bytes": implementation_path.stat().st_size,
            }
        )
    scaleup_driver = code_dir / SCALEUP_DRIVER
    input_files.append(
        {
            "role": "scaleup_driver",
            "path": str(scaleup_driver),
            "sha256": sha256_file(scaleup_driver),
            "size_bytes": scaleup_driver.stat().st_size,
        }
    )
    source = Path(__file__).resolve()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete",
        "run_root": str(run_root),
        "training_seed": training_seed,
        "environment": ENVIRONMENT,
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
            "scores_recomputed_from_episode_returns": True,
            "cached_raw_means_used": False,
        },
        "evaluation_protocol": {
            "episode_count": EPISODES,
            "environment_seeds": reference_env,
            "action_noise_seeds": reference_noise,
            "channel_gradient_noise_seeds": reference_gradient,
            "rollout_action_noise_beta": ROLLOUT_BETA,
            "all_five_arms_elementwise_paired": True,
            "reserved_formal_and_supplemental_seed_blocks_excluded": True,
        },
        "checkpoint": {
            "path": str(base_checkpoint),
            "sha256": checkpoint_sha,
            "config_path": str(base_config_path),
            "config_sha256": sha256_file(base_config_path),
            "all_five_arms_same_sha256": True,
            "training_seed_source": "base/config.json arguments.train_seed",
        },
        "calibration_path": str(calibration_path),
        "calibration_sha256": next(iter(calibration_shas)),
        "frozen_calibration_sha256_contract_satisfied": True,
        "absolute_scores": scores,
        "paired_comparisons": comparisons,
        "per_decision_and_total_cost": [
            {"control_id": control_id, **parsed[control_id]["cost"]}
            for control_id in CONTROL_PATHS
        ],
        "cost_scope_note": (
            "Neural costs are reconstructed per adapted decision; total costs vary "
            "with episode length. Inverse transforms add 48 deterministic scalar "
            "bisection iterations and are not exact-FLOP matched."
        ),
        "statistics": {
            "unit": "paired_rollout_case_within_one_fixed_training_checkpoint",
            "t_interval_confidence": 0.95,
            "sample_standard_deviation_ddof": 1,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed_formula": "20260915 + 1000 * training_seed + comparison_index",
            "cross_training_seed_inference": None,
        },
        "input_files": input_files,
        "aggregator": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
    }
    write_json_new(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = aggregate(args.run_root, args.output)
    except AggregationError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
