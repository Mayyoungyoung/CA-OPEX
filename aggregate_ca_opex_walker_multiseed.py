"""Fail-closed five-training-seed aggregation for the Walker2d CA-OPEX expansion.

Combines the two pre-existing base-policy training checkpoints (seeds 1 and 10,
formal confirmation block 39300/49300/69300 plus the post-confirmation
supplemental nominal-Q audit block 79300/89300/99300) with the new scale-up
seeds 2/3/4 (results/scaleup/walker_seed<N>) into training-seed-level
statistics.  Every score and interval is recomputed from per-episode return
arrays in the raw records; cached means are never used.  The 250 rollout
episodes are never pooled: cross-seed inference uses one mean per independent
base-policy checkpoint (n=5 training seeds).  Output is create-only.
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

SCHEMA = "ca-opex-walker-multiseed-aggregate-v1"
ENVIRONMENT = "Walker2d-v4"
EPISODES = 50
ROLLOUT_BETA = 1.25
REFERENCE_MIN = 1.629008
REFERENCE_MAX = 4592.3
NORMALIZATION_SOURCE = (
    "D4RL Walker2d-medium-v2 random/expert reference pair applied to "
    "Walker2d-v4; comparable only within this package"
)
CALIBRATED_BETA = 1.2498948872089386
EXPECTED_CALIBRATION_SHA256 = (
    "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354"
)
CALIBRATION_SOURCE = "censored_uniform_plus_clip_pair_calibration"
NOMINAL_SOURCE = "cli_known_beta_without_pair_calibration"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED_ROOT = 3_026_091_500
SCALEUP_AGGREGATE_SCHEMA = "ca-opex-walker-scaleup-aggregate-v1"

TRAINING_SEEDS = (1, 2, 3, 4, 10)
SCALEUP_SEEDS = (2, 3, 4)
PINNED_BASE_CHECKPOINTS = {
    1: "22b47965695bfa312533741abaa1e17e3cbbc091f3c5e6ba7fc8953fbc8ca6ae",
    10: "eba1af040aa73823ceba28048e10d3af2eef54c59f09c1e0194f27aa0ed4944d",
}
PINNED_BASE_CONFIGS = {
    1: "caa9696f708e51da0ba1800cfb2b97fe09b8f2a936f47cc3f47c8b7b8d68b1e4",
    10: "159805c950779b0e3b341b29afd0cebb09626f7c1ea75edc56f8a35d2d7967c7",
}

FORMAL_BLOCK = {
    "environment": (39_300, 39_349),
    "action_noise": (49_300, 49_349),
    "gradient_noise": (69_300, 69_349),
}
SUPPLEMENTAL_BLOCK = {
    "environment": (79_300, 79_349),
    "action_noise": (89_300, 89_349),
    "gradient_noise": (99_300, 99_349),
}

# Hash-pinned formal confirmation records (block 39300/49300/69300).
FORMAL_INPUTS = {
    ("complete", 1): (
        "inverse_residual_confirm/external_controls/ca_opex_inverse_seed1_beta125_50.json",
        "e6d0a827528e3c600efb73141e2d6a37d2170e64fd6dcd73633306b8eccbafb1",
    ),
    ("complete", 10): (
        "inverse_residual_confirm/external_controls/ca_opex_inverse_seed10_beta125_50.json",
        "1814cd1c6975b3d5d6d887c21c4eb796664dcc0d20fe1ab8a5be1c248970fbb7",
    ),
    ("calibrated_identity", 1): (
        "inverse_residual_confirm/external_controls/ca_opex_identity_seed1_beta125_50.json",
        "9ab9450d55bc78dbf75a6116ba398f54a60478ccfea14d47f4e056a2b6745621",
    ),
    ("calibrated_identity", 10): (
        "inverse_residual_confirm/external_controls/ca_opex_identity_seed10_beta125_50.json",
        "ddd200fc27a80159f4e38564801965de185934059f2ebcdb9922711fe15c5731",
    ),
    ("original_opex", 1): (
        "inverse_residual_confirm/external_controls/opex_original_seed1_beta125_50.json",
        "ff4fbdd0b8697984f17a8fbd620b0c4b2d1d947c92939ee345780ae665b48eac",
    ),
    ("original_opex", 10): (
        "inverse_residual_confirm/external_controls/opex_original_seed10_beta125_50.json",
        "f46214051e2aeafacd9f63f7ceb024862d7b3235969f802e76c1907bbb1ccb80",
    ),
}
FORMAL_BASE_DIRS = {
    1: "inverse_residual_confirm/base_hubl_executed_25k_seed1",
    10: "inverse_residual_confirm/base_hubl_executed_25k_seed10",
}

# Hash-pinned post-confirmation supplemental audit records (block 79300/89300/99300).
SUPPLEMENTAL_INPUTS = {
    ("complete_supplemental", 1): (
        "equal_compute_nominal_control/holdout/complete_calibrated_inverse_eta_0.1_seed1_fresh50.json",
        "a962af60ee972428b67ac4ae91fdb619399ab04394aeebf37804570c12670cdc",
    ),
    ("complete_supplemental", 10): (
        "equal_compute_nominal_control/holdout/complete_calibrated_inverse_eta_0.1_seed10_fresh50.json",
        "553b7ff3bfca058be2f15684222fa4a2a101357c5f8e04e503596089fbae3dd6",
    ),
    ("nominal_k8t2", 1): (
        "equal_compute_nominal_control/holdout/nominal_tuned_seed1_fresh50.json",
        "8ababfa7389b67ebfd0fb4827e799880b97e71be2e08b7c15b1811018f299f40",
    ),
    ("nominal_k8t2", 10): (
        "equal_compute_nominal_control/holdout/nominal_tuned_seed10_fresh50.json",
        "4f35cedf07cd7af5ebdbcb4bdd7d8a69a6eab46fb77b0a62ea50987fe914a1b3",
    ),
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

SCALEUP_CONTROL_FILES = {
    "complete": "controls/complete.json",
    "calibrated_identity": "controls/calibrated_identity.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
    "original_opex": "controls/original_opex.json",
    "inverse_only": "controls/inverse_only.json",
}

# (comparison_id, right_arm).  The left arm is always a complete CA-OPEX record;
# for seeds 1/10 the nominal-Q comparison pairs on the supplemental block.
COMPARISONS = (
    ("complete_minus_calibrated_identity", "calibrated_identity"),
    ("complete_minus_original_opex", "original_opex"),
    ("complete_minus_inverse_only", "inverse_only"),
    ("complete_minus_nominal_k8t2", "nominal_k8t2"),
    ("complete_minus_identity_command", "identity_command"),
)

PER_DECISION_COST = {
    "complete": {
        "base_actor_rows": 1,
        "q1_forward_rows": 16,
        "q1_backward_rows": 16,
        "q1_backward_calls": 2,
        "inverse_bisection_iterations": 48,
    },
    "calibrated_identity": {
        "base_actor_rows": 1,
        "q1_forward_rows": 16,
        "q1_backward_rows": 16,
        "q1_backward_calls": 2,
        "inverse_bisection_iterations": 0,
    },
    "nominal_k8t2": {
        "base_actor_rows": 1,
        "q1_forward_rows": 16,
        "q1_backward_rows": 16,
        "q1_backward_calls": 2,
        "inverse_bisection_iterations": 0,
    },
    "original_opex": {
        "base_actor_rows": 1,
        "q1_forward_rows": 1,
        "q1_backward_rows": 1,
        "q1_backward_calls": 1,
        "inverse_bisection_iterations": 0,
    },
    "inverse_only": {
        "base_actor_rows": 1,
        "q1_forward_rows": 0,
        "q1_backward_rows": 0,
        "q1_backward_calls": 0,
        "inverse_bisection_iterations": 48,
    },
    "identity_command": {
        "base_actor_rows": 1,
        "q1_forward_rows": 0,
        "q1_backward_rows": 0,
        "q1_backward_calls": 0,
        "inverse_bisection_iterations": 0,
    },
}


class AggregationError(RuntimeError):
    """Raised when an input violates the multi-seed contract."""


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


def _int_vector(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != EPISODES:
        raise AggregationError(f"{label} must contain exactly {EPISODES} integers")
    return [_integer(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _require_block(values: Sequence[int], block: tuple[int, int], label: str) -> None:
    low, high = block
    expected = list(range(low, high + 1))
    if list(values) != expected:
        raise AggregationError(f"{label} does not match its declared block [{low},{high}]")


def _load_channel_record(
    path: Path,
    *,
    expected_sha256: str,
    control_id: str,
    checkpoint_sha256: str,
    block: Mapping[str, tuple[int, int]],
    label: str,
    require_baseline_arm: bool,
) -> Dict[str, Any]:
    """Validate one hash-pinned channel_opex_v1 record and return raw arrays."""

    if not path.is_file():
        raise AggregationError(f"{label} input is missing: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha256:
        raise AggregationError(
            f"{label} SHA mismatch: {observed_sha} != pinned {expected_sha256}"
        )
    raw = read_json(path)
    contract = CHANNEL_CONTRACTS[control_id]
    if raw.get("raw_schema") != "channel_opex_v1" or raw.get("status") != "complete":
        raise AggregationError(f"{label} is not a complete channel_opex_v1 record")
    if raw.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} environment mismatch")
    if raw.get("method_id") != contract["method_id"]:
        raise AggregationError(f"{label} method contract mismatch")

    base = _mapping(raw.get("base_checkpoint"), f"{label}.base_checkpoint")
    if (
        _digest(base.get("sha256"), f"{label}.base_checkpoint.sha256")
        != checkpoint_sha256
    ):
        raise AggregationError(f"{label} checkpoint SHA mismatch")
    if _integer(base.get("step"), f"{label}.base_checkpoint.step") != 25_000:
        raise AggregationError(f"{label} checkpoint step mismatch")

    controller = _mapping(raw.get("controller"), f"{label}.controller")
    for key, expected in (
        ("baseline_transform", contract["baseline_transform"]),
        ("K", contract["K"]),
        ("gradient_steps", contract["T"]),
    ):
        if controller.get(key) != expected:
            raise AggregationError(f"{label}.controller.{key} contract mismatch")
    _same_float(controller.get("step_size"), contract["step_size"], f"{label}.step_size")
    _same_float(controller.get("delta_max"), contract["delta_max"], f"{label}.delta_max")
    calibrated = bool(contract["calibrated"])
    expected_beta = CALIBRATED_BETA if calibrated else 0.0
    _same_float(controller.get("model_beta"), expected_beta, f"{label}.model_beta")
    calibration = _mapping(raw.get("calibration"), f"{label}.calibration")
    if calibrated:
        if (
            _digest(
                calibration.get("calibration_sha256"), f"{label}.calibration.sha256"
            )
            != EXPECTED_CALIBRATION_SHA256
        ):
            raise AggregationError(f"{label} calibration SHA mismatch")
        if calibration.get("source") != CALIBRATION_SOURCE:
            raise AggregationError(f"{label} calibration source mismatch")
    else:
        if calibration.get("calibration_sha256") is not None:
            raise AggregationError(f"{label} nominal controller unexpectedly calibrated")
        if calibration.get("source") != NOMINAL_SOURCE:
            raise AggregationError(f"{label} nominal beta source mismatch")
        _same_float(calibration.get("beta"), 0.0, f"{label}.nominal_beta")

    protocol = _mapping(raw.get("evaluation_protocol"), f"{label}.evaluation_protocol")
    if protocol.get("paired_environment_and_action_noise_seeds") is not True:
        raise AggregationError(f"{label} does not declare paired rollout seeds")

    arms = _mapping(raw.get("arms"), f"{label}.arms")
    adapted = _mapping(arms.get("adapted"), f"{label}.arms.adapted")
    env_seeds = _int_vector(adapted.get("environment_seeds"), f"{label}.env_seeds")
    noise_seeds = _int_vector(adapted.get("action_noise_seeds"), f"{label}.noise_seeds")
    gradient_seeds = _int_vector(
        adapted.get("gradient_noise_seeds"), f"{label}.gradient_seeds"
    )
    _require_block(env_seeds, block["environment"], f"{label}.env_seeds")
    _require_block(noise_seeds, block["action_noise"], f"{label}.noise_seeds")
    _require_block(gradient_seeds, block["gradient_noise"], f"{label}.gradient_seeds")
    adapted_returns = _returns(adapted.get("returns"), f"{label}.adapted.returns")
    adapted_lengths = _int_vector(adapted.get("lengths"), f"{label}.adapted.lengths")
    if any(value <= 0 for value in adapted_lengths):
        raise AggregationError(f"{label}.adapted.lengths must be positive")
    _same_float(adapted.get("action_noise_beta"), ROLLOUT_BETA, f"{label}.rollout_beta")

    baseline_returns = None
    baseline = arms.get("baseline_only")
    if require_baseline_arm or baseline is not None:
        baseline = _mapping(baseline, f"{label}.arms.baseline_only")
        base_env = _int_vector(
            baseline.get("environment_seeds"), f"{label}.baseline.env_seeds"
        )
        base_noise = _int_vector(
            baseline.get("action_noise_seeds"), f"{label}.baseline.noise_seeds"
        )
        if base_env != env_seeds or base_noise != noise_seeds:
            raise AggregationError(f"{label} baseline arm seeds differ elementwise")
        baseline_returns = _returns(
            baseline.get("returns"), f"{label}.baseline.returns"
        )

    cost = _mapping(raw.get("cost"), f"{label}.cost")
    wall_time = _finite(cost.get("wall_time_seconds"), f"{label}.wall_time_seconds")
    return {
        "adapted_returns": adapted_returns,
        "baseline_returns": baseline_returns,
        "adapted_lengths": adapted_lengths,
        "wall_time_seconds": wall_time,
        "checkpoint_sha256": checkpoint_sha256,
        "contract": dict(contract),
    }


def _load_scaleup_inverse_record(
    path: Path, *, checkpoint_sha256: str, env_block: tuple[int, int],
    noise_block: tuple[int, int], label: str,
) -> Dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"{label} input is missing: {path}")
    raw = read_json(path)
    if raw.get("status") != "complete" or raw.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} is not a complete {ENVIRONMENT} record")
    if (
        _digest(raw.get("checkpoint_sha256"), f"{label}.checkpoint_sha256")
        != checkpoint_sha256
    ):
        raise AggregationError(f"{label} checkpoint SHA mismatch")
    if _integer(raw.get("checkpoint_step"), f"{label}.checkpoint_step") != 25_000:
        raise AggregationError(f"{label} checkpoint step mismatch")
    if raw.get("command_transform") != "uniform_mean_inverse":
        raise AggregationError(f"{label} command transform mismatch")
    _same_float(raw.get("command_transform_beta"), ROLLOUT_BETA, f"{label}.transform_beta")
    arm = _mapping(
        raw.get("persistent_action_noise"), f"{label}.persistent_action_noise"
    )
    env_seeds = _int_vector(arm.get("environment_seeds"), f"{label}.env_seeds")
    noise_seeds = _int_vector(arm.get("action_noise_seeds"), f"{label}.noise_seeds")
    _require_block(env_seeds, env_block, f"{label}.env_seeds")
    _require_block(noise_seeds, noise_block, f"{label}.noise_seeds")
    returns = _returns(arm.get("returns"), f"{label}.returns")
    wall_time = _finite(raw.get("wall_time_seconds"), f"{label}.wall_time_seconds")
    return {
        "adapted_returns": returns,
        "baseline_returns": None,
        "wall_time_seconds": wall_time,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _load_scaleup_seed(
    results_root: Path, seed: int
) -> Dict[str, Any]:
    run_root = results_root / "scaleup" / f"walker_seed{seed}"
    label = f"scaleup.seed{seed}"
    aggregate_path = run_root / "aggregate.json"
    if not aggregate_path.is_file():
        raise AggregationError(f"{label} per-seed aggregate is missing: {aggregate_path}")
    per_seed = read_json(aggregate_path)
    if (
        per_seed.get("schema_version") != SCALEUP_AGGREGATE_SCHEMA
        or per_seed.get("status") != "complete"
    ):
        raise AggregationError(f"{label} per-seed aggregate is not a complete scaleup report")
    if per_seed.get("training_seed") != seed:
        raise AggregationError(f"{label} aggregate training seed mismatch")
    checkpoint_path = run_root / "base/latest.pt"
    if not checkpoint_path.is_file():
        raise AggregationError(f"{label} base checkpoint missing")
    checkpoint_sha = sha256_file(checkpoint_path)
    reported = _digest(
        _mapping(per_seed.get("checkpoint"), f"{label}.checkpoint").get("sha256"),
        f"{label}.checkpoint.sha256",
    )
    if reported != checkpoint_sha:
        raise AggregationError(f"{label} checkpoint changed after per-seed aggregation")
    inventory = {
        entry["role"]: entry
        for entry in per_seed.get("input_files", [])
        if isinstance(entry, Mapping) and "role" in entry
    }
    env_block = (
        per_seed["evaluation_protocol"]["environment_seeds"][0],
        per_seed["evaluation_protocol"]["environment_seeds"][-1],
    )
    noise_block = (
        per_seed["evaluation_protocol"]["action_noise_seeds"][0],
        per_seed["evaluation_protocol"]["action_noise_seeds"][-1],
    )
    for reserved in (FORMAL_BLOCK, SUPPLEMENTAL_BLOCK):
        if env_block == reserved["environment"] or noise_block == reserved["action_noise"]:
            raise AggregationError(f"{label} reuses a reserved seed block")

    parsed: Dict[str, Dict[str, Any]] = {}
    for control_id, relative in SCALEUP_CONTROL_FILES.items():
        role = f"control:{control_id}"
        entry = inventory.get(role)
        if entry is None:
            raise AggregationError(f"{label} aggregate inventory lacks {role}")
        path = run_root / relative
        live_sha = sha256_file(path)
        if live_sha != entry["sha256"]:
            raise AggregationError(f"{label} {control_id} raw file changed after aggregation")
        if control_id == "inverse_only":
            parsed[control_id] = _load_scaleup_inverse_record(
                path,
                checkpoint_sha256=checkpoint_sha,
                env_block=env_block,
                noise_block=noise_block,
                label=f"{label}.inverse_only",
            )
        else:
            gradient_block = (
                per_seed["evaluation_protocol"]["channel_gradient_noise_seeds"][0],
                per_seed["evaluation_protocol"]["channel_gradient_noise_seeds"][-1],
            )
            block = {
                "environment": env_block,
                "action_noise": noise_block,
                "gradient_noise": gradient_block,
            }
            parsed[control_id] = _load_channel_record(
                path,
                expected_sha256=live_sha,
                control_id=control_id,
                checkpoint_sha256=checkpoint_sha,
                block=block,
                label=f"{label}.{control_id}",
                require_baseline_arm=True,
            )

    summary_path = run_root / "base/summary.json"
    base_summary = read_json(summary_path) if summary_path.is_file() else None
    base_wall = (
        _finite(base_summary.get("wall_time_seconds"), f"{label}.base_wall_time")
        if base_summary is not None and "wall_time_seconds" in base_summary
        else None
    )
    return {
        "checkpoint_sha256": checkpoint_sha,
        "parsed": parsed,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "run_root": str(run_root),
        "env_block": env_block,
        "noise_block": noise_block,
        "base_training_wall_time_seconds": base_wall,
        "base_summary_path": str(summary_path) if base_summary is not None else None,
        "base_summary_sha256": (
            sha256_file(summary_path) if base_summary is not None else None
        ),
    }


def _paired_episode_stats(values: np.ndarray, *, bootstrap_seed: int) -> Dict[str, Any]:
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
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap95_low": float(bootstrap_low),
        "bootstrap95_high": float(bootstrap_high),
        "positive_episode_count": int((values > 0.0).sum()),
    }


def _cross_seed_stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1))
    standard_error = sample_std / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    return {
        "n_training_seeds": n,
        "mean": mean,
        "sample_std_ddof1": sample_std,
        "standard_error": standard_error,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "positive_training_seed_count": int((array > 0.0).sum()),
        "negative_training_seed_count": int((array < 0.0).sum()),
        "interval_crosses_zero": bool(
            mean - critical * standard_error <= 0.0 <= mean + critical * standard_error
        ),
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


def _arm_summary(values: np.ndarray) -> Dict[str, Any]:
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)
    normalized = (values - REFERENCE_MIN) * factor
    return {
        "raw_score_mean_recomputed": float(values.mean()),
        "raw_score_std_ddof0_recomputed": float(values.std(ddof=0)),
        "normalized_score_mean_recomputed": float(normalized.mean()),
        "normalized_score_std_ddof0_recomputed": float(normalized.std(ddof=0)),
    }


def aggregate(results_root: Path, output: Path) -> Dict[str, Any]:
    results_root = results_root.resolve()
    output = output.resolve()
    _require_new_output(output)
    if len(set(TRAINING_SEEDS)) != len(TRAINING_SEEDS):
        raise AggregationError("duplicate training seeds in configuration")

    arms: Dict[str, Dict[int, Dict[str, Any]]] = {
        control_id: {} for control_id in PER_DECISION_COST
    }
    arm_sources: Dict[str, Dict[int, str]] = {control_id: {} for control_id in PER_DECISION_COST}
    input_files: list[Dict[str, Any]] = []
    base_wall_times: Dict[int, Any] = {}
    rollout_wall_times: Dict[int, Dict[str, Any]] = {}

    for seed in (1, 10):
        checkpoint_sha = PINNED_BASE_CHECKPOINTS[seed]
        base_dir = results_root / FORMAL_BASE_DIRS[seed]
        base_summary_path = base_dir / "summary.json"
        base_wall = None
        if base_summary_path.is_file():
            base_summary = read_json(base_summary_path)
            if "wall_time_seconds" in base_summary:
                base_wall = _finite(
                    base_summary["wall_time_seconds"], f"base seed {seed} wall time"
                )
        base_wall_times[seed] = base_wall
        seed_rollouts: Dict[str, Any] = {}
        for (control_id, pinned_seed), (relative, pinned_sha) in FORMAL_INPUTS.items():
            if pinned_seed != seed:
                continue
            path = results_root / relative
            require_baseline = control_id in ("complete", "calibrated_identity")
            parsed = _load_channel_record(
                path,
                expected_sha256=pinned_sha,
                control_id=control_id,
                checkpoint_sha256=checkpoint_sha,
                block=FORMAL_BLOCK,
                label=f"formal.{control_id}.seed{seed}",
                require_baseline_arm=require_baseline,
            )
            arms[control_id][seed] = parsed
            arm_sources[control_id][seed] = "formal_confirmation_block_39300"
            seed_rollouts[control_id] = parsed["wall_time_seconds"]
            input_files.append(
                {
                    "role": f"formal:{control_id}:seed{seed}",
                    "path": str(path),
                    "sha256": pinned_sha,
                    "size_bytes": path.stat().st_size,
                }
            )
        arms["inverse_only"][seed] = {
            "adapted_returns": arms["complete"][seed]["baseline_returns"],
            "baseline_returns": None,
            "wall_time_seconds": None,
            "checkpoint_sha256": checkpoint_sha,
            "source_note": "bit-matched baseline arm of the formal complete record",
        }
        arm_sources["inverse_only"][seed] = "formal_complete_baseline_arm_39300"
        arms["identity_command"][seed] = {
            "adapted_returns": arms["calibrated_identity"][seed]["baseline_returns"],
            "baseline_returns": None,
            "wall_time_seconds": None,
            "checkpoint_sha256": checkpoint_sha,
            "source_note": "baseline arm of the formal identity-anchor record",
        }
        arm_sources["identity_command"][seed] = "formal_identity_baseline_arm_39300"
        for (control_id, pinned_seed), (relative, pinned_sha) in SUPPLEMENTAL_INPUTS.items():
            if pinned_seed != seed:
                continue
            path = results_root / relative
            effective_control = (
                "complete" if control_id == "complete_supplemental" else control_id
            )
            parsed = _load_channel_record(
                path,
                expected_sha256=pinned_sha,
                control_id=effective_control,
                checkpoint_sha256=checkpoint_sha,
                block=SUPPLEMENTAL_BLOCK,
                label=f"supplemental.{effective_control}.seed{seed}",
                require_baseline_arm=False,
            )
            if control_id == "complete_supplemental":
                arms["complete_supplemental"] = arms.get("complete_supplemental", {})
                arms["complete_supplemental"][seed] = parsed
                arm_sources.setdefault("complete_supplemental", {})[seed] = (
                    "post_confirmation_supplemental_block_79300"
                )
            else:
                arms[control_id][seed] = parsed
                arm_sources[control_id][seed] = (
                    "post_confirmation_supplemental_block_79300"
                )
            seed_rollouts[control_id] = parsed["wall_time_seconds"]
            input_files.append(
                {
                    "role": f"supplemental:{effective_control}:seed{seed}",
                    "path": str(path),
                    "sha256": pinned_sha,
                    "size_bytes": path.stat().st_size,
                }
            )
        rollout_wall_times[seed] = seed_rollouts

    scaleup_info: Dict[int, Dict[str, Any]] = {}
    observed_scaleup_blocks: Dict[int, tuple[int, int]] = {}
    for seed in SCALEUP_SEEDS:
        info = _load_scaleup_seed(results_root, seed)
        for other_seed, other_block in observed_scaleup_blocks.items():
            if info["env_block"] == other_block:
                raise AggregationError(
                    f"scaleup seeds {other_seed} and {seed} share environment block "
                    f"{info['env_block']}"
                )
        observed_scaleup_blocks[seed] = info["env_block"]
        scaleup_info[seed] = info
        parsed = info["parsed"]
        for control_id, item in parsed.items():
            arms[control_id][seed] = item
            arm_sources[control_id][seed] = (
                f"scaleup_block_{info['env_block'][0]}"
            )
        arms["identity_command"][seed] = {
            "adapted_returns": parsed["calibrated_identity"]["baseline_returns"],
            "baseline_returns": None,
            "wall_time_seconds": None,
            "checkpoint_sha256": info["checkpoint_sha256"],
            "source_note": "baseline arm of the scale-up identity-anchor record",
        }
        arm_sources["identity_command"][seed] = f"scaleup_block_{info['env_block'][0]}"
        base_wall_times[seed] = info["base_training_wall_time_seconds"]
        rollout_wall_times[seed] = {
            control_id: item["wall_time_seconds"]
            for control_id, item in parsed.items()
        }
        input_files.append(
            {
                "role": f"scaleup:aggregate:seed{seed}",
                "path": info["aggregate_path"],
                "sha256": info["aggregate_sha256"],
                "size_bytes": Path(info["aggregate_path"]).stat().st_size,
            }
        )
        input_files.append(
            {
                "role": f"scaleup:checkpoint:seed{seed}",
                "path": str(Path(info["run_root"]) / "base/latest.pt"),
                "sha256": info["checkpoint_sha256"],
                "size_bytes": Path(info["run_root"], "base/latest.pt").stat().st_size,
            }
        )

    for control_id in ("complete", "calibrated_identity", "nominal_k8t2",
                       "original_opex", "inverse_only"):
        missing = [seed for seed in TRAINING_SEEDS if seed not in arms[control_id]]
        if missing:
            raise AggregationError(f"arm {control_id} is missing seeds {missing}")

    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)
    per_seed_controller_table = []
    for control_id in ("complete", "complete_supplemental", "calibrated_identity",
                       "nominal_k8t2", "original_opex", "inverse_only",
                       "identity_command"):
        seeds_present = sorted(arms.get(control_id, {}), key=lambda s: (s,))
        if not seeds_present:
            continue
        rows = {}
        for seed in seeds_present:
            values = arms[control_id][seed]["adapted_returns"]
            rows[str(seed)] = {
                **_arm_summary(values),
                "source_block": arm_sources[control_id][seed],
            }
        normalized_means = [
            _arm_summary(arms[control_id][seed]["adapted_returns"])[
                "normalized_score_mean_recomputed"
            ]
            for seed in seeds_present
        ]
        per_seed_controller_table.append(
            {
                "control_id": control_id,
                "per_seed": rows,
                "cross_seed_descriptive": _cross_seed_stats(normalized_means)
                if len(seeds_present) > 1
                else None,
            }
        )

    comparisons = []
    for comparison_index, (comparison_id, right_id) in enumerate(COMPARISONS):
        per_seed_rows = []
        normalized_delta_means = []
        raw_delta_means = []
        for seed in TRAINING_SEEDS:
            if right_id == "nominal_k8t2" and seed in (1, 10):
                left_values = arms["complete_supplemental"][seed]["adapted_returns"]
                left_source = "post_confirmation_supplemental_block_79300"
            else:
                left_values = arms["complete"][seed]["adapted_returns"]
                left_source = arm_sources["complete"][seed]
            right_values = arms[right_id][seed]["adapted_returns"]
            raw_deltas = left_values - right_values
            normalized_deltas = raw_deltas * factor
            bootstrap_seed = (
                BOOTSTRAP_SEED_ROOT + 1000 * seed + 10 * comparison_index
            )
            stats = _paired_episode_stats(normalized_deltas, bootstrap_seed=bootstrap_seed)
            per_seed_rows.append(
                {
                    "training_seed": seed,
                    "left_source_block": left_source,
                    "right_source_block": arm_sources[right_id][seed],
                    "raw_delta_mean_recomputed": float(raw_deltas.mean()),
                    **stats,
                }
            )
            normalized_delta_means.append(stats["mean"])
            raw_delta_means.append(float(raw_deltas.mean()))
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "left": "complete",
                "right": right_id,
                "per_training_seed": per_seed_rows,
                "cross_seed_normalized": _cross_seed_stats(normalized_delta_means),
                "cross_seed_raw": _cross_seed_stats(raw_delta_means),
                "paired_unit": "episode_case_within_one_fixed_checkpoint",
                "cross_seed_unit": "independent_base_policy_training_seed",
            }
        )

    calibration_path = results_root / (
        "channel_calibration/beta125_n512_seed27001_calibration.json"
    )
    calibration_sha = (
        sha256_file(calibration_path) if calibration_path.is_file() else None
    )
    if calibration_sha is not None and calibration_sha != EXPECTED_CALIBRATION_SHA256:
        raise AggregationError("live calibration file no longer matches its pinned SHA")

    total_base_wall = sum(
        value for value in base_wall_times.values() if value is not None
    )
    total_rollout_wall = sum(
        value
        for seed_map in rollout_wall_times.values()
        for value in seed_map.values()
        if value is not None
    )

    source = Path(__file__).resolve()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete",
        "results_root": str(results_root),
        "environment": ENVIRONMENT,
        "training_seeds": list(TRAINING_SEEDS),
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
            "source": NORMALIZATION_SOURCE,
            "scores_recomputed_from_episode_returns": True,
        },
        "evidence_labels": {
            "seeds_1_10_formal_arms": (
                "formal confirmation block 39300/49300/69300; not globally blind"
            ),
            "seeds_1_10_nominal_arm": (
                "post_confirmation_fresh_rollout_mechanism_audit block "
                "79300/89300/99300; designed after the formal result"
            ),
            "seeds_2_3_4": "scale-up expansion; controller hyperparameters frozen",
        },
        "blocking_note": (
            "The 250 rollout episodes are never pooled into one interval; "
            "cross-seed statistics use one mean per independent base-policy "
            "training seed (n=5). The nominal-Q comparison for seeds 1/10 uses "
            "the supplemental block, so complete-vs-nominal and "
            "complete-vs-identity/OPEX/inverse deltas for those seeds come from "
            "different 50-case blocks, each internally paired."
        ),
        "controller_contracts": {
            control_id: dict(contract) for control_id, contract in CHANNEL_CONTRACTS.items()
        },
        "per_seed_controller_means": per_seed_controller_table,
        "paired_comparisons": comparisons,
        "per_decision_cost": {
            control_id: dict(cost) for control_id, cost in PER_DECISION_COST.items()
        },
        "cost_scope_note": (
            "Equal compute means equal per-decision base-actor rows, Q1 rows, "
            "and backward calls. It is not an exact FLOP or wall-time match: "
            "inverse anchoring adds 48 deterministic scalar bisection "
            "iterations per decision, and total rows vary with episode length."
        ),
        "wall_times": {
            "base_training_seconds_per_seed": {
                str(seed): base_wall_times.get(seed) for seed in TRAINING_SEEDS
            },
            "rollout_seconds_per_seed": {
                str(seed): rollout_wall_times.get(seed, {}) for seed in TRAINING_SEEDS
            },
            "calibration_seconds": None,
            "calibration_time_note": (
                "calibration wall time is not recorded in the frozen calibration "
                "JSON; the 512-pair calibration is a one-time scalar MLE cost"
            ),
            "total_base_training_seconds": total_base_wall,
            "total_recorded_rollout_seconds": total_rollout_wall,
            "total_gpu_occupied_approximation_seconds": (
                (total_base_wall or 0.0) + total_rollout_wall
            ),
            "gpu_time_scope": (
                "sum of recorded stage wall times; an approximation of "
                "GPU-occupied time, not a FLOP count"
            ),
        },
        "input_files": input_files,
        "calibration_sha256": calibration_sha,
        "aggregator": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
    }
    write_json_new(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/hubl_research_20260914/results"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = aggregate(args.results_root, args.output)
    except AggregationError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
