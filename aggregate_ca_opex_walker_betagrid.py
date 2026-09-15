"""Fail-closed aggregation for the Walker2d beta-severity grid.

Consumes one directory per predeclared (beta, base-checkpoint) unit produced by
run_ca_opex_walker_betagrid.sh.  Every beta keeps its own 512-pair calibration
and its own 50-case rollout block; no K/T/eta/delta is ever reselected.  All
scores and paired statistics are recomputed from per-episode returns; the
output is a create-only JSON artifact and every beta, including non-improving
ones, is retained.
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

SCHEMA = "ca-opex-walker-betagrid-aggregate-v1"
ENVIRONMENT = "Walker2d-v4"
EPISODES = 50
REFERENCE_MIN = 1.629008
REFERENCE_MAX = 4592.3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED_ROOT = 4_026_091_500

# Frozen unit table: beta -> pairs seed, bootstrap seed, env/noise/grad starts.
BETA_UNITS = {
    0.5: (27_011, 28_011, 51_400, 61_400, 71_400),
    0.9: (27_012, 28_012, 51_500, 61_500, 71_500),
    1.1: (27_013, 28_013, 51_600, 61_600, 71_600),
    1.4: (27_014, 28_014, 51_700, 61_700, 71_700),
}
CHECKPOINTS = {
    1: {
        "path": "inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt",
        "sha256": "22b47965695bfa312533741abaa1e17e3cbbc091f3c5e6ba7fc8953fbc8ca6ae",
    },
    2: {
        "path": "scaleup/walker_seed2/base/latest.pt",
        "sha256": "d94b05044c9071c2a987aa28bd2dabbf04a91a7164d4215f851f37aaff06a75e",
    },
    10: {
        "path": "inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt",
        "sha256": "eba1af040aa73823ceba28048e10d3af2eef54c59f09c1e0194f27aa0ed4944d",
    },
}

CHANNEL_CONTRACTS = {
    "complete": {
        "method_id": "channel_aware_opex_inverse_anchor",
        "baseline_transform": "inverse",
        "K": 8,
        "T": 2,
        "step_size": 0.1,
        "delta_max": 0.25,
        "calibrated": True,
    },
    "nominal_k8t2": {
        "method_id": "channel_aware_opex_identity_anchor",
        "baseline_transform": "identity",
        "K": 8,
        "T": 2,
        "step_size": 0.1,
        "delta_max": 2.0,
        "calibrated": False,
    },
}
COMMAND_ARMS = {
    "inverse_only": "uniform_mean_inverse",
    "identity_command": "identity",
}
COMPARISONS = (
    ("complete_minus_nominal_k8t2", "nominal_k8t2"),
    ("complete_minus_inverse_only", "inverse_only"),
    ("complete_minus_identity_command", "identity_command"),
)
CONTROL_FILES = {
    "complete": "controls/complete.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
}


class AggregationError(RuntimeError):
    """Raised when an input violates the beta-grid contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AggregationError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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
    if not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-9):
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


def _seed_block(start: int) -> list[int]:
    return list(range(start, start + EPISODES))


def _require_block(values: object, start: int, label: str) -> list[int]:
    expected = _seed_block(start)
    if not isinstance(values, list) or [int(v) for v in values] != expected:
        raise AggregationError(f"{label} does not match block starting at {start}")
    return expected


def _load_channel_unit(
    path: Path,
    *,
    control_id: str,
    beta: float,
    checkpoint_sha: str,
    env_start: int,
    noise_start: int,
    grad_start: int,
    calibration_path: Path,
) -> Dict[str, Any]:
    label = f"{path.parent.parent.name}.{control_id}"
    if not path.is_file():
        raise AggregationError(f"{label} record is missing: {path}")
    raw = read_json(path)
    if raw.get("raw_schema") != "channel_opex_v1" or raw.get("status") != "complete":
        raise AggregationError(f"{label} is not a complete channel_opex_v1 record")
    if raw.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} environment mismatch")
    contract = CHANNEL_CONTRACTS[control_id]
    if raw.get("method_id") != contract["method_id"]:
        raise AggregationError(f"{label} method contract mismatch")

    base = _mapping(raw.get("base_checkpoint"), f"{label}.base_checkpoint")
    if _digest(base.get("sha256"), f"{label}.base_checkpoint.sha256") != checkpoint_sha:
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

    calibration = _mapping(raw.get("calibration"), f"{label}.calibration")
    if contract["calibrated"]:
        calibration_sha = _digest(
            calibration.get("calibration_sha256"), f"{label}.calibration.sha256"
        )
        if calibration_sha != sha256_file(calibration_path):
            raise AggregationError(f"{label} calibration SHA mismatch")
        calibration_payload = read_json(calibration_path)
        beta_hat = _finite(
            _mapping(calibration_payload.get("estimate"), "calibration.estimate").get(
                "beta_mle"
            ),
            "calibration.estimate.beta_mle",
        )
        _same_float(controller.get("model_beta"), beta_hat, f"{label}.model_beta")
    else:
        if calibration.get("calibration_sha256") is not None:
            raise AggregationError(f"{label} nominal controller unexpectedly calibrated")
        _same_float(controller.get("model_beta"), 0.0, f"{label}.model_beta")
        beta_hat = None

    channel = _mapping(raw.get("channel"), f"{label}.channel")
    _same_float(
        channel.get("environment_rollout_beta"), beta, f"{label}.rollout_beta"
    )

    arms = _mapping(raw.get("arms"), f"{label}.arms")
    adapted = _mapping(arms.get("adapted"), f"{label}.arms.adapted")
    env_seeds = _require_block(adapted.get("environment_seeds"), env_start, f"{label}.env_seeds")
    noise_block = _require_block(
        adapted.get("action_noise_seeds"), noise_start, f"{label}.noise_seeds"
    )
    grad_block = _require_block(
        adapted.get("gradient_noise_seeds"), grad_start, f"{label}.gradient_seeds"
    )
    returns = _returns(adapted.get("returns"), f"{label}.returns")
    lengths = adapted.get("lengths")
    if not isinstance(lengths, list) or len(lengths) != EPISODES:
        raise AggregationError(f"{label}.lengths must contain {EPISODES} entries")
    lengths = [_integer(value, f"{label}.lengths") for value in lengths]
    del env_seeds, noise_block, grad_block
    return {
        "adapted_returns": returns,
        "lengths": lengths,
        "beta_hat": beta_hat,
        "calibration_sha256": (
            sha256_file(calibration_path) if contract["calibrated"] else None
        ),
    }


def _load_command_unit(
    path: Path,
    *,
    arm_id: str,
    beta: float,
    checkpoint_sha: str,
    env_start: int,
    noise_start: int,
) -> Dict[str, Any]:
    label = f"{path.parent.parent.name}.{arm_id}"
    if not path.is_file():
        raise AggregationError(f"{label} record is missing: {path}")
    raw = read_json(path)
    if raw.get("status") != "complete" or raw.get("environment") != ENVIRONMENT:
        raise AggregationError(f"{label} is not a complete {ENVIRONMENT} record")
    if _digest(raw.get("checkpoint_sha256"), f"{label}.checkpoint_sha256") != checkpoint_sha:
        raise AggregationError(f"{label} checkpoint SHA mismatch")
    if _integer(raw.get("checkpoint_step"), f"{label}.checkpoint_step") != 25_000:
        raise AggregationError(f"{label} checkpoint step mismatch")
    _same_float(raw.get("command_scale"), 1.0, f"{label}.command_scale")
    transform = COMMAND_ARMS[arm_id]
    if raw.get("command_transform") != transform:
        raise AggregationError(f"{label} command transform mismatch")
    _same_float(raw.get("command_transform_beta"), beta, f"{label}.transform_beta")
    arm = _mapping(raw.get("persistent_action_noise"), f"{label}.persistent_action_noise")
    _same_float(arm.get("action_noise_beta"), beta, f"{label}.action_noise_beta")
    _require_block(arm.get("environment_seeds"), env_start, f"{label}.env_seeds")
    _require_block(arm.get("action_noise_seeds"), noise_start, f"{label}.noise_seeds")
    returns = _returns(arm.get("returns"), f"{label}.returns")
    return {"adapted_returns": returns, "beta_hat": None, "calibration_sha256": None}


def _paired_stats(values: np.ndarray, *, bootstrap_seed: int) -> Dict[str, Any]:
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
        "bootstrap95_low": float(bootstrap_low),
        "bootstrap95_high": float(bootstrap_high),
        "positive_episode_count": int((values > 0.0).sum()),
    }


def _cross_checkpoint(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1)) if n > 1 else 0.0
    standard_error = sample_std / math.sqrt(n) if n > 1 else 0.0
    critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else 0.0
    return {
        "n_checkpoints": n,
        "mean": mean,
        "sample_std_ddof1": sample_std,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "positive_checkpoint_count": int((array > 0.0).sum()),
    }


def _require_new_output(path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise AggregationError(f"refusing to overwrite output: {path}")


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


def _format_beta(beta: float) -> str:
    return f"{beta:g}"


def _require_expected_unit_directories(results_root: Path) -> None:
    expected = {
        f"beta{_format_beta(beta)}_seed{seed}"
        for beta in BETA_UNITS
        for seed in CHECKPOINTS
    }
    actual = {
        path.name
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith("beta")
    }
    unexpected = sorted(actual - expected)
    if unexpected:
        raise AggregationError(
            "unexpected beta-grid unit directories (possible duplicate seed): "
            + ", ".join(unexpected)
        )


def aggregate(results_root: Path, output: Path) -> Dict[str, Any]:
    results_root = results_root.resolve()
    output = output.resolve()
    _require_new_output(output)
    if not results_root.is_dir():
        raise AggregationError(f"results root is missing: {results_root}")
    _require_expected_unit_directories(results_root)
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)

    units: Dict[float, Dict[int, Dict[str, Any]]] = {}
    input_files: list[Dict[str, Any]] = []
    for beta, (pairs_seed, _bootstrap_seed, env_start, noise_start, grad_start) in (
        BETA_UNITS.items()
    ):
        calibration_path = (
            results_root / f"calibration/beta{_format_beta(beta)}_n512_seed{pairs_seed}"
            "_calibration.json"
        )
        if not calibration_path.is_file():
            raise AggregationError(f"beta {beta} calibration file is missing")
        input_files.append(
            {
                "role": f"calibration:beta{_format_beta(beta)}",
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
            }
        )
        units[beta] = {}
        for seed, checkpoint in CHECKPOINTS.items():
            unit_dir = results_root / f"beta{_format_beta(beta)}_seed{seed}"
            parsed: Dict[str, Any] = {}
            for control_id, relative in CONTROL_FILES.items():
                parsed[control_id] = _load_channel_unit(
                    unit_dir / relative,
                    control_id=control_id,
                    beta=beta,
                    checkpoint_sha=checkpoint["sha256"],
                    env_start=env_start,
                    noise_start=noise_start,
                    grad_start=grad_start,
                    calibration_path=calibration_path,
                )
            for arm_id in COMMAND_ARMS:
                parsed[arm_id] = _load_command_unit(
                    unit_dir / "controls" / f"{arm_id}.json",
                    arm_id=arm_id,
                    beta=beta,
                    checkpoint_sha=checkpoint["sha256"],
                    env_start=env_start,
                    noise_start=noise_start,
                )
            units[beta][seed] = parsed
            input_files.append(
                {
                    "role": f"unit:beta{_format_beta(beta)}:seed{seed}",
                    "path": str(unit_dir),
                    "sha256": None,
                }
            )

    per_beta: Dict[str, Any] = {}
    for beta, seed_map in units.items():
        beta_key = _format_beta(beta)
        calibration_shas = {
            parsed["complete"]["calibration_sha256"] for parsed in seed_map.values()
        }
        if len(calibration_shas) != 1 or None in calibration_shas:
            raise AggregationError(f"beta {beta}: units do not share one calibration")
        betas_hat = {
            parsed["complete"]["beta_hat"] for parsed in seed_map.values()
        }
        if len(betas_hat) != 1 or None in betas_hat:
            raise AggregationError(f"beta {beta}: units disagree on beta_hat")
        seed_rows = {}
        comparison_vectors: Dict[str, list[float]] = {
            comparison_id: [] for comparison_id, _ in COMPARISONS
        }
        arm_means: Dict[str, list[float]] = {
            control_id: [] for control_id in ("complete", "nominal_k8t2",
                                              "inverse_only", "identity_command")
        }
        for seed in sorted(seed_map):
            parsed = seed_map[seed]
            complete = parsed["complete"]["adapted_returns"]
            row: Dict[str, Any] = {
                "complete_normalized_mean": float(
                    ((complete - REFERENCE_MIN) * factor).mean()
                ),
                "complete_raw_mean": float(complete.mean()),
            }
            for control_id in ("nominal_k8t2", "inverse_only", "identity_command"):
                arm_means[control_id].append(
                    float(
                        ((parsed[control_id]["adapted_returns"] - REFERENCE_MIN) * factor).mean()
                    )
                )
            arm_means["complete"].append(row["complete_normalized_mean"])
            for comparison_index, (comparison_id, right_id) in enumerate(COMPARISONS):
                raw_deltas = complete - parsed[right_id]["adapted_returns"]
                normalized = raw_deltas * factor
                bootstrap_seed = (
                    BOOTSTRAP_SEED_ROOT + int(beta * 100) * 7 + 100 * seed + comparison_index
                )
                stats = _paired_stats(normalized, bootstrap_seed=bootstrap_seed)
                row[comparison_id] = {
                    "raw_delta_mean": float(raw_deltas.mean()),
                    **stats,
                }
                comparison_vectors[comparison_id].append(stats["mean"])
            seed_rows[str(seed)] = row
        per_beta[beta_key] = {
            "beta": beta,
            "beta_hat": next(iter(betas_hat)),
            "calibration_sha256": next(iter(calibration_shas)),
            "rollout_blocks": {
                "environment_start": BETA_UNITS[beta][2],
                "action_noise_start": BETA_UNITS[beta][3],
                "gradient_noise_start": BETA_UNITS[beta][4],
            },
            "per_checkpoint": seed_rows,
            "arm_cross_checkpoint": {
                control_id: _cross_checkpoint(means)
                for control_id, means in arm_means.items()
            },
            "comparison_cross_checkpoint": {
                comparison_id: _cross_checkpoint(values)
                for comparison_id, values in comparison_vectors.items()
            },
        }

    source = Path(__file__).resolve()
    report: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete",
        "results_root": str(results_root),
        "environment": ENVIRONMENT,
        "betas": [entry["beta"] for entry in per_beta.values()],
        "checkpoints": [
            {"training_seed": seed, **checkpoint} for seed, checkpoint in CHECKPOINTS.items()
        ],
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
            "scores_recomputed_from_episode_returns": True,
        },
        "protocol_note": (
            "K/T/eta/delta are frozen across all betas; each beta has its own "
            "512-pair calibration and its own 50-case rollout block. All betas, "
            "including non-improving ones, are retained. Per-beta cross-checkpoint "
            "intervals are descriptive over three checkpoints, not training-seed "
            "inference."
        ),
        "per_beta": per_beta,
        "input_files": input_files,
        "aggregator": {"path": str(source), "sha256": sha256_file(source)},
    }
    write_json_new(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/hubl_research_20260914/results/betagrid"),
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
