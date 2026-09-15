"""Independent verifier for the Walker2d beta-severity aggregate.

This verifier deliberately does not import the beta-grid aggregator.  It
re-reads every calibration and controller record, checks the frozen beta,
checkpoint, arm, and seed contracts, re-hashes the inputs, and recomputes the
per-checkpoint and cross-checkpoint statistics from raw episode returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy.stats import t as student_t

EPISODES = 50
REFERENCE_MIN = 1.629008
REFERENCE_MAX = 4592.3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED_ROOT = 4_026_091_500
TOLERANCE = 1e-9

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
CONTROL_FILES = {
    "complete": "controls/complete.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
    "inverse_only": "controls/inverse_only.json",
    "identity_command": "controls/identity_command.json",
}
COMPARISONS = (
    ("complete_minus_nominal_k8t2", "nominal_k8t2"),
    ("complete_minus_inverse_only", "inverse_only"),
    ("complete_minus_identity_command", "identity_command"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(observed: float, expected: float, label: str, failures: List[str]) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        failures.append(f"{label}: {observed!r} != {expected!r}")


def _finite(value: object, label: str, failures: List[str]) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        failures.append(f"{label}: not numeric")
        return None
    if not math.isfinite(result):
        failures.append(f"{label}: not finite")
        return None
    return result


def _returns(path: Path, control_id: str, failures: List[str]) -> np.ndarray | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if control_id in ("complete", "nominal_k8t2"):
            values = record["arms"]["adapted"]["returns"]
        else:
            values = record["persistent_action_noise"]["returns"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        failures.append(f"{path}: cannot read returns: {exc}")
        return None
    if not isinstance(values, list) or len(values) != EPISODES:
        failures.append(f"{path}: expected {EPISODES} returns")
        return None
    converted = [_finite(value, f"{path}[{i}]", failures) for i, value in enumerate(values)]
    if any(value is None for value in converted):
        return None
    return np.asarray(converted, dtype=np.float64)


def _paired_stats(values: np.ndarray, bootstrap_seed: int) -> Dict[str, float | int]:
    mean = float(values.mean())
    sample_std = float(values.std(ddof=1))
    critical = float(student_t.ppf(0.975, EPISODES - 1))
    generator = np.random.Generator(np.random.PCG64(bootstrap_seed))
    indices = generator.integers(0, EPISODES, size=(BOOTSTRAP_REPLICATES, EPISODES))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear").tolist()
    return {
        "mean": mean,
        "t95_low": mean - critical * sample_std / math.sqrt(EPISODES),
        "t95_high": mean + critical * sample_std / math.sqrt(EPISODES),
        "bootstrap95_low": float(low),
        "bootstrap95_high": float(high),
        "positive_episode_count": int((values > 0.0).sum()),
    }


def _cross_checkpoint(values: List[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1))
    critical = float(student_t.ppf(0.975, n - 1))
    return {
        "mean": mean,
        "sample_std_ddof1": sample_std,
        "t95_low": mean - critical * sample_std / math.sqrt(n),
        "t95_high": mean + critical * sample_std / math.sqrt(n),
        "positive_checkpoint_count": int((array > 0.0).sum()),
    }


def _check_record_hashes(
    payload: Mapping[str, Any], failures: List[str], checks: List[int]
) -> None:
    for entry in payload.get("input_files", []):
        if not isinstance(entry, Mapping):
            failures.append("input_files contains a non-object")
            continue
        role = str(entry.get("role"))
        path = Path(str(entry.get("path", "")))
        if entry.get("sha256") is None:
            if not path.is_dir():
                failures.append(f"{role}: missing unit directory {path}")
            continue
        if not path.is_file():
            failures.append(f"{role}: missing {path}")
            continue
        checks[0] += 1
        try:
            observed = sha256_file(path)
        except OSError as exc:
            failures.append(f"{role}: cannot hash {path}: {exc}")
            continue
        if observed != str(entry["sha256"]).lower():
            failures.append(f"{role}: hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate_path = args.aggregate.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite output: {output_path}")

    failures: List[str] = []
    checks = [0]
    try:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read aggregate: {exc}")
    if not isinstance(payload, Mapping):
        parser.error("aggregate root must be an object")
    if payload.get("schema_version") != "ca-opex-walker-betagrid-aggregate-v1":
        failures.append("aggregate schema mismatch")
    if payload.get("status") != "complete":
        failures.append("aggregate is not complete")
    _check_record_hashes(payload, failures, checks)

    results_root = Path(str(payload.get("results_root", ""))).resolve()
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)
    raw: Dict[float, Dict[int, Dict[str, np.ndarray]]] = {}
    for beta, (_pairs, _bootstrap, env_start, noise_start, grad_start) in BETA_UNITS.items():
        beta_key = f"{beta:g}"
        raw[beta] = {}
        calibration_path = results_root / f"calibration/beta{beta_key}_n512_seed{_pairs}_calibration.json"
        if not calibration_path.is_file():
            failures.append(f"beta {beta}: missing calibration")
        for seed, checkpoint in CHECKPOINTS.items():
            unit_dir = results_root / f"beta{beta_key}_seed{seed}"
            raw[beta][seed] = {}
            for control_id, relative in CONTROL_FILES.items():
                path = unit_dir / relative
                if not path.is_file():
                    failures.append(f"beta{beta_key}/seed{seed}/{control_id}: missing raw record")
                    continue
                checks[0] += 1
                values = _returns(path, control_id, failures)
                if values is not None:
                    raw[beta][seed][control_id] = values

    per_beta = payload.get("per_beta", {})
    if not isinstance(per_beta, Mapping):
        failures.append("per_beta is not an object")
        per_beta = {}
    for beta, seed_map in raw.items():
        beta_key = f"{beta:g}"
        report = per_beta.get(beta_key)
        if not isinstance(report, Mapping):
            failures.append(f"beta {beta}: aggregate row is missing")
            continue
        report_rows = report.get("per_checkpoint", {})
        if not isinstance(report_rows, Mapping):
            failures.append(f"beta {beta}: per_checkpoint is not an object")
            continue
        comparison_vectors: Dict[str, List[float]] = {name: [] for name, _ in COMPARISONS}
        arm_vectors: Dict[str, List[float]] = {
            control_id: [] for control_id in CONTROL_FILES
        }
        for seed, controls in seed_map.items():
            row = report_rows.get(str(seed))
            if not isinstance(row, Mapping):
                failures.append(f"beta {beta}/seed{seed}: per-checkpoint row is missing")
                continue
            complete = controls.get("complete")
            if complete is None:
                continue
            complete_norm = float(((complete - REFERENCE_MIN) * factor).mean())
            checks[0] += 1
            close(complete_norm, float(row.get("complete_normalized_mean")), f"beta{beta_key}/seed{seed} complete normalized", failures)
            close(float(complete.mean()), float(row.get("complete_raw_mean")), f"beta{beta_key}/seed{seed} complete raw", failures)
            arm_vectors["complete"].append(complete_norm)
            for control_id in ("nominal_k8t2", "inverse_only", "identity_command"):
                values = controls.get(control_id)
                if values is None:
                    continue
                arm_vectors[control_id].append(float(((values - REFERENCE_MIN) * factor).mean()))
            for comparison_index, (comparison_id, right_id) in enumerate(COMPARISONS):
                values = controls.get(right_id)
                if values is None:
                    continue
                raw_delta = complete - values
                normalized = raw_delta * factor
                stats = _paired_stats(
                    normalized,
                    BOOTSTRAP_SEED_ROOT + int(beta * 100) * 7 + 100 * seed + comparison_index,
                )
                target = row.get(comparison_id)
                if not isinstance(target, Mapping):
                    failures.append(f"beta{beta_key}/seed{seed}/{comparison_id}: row missing")
                    continue
                checks[0] += 6
                for key in ("mean", "t95_low", "t95_high", "bootstrap95_low", "bootstrap95_high"):
                    close(float(stats[key]), float(target.get(key)), f"beta{beta_key}/seed{seed}/{comparison_id}/{key}", failures)
                if int(stats["positive_episode_count"]) != int(target.get("positive_episode_count")):
                    failures.append(f"beta{beta_key}/seed{seed}/{comparison_id}/positive count mismatch")
                comparison_vectors[comparison_id].append(float(stats["mean"]))

        arm_report = report.get("arm_cross_checkpoint", {})
        for control_id, values in arm_vectors.items():
            target = arm_report.get(control_id) if isinstance(arm_report, Mapping) else None
            if not isinstance(target, Mapping) or len(values) != len(CHECKPOINTS):
                failures.append(f"beta{beta_key}/{control_id}: cross-checkpoint row missing")
                continue
            expected = _cross_checkpoint(values)
            checks[0] += 5
            for key in ("mean", "sample_std_ddof1", "t95_low", "t95_high"):
                close(float(expected[key]), float(target.get(key)), f"beta{beta_key}/{control_id}/{key}", failures)
            if int(expected["positive_checkpoint_count"]) != int(target.get("positive_checkpoint_count")):
                failures.append(f"beta{beta_key}/{control_id}/positive count mismatch")
        comparison_report = report.get("comparison_cross_checkpoint", {})
        for comparison_id, values in comparison_vectors.items():
            target = comparison_report.get(comparison_id) if isinstance(comparison_report, Mapping) else None
            if not isinstance(target, Mapping) or len(values) != len(CHECKPOINTS):
                failures.append(f"beta{beta_key}/{comparison_id}: cross-checkpoint row missing")
                continue
            expected = _cross_checkpoint(values)
            checks[0] += 5
            for key in ("mean", "sample_std_ddof1", "t95_low", "t95_high"):
                close(float(expected[key]), float(target.get(key)), f"beta{beta_key}/{comparison_id}/{key}", failures)
            if int(expected["positive_checkpoint_count"]) != int(target.get("positive_checkpoint_count")):
                failures.append(f"beta{beta_key}/{comparison_id}/positive count mismatch")

    result = {
        "schema_version": "ca-opex-walker-betagrid-verify-v1",
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "status": "verified_complete" if not failures else "failed",
        "checks_performed": checks[0],
        "failure_count": len(failures),
        "failures": failures,
        "betas": list(BETA_UNITS),
        "training_seeds": list(CHECKPOINTS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
