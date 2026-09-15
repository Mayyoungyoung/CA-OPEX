"""Independent verifier for the five-training-seed Walker2d CA-OPEX aggregate.

This tool deliberately does not import the aggregator.  It reads the raw
records referenced by the aggregate's input inventory, recomputes per-seed
controller means, per-seed paired differences, and cross-seed statistics from
per-episode returns, re-hashes every input file, and reports any deviation.
Its own output is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

EPISODES = 50
TRAINING_SEEDS = (1, 2, 3, 4, 10)
SCALEUP_SEEDS = (2, 3, 4)
FORMAL_BLOCK_ENV = (39_300, 39_349)
SUPPLEMENTAL_BLOCK_ENV = (79_300, 79_349)
REFERENCE_MIN = 1.629008
REFERENCE_MAX = 4592.3
TOLERANCE = 1e-9

FORMAL_ROLES = {
    "complete": "formal:complete:seed",
    "calibrated_identity": "formal:calibrated_identity:seed",
    "original_opex": "formal:original_opex:seed",
}
SUPPLEMENTAL_ROLES = {
    "complete_supplemental": "supplemental:complete:seed",
    "nominal_k8t2": "supplemental:nominal_k8t2:seed",
}
SCALEUP_FILES = {
    "complete": "controls/complete.json",
    "calibrated_identity": "controls/calibrated_identity.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
    "original_opex": "controls/original_opex.json",
    "inverse_only": "controls/inverse_only.json",
}
COMPARISON_RIGHTS = {
    "complete_minus_calibrated_identity": "calibrated_identity",
    "complete_minus_original_opex": "original_opex",
    "complete_minus_inverse_only": "inverse_only",
    "complete_minus_nominal_k8t2": "nominal_k8t2",
    "complete_minus_identity_command": "identity_command",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adapted_returns(record: Dict[str, Any], *, inverse_only: bool = False) -> np.ndarray:
    if inverse_only:
        arm = record["persistent_action_noise"]
    else:
        arm = record["arms"]["adapted"]
    values = np.asarray(arm["returns"], dtype=np.float64)
    if values.shape != (EPISODES,):
        raise ValueError(f"expected {EPISODES} returns, got {values.shape}")
    return values


def baseline_returns(record: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(record["arms"]["baseline_only"]["returns"], dtype=np.float64)
    if values.shape != (EPISODES,):
        raise ValueError("baseline arm does not carry 50 returns")
    return values


def env_seed_start(record: Dict[str, Any], *, inverse_only: bool = False) -> int:
    arm = record["persistent_action_noise"] if inverse_only else record["arms"]["adapted"]
    seeds = arm["environment_seeds"]
    if len(seeds) != EPISODES:
        raise ValueError("environment seed vector length mismatch")
    return int(seeds[0])


def cross_seed(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    standard_error = std / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    return {
        "mean": mean,
        "sample_std_ddof1": std,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "positive_training_seed_count": int((array > 0.0).sum()),
    }


def close(observed: float, expected: float, label: str, failures: List[str]) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        failures.append(f"{label}: recomputed {observed!r} != reported {expected!r}")


def dig(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate_path = args.aggregate.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite output: {output_path}")
    report = load_json(aggregate_path)
    failures: List[str] = []
    checks = 0

    inventory = {entry["role"]: entry for entry in report.get("input_files", [])}

    def path_of(role: str) -> Path:
        entry = inventory.get(role)
        if entry is None:
            raise KeyError(f"missing inventory role {role}")
        return Path(entry["path"])

    for role, entry in inventory.items():
        path = Path(entry["path"])
        if not path.is_file():
            failures.append(f"{role}: file missing {path}")
            continue
        observed = sha256_file(path)
        checks += 1
        if observed != entry["sha256"]:
            failures.append(f"{role}: hash mismatch {observed} != {entry['sha256']}")

    arm_returns: Dict[str, Dict[int, np.ndarray]] = {}
    arm_sources: Dict[str, Dict[int, str]] = {}

    for seed in (1, 10):
        for control_id, role_prefix in FORMAL_ROLES.items():
            record = load_json(path_of(f"{role_prefix}{seed}"))
            arm_returns.setdefault(control_id, {})[seed] = adapted_returns(record)
            arm_sources.setdefault(control_id, {})[seed] = "formal"
        arm_returns.setdefault("inverse_only", {})[seed] = baseline_returns(
            load_json(path_of(f"formal:complete:seed{seed}"))
        )
        arm_sources.setdefault("inverse_only", {})[seed] = "formal_baseline"
        arm_returns.setdefault("identity_command", {})[seed] = baseline_returns(
            load_json(path_of(f"formal:calibrated_identity:seed{seed}"))
        )
        arm_sources.setdefault("identity_command", {})[seed] = "formal_baseline"
        for control_id, role_prefix in SUPPLEMENTAL_ROLES.items():
            record = load_json(path_of(f"{role_prefix}{seed}"))
            arm_returns.setdefault(control_id, {})[seed] = adapted_returns(record)
            arm_sources.setdefault(control_id, {})[seed] = "supplemental"

    scaleup_blocks: Dict[int, int] = {}
    for seed in SCALEUP_SEEDS:
        per_seed_aggregate = load_json(path_of(f"scaleup:aggregate:seed{seed}"))
        run_root = Path(inventory[f"scaleup:aggregate:seed{seed}"]["path"]).parent
        for control_id, relative in SCALEUP_FILES.items():
            record = load_json(run_root / relative)
            inverse_only = control_id == "inverse_only"
            arm_returns.setdefault(control_id, {})[seed] = adapted_returns(
                record, inverse_only=inverse_only
            )
            arm_sources.setdefault(control_id, {})[seed] = "scaleup"
            if control_id == "complete":
                scaleup_blocks[seed] = env_seed_start(record, inverse_only=False)
        identity_record = load_json(run_root / "controls/calibrated_identity.json")
        arm_returns.setdefault("identity_command", {})[seed] = baseline_returns(
            identity_record
        )
        arm_sources.setdefault("identity_command", {})[seed] = "scaleup_baseline"
        reported_seed = per_seed_aggregate.get("training_seed")
        if reported_seed != seed:
            failures.append(f"scaleup seed {seed}: aggregate seed mismatch {reported_seed}")

    for seed, start in scaleup_blocks.items():
        for reserved in (FORMAL_BLOCK_ENV, SUPPLEMENTAL_BLOCK_ENV):
            if reserved[0] <= start <= reserved[1]:
                failures.append(f"scaleup seed {seed} overlaps reserved env block {reserved}")
        for other, other_start in scaleup_blocks.items():
            if other != seed and start == other_start:
                failures.append(
                    f"scaleup seeds {seed} and {other} share environment block {start}"
                )
    checks += 1

    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)

    for entry in report.get("per_seed_controller_means", []):
        control_id = entry["control_id"]
        for seed_text, row in entry.get("per_seed", {}).items():
            seed = int(seed_text)
            values = arm_returns.get(control_id, {}).get(seed)
            if values is None:
                failures.append(f"{control_id} seed {seed}: verifier has no raw returns")
                continue
            checks += 1
            close(
                float(values.mean()),
                row["raw_score_mean_recomputed"],
                f"{control_id} seed {seed} raw mean",
                failures,
            )
            close(
                float(((values - REFERENCE_MIN) * factor).mean()),
                row["normalized_score_mean_recomputed"],
                f"{control_id} seed {seed} normalized mean",
                failures,
            )
        descriptive = entry.get("cross_seed_descriptive")
        if descriptive is not None and entry.get("per_seed"):
            normalized_means = [
                ((arm_returns[control_id][int(seed_text)] - REFERENCE_MIN) * factor).mean()
                for seed_text in entry["per_seed"]
            ]
            recomputed = cross_seed(normalized_means)
            checks += 1
            close(
                recomputed["mean"],
                descriptive["mean"],
                f"{control_id} cross-seed mean",
                failures,
            )
            close(
                recomputed["t95_low"],
                descriptive["t95_low"],
                f"{control_id} cross-seed t95 low",
                failures,
            )
            close(
                recomputed["t95_high"],
                descriptive["t95_high"],
                f"{control_id} cross-seed t95 high",
                failures,
            )

    for comparison in report.get("paired_comparisons", []):
        comparison_id = comparison["comparison_id"]
        right_id = COMPARISON_RIGHTS.get(comparison_id)
        if right_id is None:
            failures.append(f"unknown comparison {comparison_id}")
            continue
        normalized_delta_means: List[float] = []
        raw_delta_means: List[float] = []
        for row in comparison.get("per_training_seed", []):
            seed = int(row["training_seed"])
            if right_id == "nominal_k8t2" and seed in (1, 10):
                left = arm_returns["complete_supplemental"][seed]
            else:
                left = arm_returns["complete"][seed]
            right = arm_returns[right_id][seed]
            raw_delta = float((left - right).mean())
            normalized_delta = float(((left - right) * factor).mean())
            checks += 1
            close(
                raw_delta,
                row["raw_delta_mean_recomputed"],
                f"{comparison_id} seed {seed} raw delta",
                failures,
            )
            close(
                normalized_delta,
                row["mean"],
                f"{comparison_id} seed {seed} normalized delta",
                failures,
            )
            raw_delta_means.append(raw_delta)
            normalized_delta_means.append(normalized_delta)
        cross_normalized = cross_seed(normalized_delta_means)
        cross_raw = cross_seed(raw_delta_means)
        checks += 2
        close(
            cross_normalized["mean"],
            dig(comparison, "cross_seed_normalized", "mean"),
            f"{comparison_id} cross-seed normalized mean",
            failures,
        )
        close(
            cross_normalized["t95_low"],
            dig(comparison, "cross_seed_normalized", "t95_low"),
            f"{comparison_id} cross-seed normalized t95 low",
            failures,
        )
        close(
            cross_normalized["t95_high"],
            dig(comparison, "cross_seed_normalized", "t95_high"),
            f"{comparison_id} cross-seed normalized t95 high",
            failures,
        )
        close(
            cross_raw["mean"],
            dig(comparison, "cross_seed_raw", "mean"),
            f"{comparison_id} cross-seed raw mean",
            failures,
        )
        expected_positive = int((np.asarray(normalized_delta_means) > 0).sum())
        if (
            dig(comparison, "cross_seed_normalized", "positive_training_seed_count")
            != expected_positive
        ):
            failures.append(
                f"{comparison_id}: positive seed count mismatch "
                f"{expected_positive} != "
                f"{dig(comparison, 'cross_seed_normalized', 'positive_training_seed_count')}"
            )
        checks += 1

    verifier_result = {
        "schema_version": "ca-opex-walker-multiseed-verify-v1",
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "status": "verified_complete" if not failures else "failed",
        "checks_performed": checks,
        "failure_count": len(failures),
        "failures": failures,
        "training_seeds": list(TRAINING_SEEDS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(verifier_result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verifier_result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
