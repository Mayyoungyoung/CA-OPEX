"""Independent verifier for the Hopper-v4 three-training-seed aggregate.

Does not import the aggregator.  Re-reads every raw record named by the
aggregate inventory, recomputes per-seed means and cross-seed statistics, and
re-verifies all input hashes.  Output is create-only.
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
REFERENCE_MIN = -20.272305
REFERENCE_MAX = 3234.3
TOLERANCE = 1e-9
TRAINING_SEEDS = (2, 3, 4)
RIGHT_ARMS = {
    "complete_minus_calibrated_identity": "calibrated_identity",
    "complete_minus_nominal_k8t2": "nominal_k8t2",
    "complete_minus_original_opex": "original_opex",
    "complete_minus_inverse_only": "inverse_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(observed: float, expected: float, label: str, failures: List[str]) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        failures.append(f"{label}: {observed!r} != {expected!r}")


def cross_seed(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    se = std / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    return {"mean": mean, "t95_low": mean - critical * se, "t95_high": mean + critical * se}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate_path = args.aggregate.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite output: {output_path}")
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    failures: List[str] = []
    checks = 0
    inventory = {entry["role"]: entry for entry in payload.get("input_files", [])}

    returns: Dict[str, Dict[int, np.ndarray]] = {}
    for role, entry in inventory.items():
        path = Path(entry["path"])
        if not path.is_file():
            failures.append(f"{role}: missing {path}")
            continue
        checks += 1
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"{role}: hash mismatch")
        if role.endswith(":aggregate"):
            continue
        seed_text, control_id = role.split(":")
        record = json.loads(path.read_text(encoding="utf-8"))
        if control_id == "inverse_only":
            arm = record["persistent_action_noise"]
        else:
            arm = record["arms"]["adapted"]
        values = np.asarray(arm["returns"], dtype=np.float64)
        if values.shape != (EPISODES,):
            failures.append(f"{role}: wrong return count")
            continue
        returns.setdefault(control_id, {})[
            int(seed_text.replace("hopper_seed", ""))
        ] = values

    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)

    for entry in payload.get("per_seed_controller_means", []):
        control_id = entry["control_id"]
        for seed_text, row in entry["per_seed"].items():
            values = returns.get(control_id, {}).get(int(seed_text))
            if values is None:
                failures.append(f"{control_id} seed {seed_text}: no raw returns")
                continue
            checks += 1
            close(float(values.mean()), row["raw_score_mean_recomputed"], f"{control_id}/{seed_text} raw", failures)
            close(
                float(((values - REFERENCE_MIN) * factor).mean()),
                row["normalized_score_mean_recomputed"],
                f"{control_id}/{seed_text} normalized",
                failures,
            )

    for comparison in payload.get("paired_comparisons", []):
        comparison_id = comparison["comparison_id"]
        right_id = RIGHT_ARMS[comparison_id]
        normalized_means = []
        for row in comparison["per_training_seed"]:
            seed = int(row["training_seed"])
            delta = returns["complete"][seed] - returns[right_id][seed]
            checks += 1
            close(float(delta.mean()), row["raw_delta_mean_recomputed"], f"{comparison_id}/{seed} raw", failures)
            normalized = float((delta * factor).mean())
            close(normalized, row["normalized_delta_mean_recomputed"], f"{comparison_id}/{seed} normalized", failures)
            normalized_means.append(normalized)
        expected = cross_seed(normalized_means)
        checks += 3
        close(expected["mean"], comparison["cross_seed_normalized"]["mean"], f"{comparison_id} mean", failures)
        close(expected["t95_low"], comparison["cross_seed_normalized"]["t95_low"], f"{comparison_id} lo", failures)
        close(expected["t95_high"], comparison["cross_seed_normalized"]["t95_high"], f"{comparison_id} hi", failures)

    result = {
        "schema_version": "ca-opex-hopper-multiseed-verify-v1",
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
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
