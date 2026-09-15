"""Fail-closed three-training-seed summary for the Hopper-v4 cross-task run.

Combines the per-seed Hopper aggregates (seeds 2/3/4) into cross-seed trainer
statistics.  Every mean and interval is recomputed from the per-episode
returns in the raw control records; the 150 rollout episodes are never pooled
into one interval.  Output is create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy.stats import t as student_t

SCHEMA = "ca-opex-hopper-multiseed-aggregate-v1"
EPISODES = 50
REFERENCE_MIN = -20.272305
REFERENCE_MAX = 3234.3
SCALEUP_AGGREGATE_SCHEMA = "ca-opex-hopper-crosstask-aggregate-v1"
TRAINING_SEEDS = (2, 3, 4)
SCALEUP_CONTROL_FILES = {
    "complete": "controls/complete.json",
    "calibrated_identity": "controls/calibrated_identity.json",
    "nominal_k8t2": "controls/nominal_k8t2.json",
    "original_opex": "controls/original_opex.json",
    "inverse_only": "controls/inverse_only.json",
}
COMPARISONS = (
    ("complete_minus_calibrated_identity", "calibrated_identity"),
    ("complete_minus_nominal_k8t2", "nominal_k8t2"),
    ("complete_minus_original_opex", "original_opex"),
    ("complete_minus_inverse_only", "inverse_only"),
)


class AggregationError(RuntimeError):
    """Raised when an input violates the Hopper multi-seed contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregationError(f"JSON root is not an object: {path}")
    return value


def _returns(record: Mapping[str, Any], *, inverse_only: bool) -> np.ndarray:
    if inverse_only:
        arm = record["persistent_action_noise"]
    else:
        arm = record["arms"]["adapted"]
    values = np.asarray(arm["returns"], dtype=np.float64)
    if values.shape != (EPISODES,) or not np.isfinite(values).all():
        raise AggregationError("raw returns are not 50 finite values")
    return values


def _cross_seed(values: List[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(array.mean())
    standard = float(array.std(ddof=1))
    standard_error = standard / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    return {
        "n_training_seeds": n,
        "mean": mean,
        "sample_std_ddof1": standard,
        "t95_low": mean - critical * standard_error,
        "t95_high": mean + critical * standard_error,
        "positive_training_seed_count": int((array > 0.0).sum()),
        "interval_crosses_zero": bool(
            mean - critical * standard_error <= 0.0 <= mean + critical * standard_error
        ),
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


def aggregate(results_root: Path, output: Path) -> Dict[str, Any]:
    results_root = results_root.resolve()
    output = output.resolve()
    _require_new_output(output)
    factor = 100.0 / (REFERENCE_MAX - REFERENCE_MIN)

    arms: Dict[str, Dict[int, np.ndarray]] = {}
    input_files: List[Dict[str, Any]] = []
    env_blocks: Dict[int, int] = {}
    for seed in TRAINING_SEEDS:
        run_root = results_root / f"hopper_seed{seed}"
        aggregate_path = run_root / "aggregate.json"
        if not aggregate_path.is_file():
            raise AggregationError(f"hopper seed {seed} aggregate is missing")
        per_seed = read_json(aggregate_path)
        if (
            per_seed.get("schema_version") != SCALEUP_AGGREGATE_SCHEMA
            or per_seed.get("status") != "complete"
            or per_seed.get("training_seed") != seed
            or per_seed.get("environment") != "Hopper-v4"
        ):
            raise AggregationError(f"hopper seed {seed} aggregate contract mismatch")
        env_blocks[seed] = per_seed["evaluation_protocol"]["environment_seeds"][0]
        input_files.append(
            {
                "role": f"hopper_seed{seed}:aggregate",
                "path": str(aggregate_path),
                "sha256": sha256_file(aggregate_path),
                "size_bytes": aggregate_path.stat().st_size,
            }
        )
        for control_id, relative in SCALEUP_CONTROL_FILES.items():
            path = run_root / relative
            if not path.is_file():
                raise AggregationError(f"hopper seed {seed} is missing {control_id}")
            record = read_json(path)
            arms.setdefault(control_id, {})[seed] = _returns(
                record, inverse_only=control_id == "inverse_only"
            )
            input_files.append(
                {
                    "role": f"hopper_seed{seed}:{control_id}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if len(set(env_blocks.values())) != len(env_blocks):
        raise AggregationError(f"hopper seeds share environment blocks: {env_blocks}")

    per_seed_table = []
    for control_id, seed_map in arms.items():
        rows = {
            str(seed): {
                "raw_score_mean_recomputed": float(values.mean()),
                "normalized_score_mean_recomputed": float(
                    ((values - REFERENCE_MIN) * factor).mean()
                ),
            }
            for seed, values in sorted(seed_map.items())
        }
        normalized_means = [
            rows[str(seed)]["normalized_score_mean_recomputed"]
            for seed in sorted(seed_map)
        ]
        per_seed_table.append(
            {
                "control_id": control_id,
                "per_seed": rows,
                "cross_seed_descriptive": _cross_seed(normalized_means),
            }
        )

    comparisons = []
    for comparison_id, right_id in COMPARISONS:
        per_seed_rows = []
        normalized_means = []
        raw_means = []
        for seed in TRAINING_SEEDS:
            delta = arms["complete"][seed] - arms[right_id][seed]
            normalized_means.append(float((delta * factor).mean()))
            raw_means.append(float(delta.mean()))
            per_seed_rows.append(
                {
                    "training_seed": seed,
                    "raw_delta_mean_recomputed": float(delta.mean()),
                    "normalized_delta_mean_recomputed": float((delta * factor).mean()),
                    "positive_episode_count": int((delta > 0.0).sum()),
                }
            )
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "left": "complete",
                "right": right_id,
                "per_training_seed": per_seed_rows,
                "cross_seed_normalized": _cross_seed(normalized_means),
                "cross_seed_raw": _cross_seed(raw_means),
                "paired_unit": "episode_case_within_one_fixed_checkpoint",
                "cross_seed_unit": "independent_base_policy_training_seed",
            }
        )

    report: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "complete",
        "results_root": str(results_root),
        "environment": "Hopper-v4",
        "training_seeds": list(TRAINING_SEEDS),
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
            "source": "d4rl/infos.py hopper random/expert pair (GitHub master)",
            "scores_recomputed_from_episode_returns": True,
        },
        "blocking_note": (
            "Cross-seed statistics use one mean per independent base-policy "
            "training seed (n=3); the 150 rollout episodes are never pooled."
        ),
        "per_seed_controller_means": per_seed_table,
        "paired_comparisons": comparisons,
        "input_files": input_files,
        "aggregator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_new(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/root/hubl_research_20260914/results/crosstask_hopper"),
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
