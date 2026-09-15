"""Local convenience: compact summary of the five-training-seed aggregate."""

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"file={path}")
    print(f"status={payload.get('status')} seeds={payload.get('training_seeds')}")
    print("-- per-seed controller means (normalized / raw) --")
    for entry in payload.get("per_seed_controller_means", []):
        control_id = entry["control_id"]
        cells = []
        for seed, row in sorted(entry["per_seed"].items(), key=lambda kv: int(kv[0])):
            cells.append(
                f"s{seed}={row['normalized_score_mean_recomputed']:.4f}"
                f"/{row['raw_score_mean_recomputed']:.1f}"
            )
        descriptive = entry.get("cross_seed_descriptive") or {}
        summary = ""
        if descriptive:
            summary = (
                f" | mean={descriptive['mean']:.4f}"
                f" sd={descriptive['sample_std_ddof1']:.4f}"
                f" t95=[{descriptive['t95_low']:.4f},{descriptive['t95_high']:.4f}]"
            )
        print(f"  {control_id:24s} {' '.join(cells)}{summary}")
    print("-- cross-seed comparisons (normalized deltas) --")
    for comparison in payload.get("paired_comparisons", []):
        cross = comparison["cross_seed_normalized"]
        cross_raw = comparison["cross_seed_raw"]
        per_seed = " ".join(
            f"s{row['training_seed']}="
            f"{row.get('mean', row.get('normalized_delta_mean_recomputed')):+.4f}"
            for row in comparison["per_training_seed"]
        )
        print(
            f"  {comparison['comparison_id']:38s} {per_seed}\n"
            f"    cross: mean={cross['mean']:+.4f} sd={cross['sample_std_ddof1']:.4f}"
            f" t95=[{cross['t95_low']:+.4f},{cross['t95_high']:+.4f}]"
            f" raw_mean={cross_raw['mean']:+.2f}"
            f" positive_seeds={cross['positive_training_seed_count']}"
            f"/{cross['n_training_seeds']}"
            f" crosses_zero={cross['interval_crosses_zero']}"
        )
    wall = payload.get("wall_times", {})
    print("-- wall times --")
    print(f"  base_training_seconds_per_seed={wall.get('base_training_seconds_per_seed')}")
    print(f"  total_base={wall.get('total_base_training_seconds')}")
    print(f"  total_rollout={wall.get('total_recorded_rollout_seconds')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
