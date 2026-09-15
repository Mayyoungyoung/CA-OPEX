"""Local convenience: print a compact summary of one scale-up aggregate JSON."""

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"file={path}")
    print(f"status={payload.get('status')} seed={payload.get('training_seed')}")
    print("-- absolute scores --")
    for row in payload.get("absolute_scores", []):
        print(
            f"  {row['control_id']:22s} raw={row['raw_score_mean_recomputed']:9.4f}"
            f" norm={row['normalized_score_mean_recomputed']:8.4f}"
        )
    print("-- paired comparisons (normalized deltas) --")
    for row in payload.get("paired_comparisons", []):
        print(
            f"  {row['comparison_id']:38s} mean={row['mean']:+8.4f}"
            f" t95=[{row['t95_low']:+.4f},{row['t95_high']:+.4f}]"
            f" boot95=[{row['bootstrap95_low']:+.4f},{row['bootstrap95_high']:+.4f}]"
            f" pos={row['positive_episode_count']}/50"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
