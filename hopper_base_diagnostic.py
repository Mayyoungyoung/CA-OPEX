"""Zero-GPU diagnostic: extract base-policy clean/noisy eval stats for Hopper.

Prints, per seed, the clean and persistent-noise evaluation means from the
base summary plus episode-length statistics, to test whether the Hopper base
policy is termination-dominated relative to Walker2d.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOTS = {
    seed: Path(
        f"/root/hubl_research_20260914/results/crosstask_hopper/hopper_seed{seed}"
    )
    for seed in (2, 3, 4)
}
WALKER = Path("/root/hubl_research_20260914/results/scaleup/walker_seed2")


def summarize(path: Path, label: str) -> None:
    if not path.is_file():
        print(f"{label}: missing {path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations") or []
    if not evaluations:
        print(f"{label}: no evaluations block")
        return
    last = evaluations[-1]
    for arm_name in ("clean", "persistent_action_noise"):
        arm = last.get(arm_name)
        if not isinstance(arm, dict):
            continue
        lengths = arm.get("lengths") or []
        returns = arm.get("returns") or []
        length_mean = sum(lengths) / len(lengths) if lengths else float("nan")
        return_mean = sum(returns) / len(returns) if returns else float("nan")
        truncated = sum(1 for value in lengths if value >= 1000)
        print(
            f"{label} {arm_name:26s} n={len(returns):3d} "
            f"return_mean={return_mean:9.2f} length_mean={length_mean:7.1f} "
            f"full_horizon={truncated}/{len(lengths)} "
            f"norm={arm.get('normalized_score_mean')}"
        )


def main() -> int:
    for seed, root in ROOTS.items():
        summarize(root / "base/summary.json", f"hopper_seed{seed}")
    summarize(WALKER / "base/summary.json", "walker_seed2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
