"""Exploratory helper: extract pinned hashes for multi-seed aggregator inputs.

Read-only: prints formal-aggregate input hashes for the six formal records and
live hashes of the four supplemental holdout records, plus seed-2 progress.
"""

import hashlib
import json
import sys
from pathlib import Path

FORMAL_AGG = Path(
    "/root/hubl_research_20260914/results/inverse_residual_confirm/aggregate/"
    "frozen_confirmation.json"
)
HOLDOUT_DIR = Path(
    "/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout"
)
FORMAL_FILES = [
    "ca_opex_inverse_seed1_beta125_50.json",
    "ca_opex_inverse_seed10_beta125_50.json",
    "ca_opex_identity_seed1_beta125_50.json",
    "ca_opex_identity_seed10_beta125_50.json",
    "opex_original_seed1_beta125_50.json",
    "opex_original_seed10_beta125_50.json",
]
HOLDOUT_FILES = [
    "nominal_tuned_seed1_fresh50.json",
    "nominal_tuned_seed10_fresh50.json",
    "complete_calibrated_inverse_eta_0.1_seed1_fresh50.json",
    "complete_calibrated_inverse_eta_0.1_seed10_fresh50.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    payload = json.loads(FORMAL_AGG.read_text(encoding="utf-8"))
    by_name = {}
    for entry in payload["input_files"]:
        by_name[Path(entry["path"]).name] = entry
    print("== formal pinned hashes (from formal aggregate) ==")
    for name in FORMAL_FILES:
        entry = by_name.get(name)
        if entry is None:
            print(f"MISSING_IN_FORMAL_AGGREGATE {name}")
            continue
        live = Path(entry["path"])
        live_sha = sha256_file(live) if live.is_file() else "ABSENT"
        match = "OK" if live_sha == entry["sha256"] else "MISMATCH"
        print(f"{name}\n  pinned={entry['sha256']}\n  live  ={live_sha} [{match}]")

    print("\n== supplemental holdout live hashes ==")
    for name in HOLDOUT_FILES:
        path = HOLDOUT_DIR / name
        if not path.is_file():
            print(f"ABSENT {name}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        adapted = record.get("arms", {}).get("adapted", {})
        rets = adapted.get("returns", [])
        print(
            f"{name}\n  sha256={sha256_file(path)}\n"
            f"  adapted_mean={sum(rets) / len(rets):.4f} n={len(rets)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
