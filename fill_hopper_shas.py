"""Fill the Hopper driver/aggregator SHA placeholders after dataset creation.

Reads the dataset and calibration SHA-256 sidecar files, replaces the two
placeholder tokens in run_ca_opex_hopper_crosstask.sh and in
aggregate_ca_opex_hopper_crosstask.py, refuses to run if any placeholder is
missing or already filled, and prints the resulting file hashes for the record.
"""

import hashlib
import sys
from pathlib import Path

CODE = Path("/root/hubl_research_20260914/code")
DATA = Path("/root/hubl_backup_data")
DATASET = DATA / "action_noise_dev/hopper_v4_beta1_seed2201.hdf5"
CALIBRATION = (
    Path("/root/hubl_research_20260914/results/channel_calibration/"
         "hopper_beta125_n512_seed27002_calibration.json")
)
TARGETS = [
    CODE / "run_ca_opex_hopper_crosstask.sh",
    CODE / "aggregate_ca_opex_hopper_crosstask.py",
]
DATASET_PLACEHOLDER = "__HOPPER_DATASET_SHA256__"
CALIBRATION_PLACEHOLDER = "__HOPPER_CALIBRATION_SHA256__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for path in (DATASET, CALIBRATION):
        if not path.is_file():
            print(f"missing required artifact: {path}")
            return 1
    dataset_sha = sha256_file(DATASET)
    calibration_sha = sha256_file(CALIBRATION)
    print(f"dataset_sha256={dataset_sha}")
    print(f"calibration_sha256={calibration_sha}")

    driver = (CODE / "run_ca_opex_hopper_crosstask.sh").read_text(encoding="utf-8")
    if DATASET_PLACEHOLDER not in driver or CALIBRATION_PLACEHOLDER not in driver:
        print("driver placeholders already filled or missing; refusing to touch")
        return 1

    for target in TARGETS:
        text = target.read_text(encoding="utf-8")
        text = text.replace(DATASET_PLACEHOLDER, dataset_sha)
        text = text.replace(CALIBRATION_PLACEHOLDER, calibration_sha)
        if DATASET_PLACEHOLDER in text or CALIBRATION_PLACEHOLDER in text:
            print(f"placeholder left in {target}; aborting")
            return 1
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"filled {target.name} sha256={sha256_file(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
