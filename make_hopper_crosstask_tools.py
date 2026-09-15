"""Deterministic transformation of the frozen Walker2d scale-up tools into the
Hopper-v4 cross-task variants.

Reads run_ca_opex_walker_scaleup.sh and aggregate_ca_opex_walker_scaleup.py,
applies exact string replacements, and writes the Hopper driver and aggregator
with two SHA placeholders that must be filled after the Hopper dataset and
calibration artifacts exist.  The transformation fails loudly if any expected
source pattern is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DATASET_PLACEHOLDER = "__HOPPER_DATASET_SHA256__"
CALIBRATION_PLACEHOLDER = "__HOPPER_CALIBRATION_SHA256__"

DRIVER_REPLACEMENTS = [
    (
        "# One-shot Walker2d seed-expansion driver for the frozen CA-OPEX controller.\n"
        "# This script intentionally has no resume mode. A real run owns one completely\n"
        "# new walker_seed<N> directory; any pre-existing directory is a hard failure.",
        "# One-shot Hopper-v4 cross-task replication driver for the frozen CA-OPEX\n"
        "# controller. Controller hyperparameters, base-learner recipe, and protocol\n"
        "# are carried over from Walker2d without task-specific tuning. This script\n"
        "# intentionally has no resume mode. A real run owns one completely new\n"
        "# hopper_seed<N> directory; any pre-existing directory is a hard failure.",
    ),
    (
        'readonly dataset="/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5"\n'
        'readonly dataset_sha256="159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a"',
        'readonly dataset="/root/hubl_backup_data/action_noise_dev/hopper_v4_beta1_seed2201.hdf5"\n'
        'readonly dataset_sha256="' + DATASET_PLACEHOLDER + '"',
    ),
    (
        'readonly calibration="${research_root}/results/channel_calibration/beta125_n512_seed27001_calibration.json"\n'
        'readonly calibration_sha256="b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354"',
        'readonly calibration="${research_root}/results/channel_calibration/hopper_beta125_n512_seed27002_calibration.json"\n'
        'readonly calibration_sha256="' + CALIBRATION_PLACEHOLDER + '"',
    ),
    (
        'readonly results_root="${CA_OPEX_SCALEUP_RESULTS_ROOT:-${research_root}/results/scaleup}"',
        'readonly results_root="${CA_OPEX_HOPPER_RESULTS_ROOT:-${research_root}/results/crosstask_hopper}"',
    ),
    ('readonly environment="Walker2d-v4"', 'readonly environment="Hopper-v4"'),
    (
        'readonly reference_min="1.629008"\nreadonly reference_max="4592.3"',
        'readonly reference_min="-20.272305"\nreadonly reference_max="3234.3"',
    ),
    (
        "# Exact base learner configuration used by the frozen seed-1/seed-10 block.",
        "# Base learner configuration carried over verbatim from the frozen Walker2d\n"
        "# recipe (no Hopper-specific tuning).",
    ),
    (
        "case \"${seed}\" in\n"
        "  0) die \"training seed 0 is reserved for development and is not a scale-up seed\" ;;\n"
        "  1|10) die \"training seed ${seed} is reserved for formal confirmation and is not a scale-up seed\" ;;\n"
        "esac",
        "case \"${seed}\" in\n"
        "  0) die \"training seed 0 is reserved for development and is not a confirmation seed\" ;;\n"
        "esac",
    ),
    (
        "# Namespace-specific comparisons only: equal numeric starts across distinct RNG\n"
        "# namespaces are valid and intentionally not rejected.\n"
        "reject_reserved_block environment \"${env_seed}\" 39300 79300\n"
        "reject_reserved_block actuator \"${noise_seed}\" 49300 89300\n"
        "reject_reserved_block gradient \"${grad_seed}\" 69300 99300",
        "# Namespace-specific comparisons only: equal numeric starts across distinct RNG\n"
        "# namespaces are valid and intentionally not rejected. All Walker2d blocks\n"
        "# (formal, supplemental, and scale-up) remain reserved so the Hopper namespace\n"
        "# cannot silently reuse them.\n"
        "reject_all_reserved() {\n"
        "  local namespace=\"$1\"\n"
        "  local candidate=\"$2\"\n"
        "  shift 2\n"
        "  local reserved\n"
        "  for reserved in \"$@\"; do\n"
        "    if ranges_overlap \"${candidate}\" \"${reserved}\"; then\n"
        "      die \"${namespace} seed block ${candidate}..$((candidate + 49)) overlaps reserved block ${reserved}..$((reserved + 49))\"\n"
        "    fi\n"
        "  done\n"
        "}\n"
        "reject_all_reserved environment \"${env_seed}\" 39300 79300 131300 131400 131500\n"
        "reject_all_reserved actuator \"${noise_seed}\" 49300 89300 231300 231400 231500\n"
        "reject_all_reserved gradient \"${grad_seed}\" 69300 99300 331300 331400 331500",
    ),
    (
        "require_file \"${code_dir}/aggregate_ca_opex_walker_scaleup.py\"",
        "require_file \"${code_dir}/aggregate_ca_opex_hopper_crosstask.py\"",
    ),
    (
        'readonly run_root="${results_root}/walker_seed${seed}"',
        'readonly run_root="${results_root}/hopper_seed${seed}"',
    ),
    (
        "usage: run_ca_opex_walker_scaleup.sh \\",
        "usage: run_ca_opex_hopper_crosstask.sh \\",
    ),
    (
        'echo "Walker CA-OPEX seed expansion completed in ${run_root}."',
        'echo "Hopper CA-OPEX cross-task replication completed in ${run_root}."',
    ),
    (
        "  \"${python_bin}\" \"${code_dir}/aggregate_ca_opex_walker_scaleup.py\" \\",
        "  \"${python_bin}\" \"${code_dir}/aggregate_ca_opex_hopper_crosstask.py\" \\",
    ),
]

AGGREGATOR_REPLACEMENTS = [
    (
        '"""Fail-closed aggregation for one Walker2d CA-OPEX scale-up seed.',
        '"""Fail-closed aggregation for one Hopper-v4 CA-OPEX cross-task seed.',
    ),
    (
        'SCHEMA = "ca-opex-walker-scaleup-aggregate-v1"',
        'SCHEMA = "ca-opex-hopper-crosstask-aggregate-v1"',
    ),
    (
        'ENVIRONMENT = "Walker2d-v4"',
        'ENVIRONMENT = "Hopper-v4"',
    ),
    (
        "REFERENCE_MIN = 1.629008\nREFERENCE_MAX = 4592.3",
        "REFERENCE_MIN = -20.272305\nREFERENCE_MAX = 3234.3",
    ),
    (
        'EXPECTED_CALIBRATION_SHA256 = (\n'
        '    "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354"\n'
        ")",
        'EXPECTED_CALIBRATION_SHA256 = "' + CALIBRATION_PLACEHOLDER + '"',
    ),
    (
        "RESERVED_TRAINING_SEEDS = frozenset({0, 1, 10})",
        "RESERVED_TRAINING_SEEDS = frozenset({0})",
    ),
    (
        "RESERVED_SEED_BLOCKS = {\n"
        '    "environment": ((39_300, 39_349), (79_300, 79_349)),\n'
        '    "action_noise": ((49_300, 49_349), (89_300, 89_349)),\n'
        '    "gradient_noise": ((69_300, 69_349), (99_300, 99_349)),\n'
        "}",
        "RESERVED_SEED_BLOCKS = {\n"
        '    "environment": (\n'
        "        (39_300, 39_349),\n"
        "        (79_300, 79_349),\n"
        "        (131_300, 131_349),\n"
        "        (131_400, 131_449),\n"
        "        (131_500, 131_549),\n"
        "    ),\n"
        '    "action_noise": (\n'
        "        (49_300, 49_349),\n"
        "        (89_300, 89_349),\n"
        "        (231_300, 231_349),\n"
        "        (231_400, 231_449),\n"
        "        (231_500, 231_549),\n"
        "    ),\n"
        '    "gradient_noise": (\n'
        "        (69_300, 69_349),\n"
        "        (99_300, 99_349),\n"
        "        (331_300, 331_349),\n"
        "        (331_400, 331_449),\n"
        "        (331_500, 331_549),\n"
        "    ),\n"
        "}",
    ),
    (
        'SCALEUP_DRIVER = "run_ca_opex_walker_scaleup.sh"',
        'SCALEUP_DRIVER = "run_ca_opex_hopper_crosstask.sh"',
    ),
    (
        '            f"training seed {train_seed} is reserved by development/formal confirmation"',
        '            f"training seed {train_seed} is reserved for development"',
    ),
]


def apply(source: Path, target: Path, replacements) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"pattern missing from {source.name}:\n{old[:120]}")
        if text.count(old) != 1:
            raise SystemExit(f"pattern not unique in {source.name}: {old[:80]}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target} ({len(text.splitlines())} lines)")


def main() -> int:
    apply(
        HERE / "run_ca_opex_walker_scaleup.sh",
        HERE / "run_ca_opex_hopper_crosstask.sh",
        DRIVER_REPLACEMENTS,
    )
    apply(
        HERE / "aggregate_ca_opex_walker_scaleup.py",
        HERE / "aggregate_ca_opex_hopper_crosstask.py",
        AGGREGATOR_REPLACEMENTS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
