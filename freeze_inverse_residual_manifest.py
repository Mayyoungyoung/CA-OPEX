"""Freeze the strict confirmation manifest through the canonical aggregator.

This compatibility entry point keeps the original explicit artifact arguments,
but delegates every validation and atomic-write decision to
``freeze_strict_manifest_template``.  In particular, the canonical freezer
verifies the development-selection provenance, existing external baseline raw
files, absence of all future OPEX outputs, and every implementation placeholder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from aggregate_inverse_residual_results import freeze_strict_manifest_template


def freeze(args: argparse.Namespace) -> Dict[str, Any]:
    return freeze_strict_manifest_template(
        args.template,
        args.output,
        implementation_root=args.code_dir,
        artifact_paths=(
            args.dataset,
            args.calibration,
            args.base_seed1,
            args.base_seed10,
        ),
        results_root_override=args.results_root,
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "inverse_residual_frozen_confirmation_manifest.template.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "inverse_residual_frozen_confirmation_manifest.json",
    )
    parser.add_argument("--code-dir", type=Path, default=root)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--base-seed1", type=Path, required=True)
    parser.add_argument("--base-seed10", type=Path, required=True)
    return parser


def main() -> int:
    print(json.dumps(freeze(build_parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
