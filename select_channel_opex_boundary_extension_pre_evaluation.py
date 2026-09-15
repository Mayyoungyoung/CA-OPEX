"""Validate a locked CA-OPEX boundary extension and freeze the union selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from select_channel_opex_dev import (
    _validate_candidate,
    read_object,
    sha256_file,
)
from train_inverse_residual_adapter import require_new_output_file
from train_iql import save_json


def _resolve(base: Path, relative: str) -> Path:
    return (base / relative).resolve()


def _record_candidate(
    path: Path,
    *,
    anchor: str,
    step_size: float,
    protocol: Dict[str, Any],
    evaluator_sha256: str,
    source_grid: str,
) -> Dict[str, Any]:
    payload = read_object(path)
    score = _validate_candidate(
        payload,
        anchor=anchor,
        step_size=step_size,
        protocol=protocol,
        evaluator_sha256=evaluator_sha256,
    )
    return {
        "step_size": float(step_size),
        "adapted_normalized_score_mean": float(score),
        "baseline_normalized_score_mean": float(
            payload["arms"]["baseline_only"]["normalized_score_mean"]
        ),
        "paired_raw_return_difference_mean": float(
            payload["paired"]["return_difference_mean"]
        ),
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "source_grid": source_grid,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_path = args.protocol.resolve()
    output_path = require_new_output_file(args.output)
    protocol = read_object(protocol_path)
    if protocol.get("schema_version") != (
        "channel-opex-boundary-extension-protocol-v1"
    ):
        raise ValueError("unsupported boundary-extension protocol")
    if protocol.get("frozen_before_extension_evaluations") is not True:
        raise ValueError("extension protocol is not declared pre-evaluation frozen")

    implementation = protocol.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("protocol lacks implementation contract")
    evaluator_path = Path(__file__).resolve().with_name("evaluate_channel_opex.py")
    evaluator_sha256 = sha256_file(evaluator_path)
    selector_sha256 = sha256_file(Path(__file__).resolve())
    if implementation.get("evaluate_channel_opex_sha256") != evaluator_sha256:
        raise ValueError("boundary protocol evaluator SHA mismatch")
    if implementation.get("select_boundary_extension_sha256") != selector_sha256:
        raise ValueError("boundary protocol selector SHA mismatch")

    prior = protocol.get("prior_grid")
    if not isinstance(prior, dict):
        raise ValueError("protocol lacks prior-grid provenance")
    prior_protocol_path = _resolve(protocol_path.parent, prior["protocol_path"])
    prior_selection_path = _resolve(protocol_path.parent, prior["selection_path"])
    if sha256_file(prior_protocol_path) != prior["protocol_sha256"]:
        raise ValueError("prior protocol SHA mismatch")
    if sha256_file(prior_selection_path) != prior["selection_sha256"]:
        raise ValueError("prior selection SHA mismatch")
    prior_selection = read_object(prior_selection_path)
    if (
        prior_selection.get("status") != "complete"
        or prior_selection.get("evidence_label") != "development"
        or prior_selection.get("confirmation_data_read") is not False
        or prior_selection.get("protocol_sha256") != prior["protocol_sha256"]
        or prior_selection.get("selection_script_sha256")
        != prior["selection_script_sha256"]
        or prior_selection.get("evaluator_sha256") != evaluator_sha256
    ):
        raise ValueError("prior selection metadata mismatch")
    for anchor in ("inverse", "identity"):
        selected = prior_selection.get("selected", {}).get(anchor)
        if not isinstance(selected, dict) or float(selected["step_size"]) != float(
            prior[f"selected_{anchor}_step_size"]
        ):
            raise ValueError(f"prior selected {anchor} step size mismatch")

    candidates: Dict[str, List[Dict[str, Any]]] = {"inverse": [], "identity": []}
    expected_original_grid = [
        float(value) for value in protocol["method"]["original_step_size_grid"]
    ]
    for anchor in ("inverse", "identity"):
        prior_records = prior_selection.get("candidates", {}).get(anchor)
        if not isinstance(prior_records, list):
            raise ValueError(f"prior selection lacks {anchor} candidates")
        if [float(item["step_size"]) for item in prior_records] != expected_original_grid:
            raise ValueError(f"prior {anchor} grid differs from extension contract")
        for item in prior_records:
            raw_path = Path(str(item["path"])).resolve()
            if not raw_path.is_file() or sha256_file(raw_path) != item["raw_sha256"]:
                raise ValueError(f"prior {anchor} raw artifact mismatch")
            record = _record_candidate(
                raw_path,
                anchor=anchor,
                step_size=float(item["step_size"]),
                protocol=protocol,
                evaluator_sha256=evaluator_sha256,
                source_grid="original",
            )
            if record["raw_sha256"] != item["raw_sha256"]:
                raise ValueError(f"prior {anchor} candidate SHA mismatch")
            candidates[anchor].append(record)

    extension_paths = protocol.get("extension_output_paths")
    if not isinstance(extension_paths, dict):
        raise ValueError("protocol lacks extension output paths")
    for key, relative in extension_paths.items():
        anchor, marker = key.split("_step_size_", 1)
        if anchor not in candidates:
            raise ValueError(f"unsupported extension anchor {anchor!r}")
        candidates[anchor].append(
            _record_candidate(
                _resolve(protocol_path.parent, str(relative)),
                anchor=anchor,
                step_size=float(marker),
                protocol=protocol,
                evaluator_sha256=evaluator_sha256,
                source_grid="boundary_extension",
            )
        )

    expected_combined = [
        float(value) for value in protocol["method"]["combined_step_size_grid"]
    ]
    selections: Dict[str, Dict[str, Any]] = {}
    for anchor, records in candidates.items():
        records.sort(key=lambda item: item["step_size"])
        if [item["step_size"] for item in records] != expected_combined:
            raise ValueError(f"combined {anchor} grid mismatch")
        selections[anchor] = sorted(
            records,
            key=lambda item: (
                -item["adapted_normalized_score_mean"],
                item["step_size"],
            ),
        )[0]

    result = {
        "schema_version": "channel-opex-boundary-extension-selection-v1",
        "status": "complete",
        "evidence_label": "development",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selector_sha256": selector_sha256,
        "evaluator_sha256": evaluator_sha256,
        "prior_selection_path": str(prior_selection_path),
        "prior_selection_sha256": prior["selection_sha256"],
        "selection_rule": protocol["selection"]["rule"],
        "candidates": candidates,
        "selected": selections,
        "confirmation_data_read": False,
    }
    save_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
