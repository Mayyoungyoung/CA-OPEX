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
        "channel-opex-boundary-extension-provenance-amendment-v3"
    ):
        raise ValueError("unsupported boundary-extension protocol")
    if protocol.get("protocol_created_after_extension_evaluations") is not True:
        raise ValueError("boundary amendment chronology is not explicit")
    if protocol.get("provenance_revalidation_after_evaluations") is not True:
        raise ValueError("boundary protocol lacks explicit post-evaluation amendment label")

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

    pre_evaluation = protocol.get("pre_evaluation_record")
    if not isinstance(pre_evaluation, dict):
        raise ValueError("boundary protocol lacks pre-evaluation provenance")
    pre_protocol_path = _resolve(
        protocol_path.parent, pre_evaluation["protocol_path"]
    )
    pre_selection_path = _resolve(
        protocol_path.parent, pre_evaluation["selection_path"]
    )
    pre_selector_path = _resolve(
        protocol_path.parent, pre_evaluation["selector_path"]
    )
    if sha256_file(pre_protocol_path) != pre_evaluation["protocol_sha256"]:
        raise ValueError("pre-evaluation boundary protocol SHA mismatch")
    if sha256_file(pre_selection_path) != pre_evaluation["selection_sha256"]:
        raise ValueError("pre-evaluation boundary selection SHA mismatch")
    if sha256_file(pre_selector_path) != pre_evaluation["selector_sha256"]:
        raise ValueError("pre-evaluation boundary selector SHA mismatch")
    pre_protocol = read_object(pre_protocol_path)
    if (
        pre_evaluation.get("frozen_before_extension_evaluations") is not True
        or pre_protocol.get("frozen_before_extension_evaluations") is not True
        or pre_protocol.get("schema_version")
        != "channel-opex-boundary-extension-protocol-v1"
        or pre_protocol.get("implementation", {}).get(
            "select_boundary_extension_sha256"
        )
        != pre_evaluation["selector_sha256"]
    ):
        raise ValueError("pre-evaluation protocol/selector provenance mismatch")
    for field in (
        "method",
        "base_checkpoint",
        "channel_calibration",
        "rollout_protocol",
        "extension_output_paths",
    ):
        if protocol.get(field) != pre_protocol.get(field):
            raise ValueError(f"post-evaluation amendment changed numeric field {field}")
    for field in (
        "primary_metric",
        "rule",
        "all_original_and_extension_runs_retained",
    ):
        if protocol.get("selection", {}).get(field) != pre_protocol.get(
            "selection", {}
        ).get(field):
            raise ValueError(f"post-evaluation amendment changed selection field {field}")
    chronology = protocol.get("confirmation_chronology")
    if not isinstance(chronology, dict) or chronology != {
        "selected_ca_opex_confirmation_outputs_read_before_selection": False,
        "selected_adapter_confirmation_outputs_read_before_selection": False,
        "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": True,
        "global_confirmation_blindness_claim": False,
        "historical_v1_confirmation_data_read_field_scope": (
            "The boundary selector process did not read a confirmation result file; "
            "this was not a claim that no external baseline result on those seed "
            "identifiers had previously been observed."
        ),
    }:
        raise ValueError("confirmation chronology is missing or ambiguous")
    historical_selection = read_object(pre_selection_path)
    if (
        historical_selection.get("status") != "complete"
        or historical_selection.get("evidence_label") != "development"
        or historical_selection.get("confirmation_data_read") is not False
        or historical_selection.get("protocol_sha256")
        != pre_evaluation["protocol_sha256"]
        or historical_selection.get("selector_sha256")
        != pre_evaluation["selector_sha256"]
        or historical_selection.get("evaluator_sha256") != evaluator_sha256
    ):
        raise ValueError("pre-evaluation boundary selection metadata mismatch")

    prior = protocol.get("prior_grid")
    if not isinstance(prior, dict):
        raise ValueError("protocol lacks prior-grid provenance")
    prior_protocol_path = _resolve(protocol_path.parent, prior["protocol_path"])
    prior_selection_path = _resolve(protocol_path.parent, prior["selection_path"])
    prior_selector_path = Path(__file__).resolve().with_name(
        "select_channel_opex_dev.py"
    )
    prior_selector_sha256 = sha256_file(prior_selector_path)
    if sha256_file(prior_protocol_path) != prior["protocol_sha256"]:
        raise ValueError("prior protocol SHA mismatch")
    if sha256_file(prior_selection_path) != prior["selection_sha256"]:
        raise ValueError("prior selection SHA mismatch")
    if prior_selector_sha256 != prior["selection_script_sha256"]:
        raise ValueError("prior selector dependency SHA mismatch")
    if prior_selector_sha256 != implementation.get(
        "select_channel_opex_dev_dependency_sha256"
    ):
        raise ValueError("boundary protocol imported-selector SHA mismatch")
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
        historical_records = historical_selection.get("candidates", {}).get(anchor)
        if not isinstance(historical_records, list) or len(historical_records) != len(
            records
        ):
            raise ValueError(f"historical {anchor} candidate count mismatch")
        comparison_fields = (
            "step_size",
            "adapted_normalized_score_mean",
            "baseline_normalized_score_mean",
            "paired_raw_return_difference_mean",
            "raw_sha256",
            "source_grid",
        )
        for current, historical in zip(records, historical_records):
            if any(current.get(field) != historical.get(field) for field in comparison_fields):
                raise ValueError(f"historical {anchor} candidate record mismatch")
        historical_selected = historical_selection.get("selected", {}).get(anchor)
        if not isinstance(historical_selected, dict) or any(
            selections[anchor].get(field) != historical_selected.get(field)
            for field in comparison_fields
        ):
            raise ValueError(f"historical selected {anchor} mismatch")

    result = {
        "schema_version": (
            "channel-opex-boundary-extension-selection-revalidated-v3"
        ),
        "status": "complete",
        "evidence_label": "development",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selector_sha256": selector_sha256,
        "evaluator_sha256": evaluator_sha256,
        "prior_selection_path": str(prior_selection_path),
        "prior_selection_sha256": prior["selection_sha256"],
        "prior_selector_dependency_sha256": prior_selector_sha256,
        "provenance_revalidation_after_evaluations": True,
        "pre_evaluation_protocol_path": str(pre_protocol_path),
        "pre_evaluation_protocol_sha256": pre_evaluation["protocol_sha256"],
        "pre_evaluation_selection_path": str(pre_selection_path),
        "pre_evaluation_selection_sha256": pre_evaluation["selection_sha256"],
        "pre_evaluation_selector_path": str(pre_selector_path),
        "pre_evaluation_selector_sha256": pre_evaluation["selector_sha256"],
        "selection_rule": protocol["selection"]["rule"],
        "candidates": candidates,
        "selected": selections,
        "selected_ca_opex_confirmation_outputs_read_before_selection": False,
        "selected_adapter_confirmation_outputs_read_before_selection": False,
        "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": True,
        "global_confirmation_blindness_claim": False,
        "historical_v1_confirmation_data_read_field_scope": chronology[
            "historical_v1_confirmation_data_read_field_scope"
        ],
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
