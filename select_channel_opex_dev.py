"""Validate a predeclared CA-OPEX development grid and freeze its selections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from train_inverse_residual_adapter import require_new_output_file
from train_iql import save_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _expected_path(protocol_path: Path, relative: str) -> Path:
    return (protocol_path.parent / relative).resolve()


def _validate_candidate(
    payload: Dict[str, Any],
    *,
    anchor: str,
    step_size: float,
    protocol: Dict[str, Any],
    evaluator_sha256: str,
) -> float:
    if payload.get("raw_schema") != "channel_opex_v1" or payload.get(
        "status"
    ) != "complete":
        raise ValueError("candidate is not a complete channel_opex_v1 record")
    implementation = payload.get("implementation")
    if not isinstance(implementation, dict) or implementation.get(
        "evaluate_sha256"
    ) != evaluator_sha256:
        raise ValueError("candidate evaluator implementation hash mismatch")
    method = protocol["method"]
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("candidate lacks controller metadata")
    expected = {
        "baseline_transform": anchor,
        "gradient_steps": int(method["gradient_steps"]),
        "K": int(method["gradient_noise_samples"]),
        "delta_max": float(method["delta_max_linf_by_anchor"][anchor]),
        "step_size": float(step_size),
        "q_reducer": str(method["q_reducer"]),
    }
    for field, value in expected.items():
        observed = controller.get(field)
        if isinstance(value, float):
            if not np.isclose(float(observed), value, rtol=0.0, atol=1e-12):
                raise ValueError(f"candidate controller {field} mismatch")
        elif observed != value:
            raise ValueError(f"candidate controller {field} mismatch")
    calibration = protocol["channel_calibration"]
    if not np.isclose(
        float(controller.get("model_beta")),
        float(calibration["estimated_beta"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("candidate calibrated beta mismatch")
    base = payload.get("base_checkpoint")
    if not isinstance(base, dict) or base.get("sha256") != protocol[
        "base_checkpoint"
    ]["sha256"]:
        raise ValueError("candidate base checkpoint mismatch")
    recorded_calibration = payload.get("calibration")
    if not isinstance(recorded_calibration, dict) or recorded_calibration.get(
        "calibration_sha256"
    ) != calibration["sha256"]:
        raise ValueError("candidate calibration artifact mismatch")

    rollout = protocol["rollout_protocol"]
    evaluation = payload.get("evaluation_protocol")
    channel = payload.get("channel")
    if not isinstance(evaluation, dict) or not isinstance(channel, dict):
        raise ValueError("candidate lacks rollout/channel metadata")
    protocol_checks = {
        "environment": rollout["environment"],
        "episode_count_per_arm": int(rollout["episode_count_per_arm"]),
        "environment_seed_start": int(rollout["environment_seed_start"]),
        "action_noise_seed_start": int(rollout["action_noise_seed_start"]),
        "gradient_noise_seed_start": int(rollout["gradient_noise_seed_start"]),
    }
    for field, value in protocol_checks.items():
        if evaluation.get(field) != value:
            raise ValueError(f"candidate evaluation protocol {field} mismatch")
    if evaluation.get("gradient_noise_sampling_frequency") != "per_gradient_step":
        raise ValueError("candidate gradient-noise sampling frequency mismatch")
    if not np.isclose(
        float(channel.get("environment_rollout_beta")),
        float(rollout["environment_rollout_beta"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("candidate rollout beta mismatch")
    arms = payload.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("candidate lacks arms")
    adapted = arms.get("adapted")
    baseline = arms.get("baseline_only")
    if not isinstance(adapted, dict) or not isinstance(baseline, dict):
        raise ValueError("candidate lacks paired adapted/baseline arms")
    episode_count = int(rollout["episode_count_per_arm"])
    expected_environment_seeds = list(
        range(
            int(rollout["environment_seed_start"]),
            int(rollout["environment_seed_start"]) + episode_count,
        )
    )
    expected_action_noise_seeds = list(
        range(
            int(rollout["action_noise_seed_start"]),
            int(rollout["action_noise_seed_start"]) + episode_count,
        )
    )
    for arm in (baseline, adapted):
        if arm.get("environment_seeds") != expected_environment_seeds:
            raise ValueError("candidate environment seed array mismatch")
        if arm.get("action_noise_seeds") != expected_action_noise_seeds:
            raise ValueError("candidate action-noise seed array mismatch")
        if len(arm.get("returns", [])) != episode_count:
            raise ValueError("candidate return count mismatch")
    if adapted.get("gradient_noise_sampling_frequency") != "per_gradient_step":
        raise ValueError("adapted arm gradient-noise sampling frequency mismatch")
    cost = payload.get("cost")
    if not isinstance(cost, dict) or cost.get("cost_scope") != (
        "adapted_arm_deployment_controller_only"
    ):
        raise ValueError("candidate deployment-cost scope mismatch")
    score = float(adapted["normalized_score_mean"])
    if not np.isfinite(score):
        raise ValueError("candidate score is not finite")
    return score


def run(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_path = args.protocol.resolve()
    output_path = require_new_output_file(args.output)
    protocol = read_object(protocol_path)
    if protocol.get("schema_version") != "channel-opex-development-protocol-v1":
        raise ValueError("unsupported development protocol")
    if protocol.get("frozen_before_any_listed_evaluation") is not True:
        raise ValueError("protocol does not declare pre-evaluation freezing")
    implementation_contract = protocol.get("implementation")
    if not isinstance(implementation_contract, dict):
        raise ValueError("protocol lacks implementation contract")
    evaluator_path = Path(__file__).resolve().with_name("evaluate_channel_opex.py")
    evaluator_sha256 = sha256_file(evaluator_path)
    if implementation_contract.get("evaluate_channel_opex_sha256") != (
        evaluator_sha256
    ):
        raise ValueError("protocol evaluator implementation hash mismatch")
    output_specs = protocol.get("output_paths")
    if not isinstance(output_specs, dict):
        raise ValueError("protocol lacks output paths")
    candidates: Dict[str, List[Dict[str, Any]]] = {
        "inverse": [],
        "identity": [],
    }
    for key, relative in output_specs.items():
        anchor, marker = key.split("_step_size_", 1)
        if anchor not in candidates:
            raise ValueError(f"unsupported anchor in output key {key!r}")
        step_size = float(marker)
        path = _expected_path(protocol_path, str(relative))
        payload = read_object(path)
        score = _validate_candidate(
            payload,
            anchor=anchor,
            step_size=step_size,
            protocol=protocol,
            evaluator_sha256=evaluator_sha256,
        )
        candidates[anchor].append(
            {
                "step_size": step_size,
                "adapted_normalized_score_mean": score,
                "baseline_normalized_score_mean": float(
                    payload["arms"]["baseline_only"]["normalized_score_mean"]
                ),
                "paired_raw_return_difference_mean": float(
                    payload["paired"]["return_difference_mean"]
                ),
                "path": str(path),
                "raw_sha256": sha256_file(path),
            }
        )
    selections: Dict[str, Dict[str, Any]] = {}
    for anchor, records in candidates.items():
        ordered = sorted(
            records,
            key=lambda item: (
                -item["adapted_normalized_score_mean"],
                item["step_size"],
            ),
        )
        selections[anchor] = ordered[0]
        candidates[anchor] = sorted(records, key=lambda item: item["step_size"])
    result = {
        "schema_version": "channel-opex-development-selection-v1",
        "status": "complete",
        "evidence_label": "development",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selection_rule": protocol["selection"]["rule"],
        "candidates": candidates,
        "selected": selections,
        "selection_script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_sha256": evaluator_sha256,
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
