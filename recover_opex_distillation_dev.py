"""Revalidate and summarize the completed OPEX-distillation development run.

The original single-use runner finished training, then stopped at a post-training
assertion because its frozen protocol contained transcription errors in two
deterministic split SHA declarations.  This recovery never trains or
evaluates a policy.  It verifies the immutable original protocol, all completed
artifacts, the deterministic split recomputed from the source dataset, and the
already-written paired evaluation before atomically writing one comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from train_inverse_residual_adapter import build_observation_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def require_hash(path: Path, expected: str, role: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{role} SHA mismatch: expected={expected}, actual={actual}"
        )
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def verify_arm(
    arm: Mapping[str, Any],
    *,
    expected_env: list[int],
    expected_noise: list[int],
    reference_min: float,
    reference_max: float,
) -> list[float]:
    if arm.get("environment_seeds") != expected_env:
        raise ValueError("environment seed array mismatch")
    if arm.get("action_noise_seeds") != expected_noise:
        raise ValueError("action-noise seed array mismatch")
    returns = np.asarray(arm.get("returns"), dtype=np.float64)
    if returns.shape != (len(expected_env),) or not np.all(np.isfinite(returns)):
        raise ValueError("return array is incomplete or non-finite")
    mean = float(np.mean(returns))
    normalized = 100.0 * (mean - reference_min) / (reference_max - reference_min)
    if not close(arm.get("return_mean"), mean):
        raise ValueError("reported return mean does not match raw returns")
    if not close(arm.get("normalized_score_mean"), normalized):
        raise ValueError("reported normalized mean does not match raw returns")
    return returns.tolist()


def run(protocol_path: Path, output_path: Path) -> Dict[str, Any]:
    protocol_path = protocol_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite recovery output: {output_path}")
    protocol = read_object(protocol_path)
    if protocol.get("schema_version") != "opex-distillation-recovery-protocol-v1":
        raise ValueError("unsupported recovery protocol")
    if (
        protocol.get("created_after_training_failure") is not True
        or protocol.get("confirmation_evidence_eligible") is not False
        or protocol.get("rerun_training_or_evaluation") is not False
    ):
        raise ValueError("recovery chronology/scope is missing")

    code_dir = protocol_path.parent
    implementation_sha = sha256_file(Path(__file__).resolve())
    if implementation_sha != protocol["recovery_implementation_sha256"]:
        raise ValueError("recovery implementation SHA mismatch")
    sources = [
        require_hash(
            Path(__file__).resolve(), implementation_sha, "recovery_implementation"
        )
    ]
    resolved: Dict[str, Path] = {}
    for role, spec in protocol["sources"].items():
        path = Path(spec["path"]).resolve()
        resolved[role] = path
        sources.append(require_hash(path, spec["sha256"], role))

    original_protocol = read_object(resolved["original_protocol"])
    original_runner_sha = sha256_file(resolved["original_runner"])
    original_split = original_protocol["training"]["observation_split"]
    if (
        original_protocol.get("schema_version")
        != "opex-distillation-development-protocol-v1"
        or original_runner_sha != protocol["sources"]["original_runner"]["sha256"]
        or original_split["expected_train_indices_sha256"]
        != protocol["failure"]["mistyped_expected_train_indices_sha256"]
        or original_split["expected_audit_indices_sha256"]
        != protocol["failure"]["mistyped_expected_audit_indices_sha256"]
    ):
        raise ValueError("original frozen failure record is inconsistent")
    failure_log = resolved["driver_failure_log"].read_text(
        encoding="utf-8", errors="replace"
    )
    if "AssertionError" not in failure_log or "line 20" not in failure_log:
        raise ValueError("original post-training failure log mismatch")
    for filename, expected in original_protocol["locked_implementation"].items():
        sources.append(
            require_hash(code_dir / filename, expected, f"locked_implementation:{filename}")
        )

    config = read_object(resolved["training_config"])
    summary = read_object(resolved["training_summary"])
    evaluation = read_object(resolved["student_evaluation"])
    teacher = read_object(resolved["teacher_evaluation"])
    if summary.get("status") != "complete" or summary.get("updates") != 5000:
        raise ValueError("training did not complete the frozen 5k target")
    if summary.get("checkpoint_sha256") != protocol["sources"]["checkpoint"]["sha256"]:
        raise ValueError("summary checkpoint hash mismatch")
    if summary.get("base_parameters_unchanged") is not True:
        raise ValueError("base actor/critic changed during distillation")
    if summary.get("teacher_accounting", {}).get("q1_input_rows") != 20_480_000:
        raise ValueError("teacher Q-row accounting mismatch")
    if summary.get("deployment_q1_rows_per_action") != 0:
        raise ValueError("student deployment-cost metadata mismatch")

    dataset = resolved["dataset"]
    _, train_indices, audit_indices, split = build_observation_split(
        dataset, None, 0.1, 424242
    )
    corrected = protocol["corrected_split"]
    expected_split = {
        "train_observation_count": 899_940,
        "audit_observation_count": 100_060,
        "train_indices_sha256": corrected["train_indices_sha256"],
        "audit_indices_sha256": corrected["audit_indices_sha256"],
        "audit_episode_units_sha256": corrected["audit_episode_units_sha256"],
        "audit_episode_unit_count": 303,
    }
    if train_indices.size != 899_940 or audit_indices.size != 100_060:
        raise ValueError("recomputed split counts mismatch")
    for field, expected in expected_split.items():
        if split.get(field) != expected:
            raise ValueError(f"recomputed split {field} mismatch")
        if config.get("observation_split", {}).get(field) != expected:
            raise ValueError(f"training config split {field} mismatch")
        if summary.get("observation_split", {}).get(field) != expected:
            raise ValueError(f"training summary split {field} mismatch")

    arguments = config.get("arguments", {})
    expected_arguments = {
        "train_seed": 0,
        "updates": 5000,
        "batch_size": 256,
        "hidden_dim": 128,
        "depth": 2,
        "learning_rate": 0.0003,
        "delta_max": 0.25,
        "teacher_step_size": 0.1,
        "teacher_k": 8,
        "teacher_gradient_steps": 2,
        "teacher_noise_seed": 271828,
        "audit_fraction": 0.1,
        "split_seed": 424242,
    }
    for field, expected in expected_arguments.items():
        observed = arguments.get(field)
        if isinstance(expected, float):
            if not close(observed, expected, 1e-12):
                raise ValueError(f"training argument {field} mismatch")
        elif observed != expected:
            raise ValueError(f"training argument {field} mismatch")

    expected_env = list(range(28300, 28310))
    expected_noise = list(range(38300, 38310))
    reference_min = 1.629008
    reference_max = 4592.3
    if evaluation.get("status") != "complete" or teacher.get("status") != "complete":
        raise ValueError("paired evaluation is incomplete")
    if evaluation.get("environment") != "Walker2d-v4":
        raise ValueError("student environment mismatch")
    if not close(evaluation.get("channel", {}).get("environment_rollout_beta"), 1.25):
        raise ValueError("student rollout beta mismatch")
    inverse_returns = verify_arm(
        evaluation["baseline_only"],
        expected_env=expected_env,
        expected_noise=expected_noise,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    student_returns = verify_arm(
        evaluation["adapted"],
        expected_env=expected_env,
        expected_noise=expected_noise,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    teacher_inverse = verify_arm(
        teacher["arms"]["baseline_only"],
        expected_env=expected_env,
        expected_noise=expected_noise,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    teacher_returns = verify_arm(
        teacher["arms"]["adapted"],
        expected_env=expected_env,
        expected_noise=expected_noise,
        reference_min=reference_min,
        reference_max=reference_max,
    )
    if not np.allclose(inverse_returns, teacher_inverse, rtol=0.0, atol=1e-6):
        raise ValueError("inverse anchor is not bit-compatible across evaluations")
    paired = np.asarray(student_returns) - np.asarray(inverse_returns)
    if not close(evaluation.get("paired", {}).get("return_difference_mean"), paired.mean()):
        raise ValueError("student paired mean mismatch")

    inverse_mean = float(np.mean(inverse_returns))
    student_mean = float(np.mean(student_returns))
    teacher_mean = float(np.mean(teacher_returns))
    teacher_gain = teacher_mean - inverse_mean
    recovery_fraction = (
        (student_mean - inverse_mean) / teacher_gain if teacher_gain != 0.0 else None
    )
    payload: Dict[str, Any] = {
        "schema_version": "opex-distillation-development-recovery-v2",
        "status": "complete",
        "evidence_label": "development",
        "confirmation_evidence_eligible": False,
        "post_training_provenance_recovery": True,
        "rerun_training_or_evaluation": False,
        "failure_corrected": protocol["failure"],
        "recovery_protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "verified_sources": sources,
        "corrected_split": split,
        "evaluation_protocol": {
            "environment": "Walker2d-v4",
            "rollout_beta": 1.25,
            "environment_seeds": expected_env,
            "action_noise_seeds": expected_noise,
            "all_episodes_retained": True,
        },
        "normalized_score_mean": {
            "inverse_only_anchor": evaluation["baseline_only"]["normalized_score_mean"],
            "opex_distilled_student": evaluation["adapted"]["normalized_score_mean"],
            "frozen_channel_aware_opex_teacher": teacher["arms"]["adapted"]["normalized_score_mean"],
        },
        "raw_return_mean": {
            "inverse_only_anchor": inverse_mean,
            "opex_distilled_student": student_mean,
            "frozen_channel_aware_opex_teacher": teacher_mean,
        },
        "paired_return_differences": {
            "student_minus_inverse": paired.tolist(),
            "teacher_minus_inverse": (
                np.asarray(teacher_returns) - np.asarray(inverse_returns)
            ).tolist(),
            "student_minus_teacher": (
                np.asarray(student_returns) - np.asarray(teacher_returns)
            ).tolist(),
        },
        "fraction_of_teacher_mean_return_improvement_over_inverse_recovered_by_student": recovery_fraction,
        "cost": {
            "training_teacher_q1_rows": 20_480_000,
            "student_deployment_q1_rows_per_step": 0,
            "channel_aware_opex_teacher_deployment_q1_rows_per_step": 16,
        },
        "interpretation": (
            "PA-RL-like amortization control, not primary novelty; post-training "
            "development recovery caused only by a frozen expected-hash typo."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.protocol, args.output)
    print(json.dumps(payload["normalized_score_mean"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
