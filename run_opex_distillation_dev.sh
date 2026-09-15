#!/usr/bin/env bash
set -euo pipefail

# Single-use development runner.  It deliberately cannot resume or overwrite a
# partial/completed run.  Use --preflight-only to validate every frozen input
# without creating files or starting training.
readonly mode="${1:-run}"
if [[ $# -gt 1 ]] || [[ "${mode}" != "run" && "${mode}" != "--preflight-only" ]]; then
  echo "usage: $0 [--preflight-only]" >&2
  exit 2
fi

readonly research_root="/root/hubl_research_20260914"
readonly code_dir="${research_root}/code"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly protocol="${code_dir}/opex_distillation_dev_protocol.json"
readonly protocol_sha256="42bffb0b358b5afc7491b2894d4541b01379e180ffaa339f62a6c9d545e60902"
readonly dataset="/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5"
readonly base_checkpoint="${research_root}/results/channel_dev/hubl_constant_executed_beta05_25k_seed0/latest.pt"
readonly calibration="${research_root}/results/channel_calibration/beta125_n512_seed27001_calibration.json"
readonly teacher_raw="${research_root}/results/channel_opex_dev/inverse_t2_k8_d025_eta01_seed0_dev10.json"
readonly output_dir="${research_root}/results/opex_distill_dev/inverse_teacher_eta01_k8_t2_d025_u5k_seed0"
readonly driver_log="${research_root}/results/opex_distill_dev/inverse_teacher_eta01_k8_t2_d025_u5k_seed0.driver.log"
readonly train_log="${output_dir}.train.console.log"
readonly evaluation="${output_dir}/matched_dev10.json"
readonly evaluation_log="${output_dir}/matched_dev10.console.log"
readonly comparison="${output_dir}/matched_dev10_comparison.json"

verify_sha256() {
  local expected="$1"
  local path="$2"
  [[ -f "${path}" ]] || { echo "missing frozen input: ${path}" >&2; exit 1; }
  local actual
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  }
}

[[ -x "${python_bin}" ]] || { echo "missing Python environment: ${python_bin}" >&2; exit 1; }
verify_sha256 "${protocol_sha256}" "${protocol}"
verify_sha256 "159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a" "${dataset}"
verify_sha256 "d29501edf9b6961225bcbc157d7d7034bc5b625856dedcbbf9421eea25898106" "${base_checkpoint}"
verify_sha256 "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354" "${calibration}"
verify_sha256 "700276fb6d171976443d111106e9d54926074f58ee37bd4f2648df0db41e299f" "${teacher_raw}"
verify_sha256 "f1be7520cc84bc7851c9812ff029e67289f912d163c2d2ee64843e17b61ac5ea" "${code_dir}/train_opex_distilled_adapter.py"
verify_sha256 "38747e3a732362734757e6bde3715eb2ac5eb9ecd0870ccead2193670da89eed" "${code_dir}/evaluate_inverse_residual_adapter.py"
verify_sha256 "cf659f47fddd8f848c84f5f114be1fcb9081a55470fe59e8c073d4f84af0bff1" "${code_dir}/evaluate_channel_opex.py"
verify_sha256 "5700efa8bbefac91c991a76878c3b78c5b3e84aea83568987f7a7e8919e50041" "${code_dir}/inverse_residual_core.py"
verify_sha256 "24318269a6696fd22c8112b2d96913bdceb830d83de9df31ca234e1505f1aec9" "${code_dir}/evaluation_controls.py"
verify_sha256 "e27cec131b533dd542e77ba9de2de492db4890c6fb9723796f59a02002d1f052" "${code_dir}/td3bc_core.py"
verify_sha256 "fb065682c78e6c058d1165fa68f680fb682d235ec8c9c3abe70b8098177ee0e4" "${code_dir}/train_inverse_residual_adapter.py"
verify_sha256 "b3a159fc8c0612c5c23578a1496a6beb858063462c086a824fe7874d09b089ee" "${code_dir}/train_iql.py"

for forbidden in \
  "${output_dir}" \
  "${output_dir}.tmp" \
  "${driver_log}" \
  "${driver_log}.tmp" \
  "${train_log}" \
  "${train_log}.tmp"; do
  [[ ! -e "${forbidden}" ]] || {
    echo "single-use runner refuses existing output: ${forbidden}" >&2
    exit 1
  }
done

"${python_bin}" - "${protocol}" "${teacher_raw}" <<'PY'
import json
import sys
from pathlib import Path

protocol_path, teacher_path = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text("utf-8"))
assert protocol["schema_version"] == "opex-distillation-development-protocol-v1"
assert protocol["status"] == "frozen_before_execution"
assert protocol["evidence_label"] == "development"
assert protocol["confirmation_evidence_eligible"] is False
assert protocol["novelty_position"]["is_primary_novelty"] is False
assert protocol["training"]["train_seed"] == 0
assert protocol["training"]["updates"] == 5000
assert protocol["training"]["batch_size"] == 256
assert protocol["training"]["delta_max"] == 0.25
assert protocol["training"]["observation_split"] == {
    "audit_fraction": 0.1,
    "split_seed": 424242,
    "expected_train_observation_count": 899940,
    "expected_audit_observation_count": 100060,
    "expected_train_indices_sha256": "f90181e680de975cf43be71ff477462513b6e9157453abb96302f9bf46adece5d",
    "expected_audit_indices_sha256": "e122718e0aa418e07284c04363d65494ce211b3c0d0878a0d3af54f6c3b4019b3",
}
teacher_cfg = protocol["training"]["teacher"]
assert teacher_cfg["baseline_transform"] == "inverse"
assert teacher_cfg["step_size"] == 0.1
assert teacher_cfg["gradient_noise_samples"] == 8
assert teacher_cfg["gradient_steps"] == 2
assert teacher_cfg["gradient_noise_seed"] == 271828
evaluation = protocol["evaluation"]
assert evaluation["environment"] == "Walker2d-v4"
assert evaluation["rollout_action_noise_beta"] == 1.25
assert evaluation["episode_count_per_arm"] == 10
assert evaluation["environment_seed_start"] == 28300
assert evaluation["action_noise_seed_start"] == 38300

teacher = json.loads(teacher_path.read_text("utf-8"))
assert teacher["status"] == "complete"
assert teacher["raw_schema"] == "channel_opex_v1"
assert teacher["method_id"] == "channel_aware_opex_inverse_anchor"
assert teacher["controller"]["baseline_transform"] == "inverse"
assert teacher["controller"]["step_size"] == 0.1
assert teacher["controller"]["K"] == 8
assert teacher["controller"]["gradient_steps"] == 2
assert teacher["controller"]["delta_max"] == 0.25
assert teacher["controller"]["model_beta"] == 1.2498948872089386
assert teacher["evaluation_protocol"]["episode_count_per_arm"] == 10
assert teacher["arms"]["baseline_only"]["environment_seeds"] == list(range(28300, 28310))
assert teacher["arms"]["adapted"]["environment_seeds"] == list(range(28300, 28310))
assert teacher["arms"]["baseline_only"]["action_noise_seeds"] == list(range(38300, 38310))
assert teacher["arms"]["adapted"]["action_noise_seeds"] == list(range(38300, 38310))
assert teacher["implementation"]["evaluate_sha256"] == "cf659f47fddd8f848c84f5f114be1fcb9081a55470fe59e8c073d4f84af0bff1"
assert teacher["base_checkpoint"]["sha256"] == "d29501edf9b6961225bcbc157d7d7034bc5b625856dedcbbf9421eea25898106"
assert teacher["calibration"]["calibration_sha256"] == "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354"
print("frozen development protocol and teacher record validated")
PY

if [[ "${mode}" == "--preflight-only" ]]; then
  echo "preflight complete; no files created and no training launched"
  exit 0
fi

mkdir -p "$(dirname "${output_dir}")"
# Make every shell redirection fail if a competing/stale artifact appeared
# after preflight.  This keeps logs under the same no-overwrite rule as JSON.
set -o noclobber
exec > "${driver_log}" 2>&1
cd "${code_dir}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start immutable OPEX distillation development run"

"${python_bin}" train_opex_distilled_adapter.py \
  --dataset "${dataset}" \
  --base-checkpoint "${base_checkpoint}" \
  --channel-calibration "${calibration}" \
  --output-dir "${output_dir}" \
  --device cuda \
  --train-seed 0 \
  --updates 5000 \
  --batch-size 256 \
  --hidden-dim 128 \
  --depth 2 \
  --learning-rate 0.0003 \
  --delta-max 0.25 \
  --teacher-step-size 0.1 \
  --teacher-k 8 \
  --teacher-gradient-steps 2 \
  --teacher-noise-seed 271828 \
  --audit-fraction 0.1 \
  --split-seed 424242 \
  --precompute-batch-size 8192 \
  --log-period 100 \
  --checkpoint-period 1000 \
  --torch-threads 2 \
  > "${train_log}" 2>&1

cp --no-clobber -- "${protocol}" "${output_dir}/development_protocol.json"
verify_sha256 "${protocol_sha256}" "${output_dir}/development_protocol.json"

"${python_bin}" - "${output_dir}/config.json" "${output_dir}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

config_path, summary_path = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text("utf-8"))
summary = json.loads(summary_path.read_text("utf-8"))
assert config["method"] == "calibrated_channel_opex_teacher_distillation"
assert config["development_only"] is True
assert config["arguments"]["train_seed"] == 0
assert config["arguments"]["updates"] == 5000
assert config["arguments"]["batch_size"] == 256
assert config["arguments"]["teacher_step_size"] == 0.1
assert config["arguments"]["teacher_k"] == 8
assert config["arguments"]["teacher_gradient_steps"] == 2
assert config["arguments"]["teacher_noise_seed"] == 271828
assert config["arguments"]["delta_max"] == 0.25
assert config["observation_split"]["train_observation_count"] == 899940
assert config["observation_split"]["audit_observation_count"] == 100060
assert config["observation_split"]["train_indices_sha256"] == "f90181e680de975cf43be71ff477462513b6e9157453abb96302f9bf46adece5d"
assert config["observation_split"]["audit_indices_sha256"] == "e122718e0aa418e07284c04363d65494ce211b3c0d0878a0d3af54f6c3b4019b3"
assert summary["status"] == "complete"
assert summary["development_only"] is True
assert summary["updates"] == 5000
assert summary["teacher_accounting"]["q1_input_rows"] == 20480000
assert summary["base_parameters_unchanged"] is True
assert summary["deployment_q1_rows_per_action"] == 0
print("completed training artifacts validated")
PY

"${python_bin}" evaluate_inverse_residual_adapter.py \
  --adapter-checkpoint "${output_dir}/latest.pt" \
  --base-checkpoint "${base_checkpoint}" \
  --output "${evaluation}" \
  --env-name Walker2d-v4 \
  --device cuda \
  --eval-episodes 10 \
  --eval-seed 28300 \
  --eval-noise-seed 38300 \
  --eval-action-noise-beta 1.25 \
  --reference-min-score 1.629008 \
  --reference-max-score 4592.3 \
  > "${evaluation_log}" 2>&1

"${python_bin}" - "${protocol}" "${output_dir}/summary.json" "${evaluation}" "${teacher_raw}" "${comparison}" <<'PY'
import hashlib
import json
import math
import os
import sys
from pathlib import Path

protocol_path, summary_path, student_path, teacher_path, output_path = map(Path, sys.argv[1:])
if output_path.exists() or output_path.with_suffix(output_path.suffix + ".tmp").exists():
    raise FileExistsError(f"refusing to overwrite {output_path}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

summary = json.loads(summary_path.read_text("utf-8"))
student = json.loads(student_path.read_text("utf-8"))
teacher = json.loads(teacher_path.read_text("utf-8"))
expected_env = list(range(28300, 28310))
expected_noise = list(range(38300, 38310))
assert student["status"] == "complete"
assert student["environment"] == "Walker2d-v4"
assert student["channel"]["environment_rollout_beta"] == 1.25
assert student["evaluation_protocol"]["episode_count_per_method"] == 10
for arm_name in ("baseline_only", "adapted"):
    assert student[arm_name]["environment_seeds"] == expected_env
    assert student[arm_name]["action_noise_seeds"] == expected_noise
for arm_name in ("baseline_only", "adapted"):
    assert teacher["arms"][arm_name]["environment_seeds"] == expected_env
    assert teacher["arms"][arm_name]["action_noise_seeds"] == expected_noise

inverse_returns = student["baseline_only"]["returns"]
teacher_inverse_returns = teacher["arms"]["baseline_only"]["returns"]
if len(inverse_returns) != 10 or any(
    not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
    for left, right in zip(inverse_returns, teacher_inverse_returns)
):
    raise RuntimeError("inverse-only anchor did not reproduce the frozen teacher baseline")
student_returns = student["adapted"]["returns"]
teacher_returns = teacher["arms"]["adapted"]["returns"]
inverse_mean = sum(inverse_returns) / len(inverse_returns)
student_mean = sum(student_returns) / len(student_returns)
teacher_mean = sum(teacher_returns) / len(teacher_returns)
teacher_gain = teacher_mean - inverse_mean
recovery = (student_mean - inverse_mean) / teacher_gain if teacher_gain != 0.0 else None

payload = {
    "schema_version": "opex-distillation-development-comparison-v1",
    "status": "complete",
    "evidence_label": "development",
    "confirmation_evidence_eligible": False,
    "novelty_position": "PA-RL-like amortization control; not the primary novelty",
    "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
    "training_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
    "student_evaluation": {"path": str(student_path), "sha256": sha256(student_path)},
    "frozen_teacher_evaluation": {"path": str(teacher_path), "sha256": sha256(teacher_path)},
    "evaluation_protocol": {
        "environment": "Walker2d-v4",
        "rollout_beta": 1.25,
        "environment_seeds": expected_env,
        "action_noise_seeds": expected_noise,
        "all_episodes_retained": True,
    },
    "normalized_score_mean": {
        "inverse_only_anchor": student["baseline_only"]["normalized_score_mean"],
        "opex_distilled_student": student["adapted"]["normalized_score_mean"],
        "frozen_channel_aware_opex_teacher": teacher["arms"]["adapted"]["normalized_score_mean"],
    },
    "raw_return_mean": {
        "inverse_only_anchor": inverse_mean,
        "opex_distilled_student": student_mean,
        "frozen_channel_aware_opex_teacher": teacher_mean,
    },
    "paired_return_differences": {
        "student_minus_inverse": [s - b for s, b in zip(student_returns, inverse_returns)],
        "teacher_minus_inverse": [t - b for t, b in zip(teacher_returns, inverse_returns)],
        "student_minus_teacher": [s - t for s, t in zip(student_returns, teacher_returns)],
    },
    "fraction_of_teacher_mean_return_improvement_over_inverse_recovered_by_student": recovery,
    "cost": {
        "training_teacher_q1_rows": summary["teacher_accounting"]["q1_input_rows"],
        "inverse_only_deployment_q1_rows_per_step": 0,
        "distilled_student_deployment_q1_rows_per_step": 0,
        "channel_aware_opex_teacher_deployment_q1_rows_per_step": 16,
    },
    "selection_rule": "All ten predeclared development episodes retained; no run filtering.",
}
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
print(json.dumps(payload["normalized_score_mean"], sort_keys=True))
PY

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) immutable development run complete"
