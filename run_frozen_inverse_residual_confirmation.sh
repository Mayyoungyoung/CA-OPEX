#!/usr/bin/env bash
set -euo pipefail

# Frozen Walker2d-v4 confirmation block for the inverse-residual adapter and
# its four predeclared mechanism/resource controls.  A real invocation is
# intentionally fail-closed: no declared run directory, JSON, temporary JSON,
# or console log may already exist.  Use --dry-run only to inspect commands.

readonly research_root="/root/hubl_research_20260914"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly code_dir="${research_root}/code"
readonly result_root="${research_root}/results/inverse_residual_confirm"
readonly dataset="/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5"
readonly calibration="${research_root}/results/channel_calibration/beta125_n512_seed27001_calibration.json"
readonly manifest="${code_dir}/inverse_residual_frozen_confirmation_manifest.json"

readonly updates=5000
readonly batch_size=256
readonly hidden_dim=128
readonly depth=2
readonly execution_noise_samples=8
readonly audit_fraction=0.1
readonly split_seed=424242
readonly scale_calibration_observations=4096
readonly eval_episodes=50
readonly eval_seed=39300
readonly eval_noise_seed=49300
readonly audit_noise_samples=64
readonly audit_noise_seed=987654
readonly physical_beta=1.25

dry_run=0
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

progress() {
  local run_id="$1"
  local stage="$2"
  local status="$3"
  printf '%s run_id=%s stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${run_id}" "${stage}" "${status}"
}
if [[ $# -eq 1 ]]; then
  if [[ "$1" != "--dry-run" ]]; then
    echo "usage: $0 [--dry-run]" >&2
    exit 2
  fi
  dry_run=1
fi

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file is missing: ${path}"
}

require_absent() {
  local path
  for path in "$@"; do
    [[ ! -e "${path}" ]] || die "refusing to overwrite existing output: ${path}"
  done
}

print_command() {
  local log_path="$1"
  shift
  printf 'DRY-RUN: '
  printf '%q ' "$@"
  printf '> %q 2>&1\n' "${log_path}"
}

run_logged() {
  local log_path="$1"
  shift
  if [[ "${dry_run}" -eq 1 ]]; then
    print_command "${log_path}" "$@"
    return 0
  fi
  require_absent "${log_path}"
  "$@" > "${log_path}" 2>&1
}

base_checkpoint_for_seed() {
  local seed="$1"
  printf '%s\n' "${result_root}/base_hubl_executed_25k_seed${seed}/latest.pt"
}

training_dir_for() {
  local variant="$1"
  local seed="$2"
  printf '%s\n' "${result_root}/${variant}_seed${seed}"
}

verify_manifest_contract() {
  "${python_bin}" - "${manifest}" "${code_dir}" "${dataset}" "${calibration}" "${result_root}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
code_dir = Path(sys.argv[2]).resolve()
dataset_path = Path(sys.argv[3]).resolve()
calibration_path = Path(sys.argv[4]).resolve()
result_root = Path(sys.argv[5]).resolve()
sys.path.insert(0, str(code_dir))

from aggregate_inverse_residual_results import (  # noqa: E402
    IMPLEMENTATION_SOURCE_FILES,
    _recompute_observation_split_fingerprint,
    sha256_file,
    validate_strict_manifest_definition,
)

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
validate_strict_manifest_definition(payload, allow_placeholders=False)
if payload.get("schema_version") != "inverse-residual-manifest-v2":
    raise SystemExit("unexpected frozen manifest schema_version")
if payload.get("results_root") != "../results":
    raise SystemExit("frozen manifest results_root must be ../results")

implementation = payload["frozen_implementation"]
for field, filename in IMPLEMENTATION_SOURCE_FILES.items():
    actual = sha256_file(code_dir / filename)
    if implementation.get(field) != actual:
        raise SystemExit(
            f"frozen implementation hash mismatch for {filename}: "
            f"expected={implementation.get(field)!r}, actual={actual!r}"
        )

variants = {
    "inverse_p001_k8_u5k": "inverse_sampled_main",
    "identity_p001_k8_u5k": "direct_sampled_main",
    "inverse_qmean_p001_k8_u5k": "inverse_q1_at_channel_mean",
    "identity_wide_p064_d2_k8_u5k": "direct_sampled_wide",
    "identity_nominal_beta0_p064_d2_k8_u5k": "direct_sampled_nominal_beta0",
}
expected = {}
split = _recompute_observation_split_fingerprint(
    dataset_path,
    max_observations=None,
    audit_fraction=0.1,
    split_seed=424242,
)
noisy_protocol = {
    "environment": "Walker2d-v4",
    "episode_count": 50,
    "rollout_beta": 1.25,
    "environment_seed_start": 39300,
    "action_noise_seed_start": 49300,
}
clean_protocol = {**noisy_protocol, "rollout_beta": 0.0}
audit_protocol = {
    "noise_samples": 64,
    "noise_seed": 987654,
    "action_noise_beta": 1.25,
    "selected_observations": split["audit_observation_count"],
    "train_indices_sha256": split["train_indices_sha256"],
    "audit_indices_sha256": split["audit_indices_sha256"],
    "audit_episode_units_sha256": split["audit_episode_units_sha256"],
}
for prefix, method_id in variants.items():
    for seed in (1, 10):
        run_id = f"{prefix}_seed{seed}"
        evaluations = [
            {
                "evaluation_id": "confirm_beta125",
                "path": f"inverse_residual_confirm/{run_id}/confirm_beta125_50.json",
                "evidence_label": "confirmation",
                "expected_protocol": noisy_protocol,
            }
        ]
        if prefix in {"inverse_p001_k8_u5k", "identity_p001_k8_u5k"}:
            evaluations.append(
                {
                    "evaluation_id": "confirm_clean",
                    "path": f"inverse_residual_confirm/{run_id}/confirm_clean_50.json",
                    "evidence_label": "confirmation",
                    "expected_protocol": clean_protocol,
                }
            )
        expected[run_id] = {
            "method_id": method_id,
            "training_dir": f"inverse_residual_confirm/{run_id}",
            "evaluations": evaluations,
            "audits": [
                {
                    "audit_id": "heldout_k64",
                    "path": f"inverse_residual_confirm/{run_id}/heldout_audit_k64.json",
                    "evidence_label": "confirmation",
                    "expected_protocol": audit_protocol,
                }
            ],
        }

runs = payload.get("runs")
if not isinstance(runs, list):
    raise SystemExit("frozen manifest runs must be a list")
actual_ids = [run.get("run_id") for run in runs if isinstance(run, dict)]
if len(actual_ids) != len(runs) or len(actual_ids) != len(set(actual_ids)):
    raise SystemExit("frozen manifest has invalid or duplicate run_id values")
if set(actual_ids) != set(expected):
    raise SystemExit(
        f"frozen manifest run_id mismatch: expected={sorted(expected)}, "
        f"actual={sorted(actual_ids)}"
    )

for run in runs:
    run_id = run["run_id"]
    contract = expected[run_id]
    for field in ("method_id", "training_dir"):
        if run.get(field) != contract[field]:
            raise SystemExit(
                f"frozen manifest {run_id} {field} mismatch: "
                f"expected={contract[field]!r}, actual={run.get(field)!r}"
            )
    actual_evaluations = [
        {
            key: entry.get(key)
            for key in ("evaluation_id", "path", "evidence_label", "expected_protocol")
        }
        for entry in run.get("evaluations", [])
    ]
    if actual_evaluations != contract["evaluations"]:
        raise SystemExit(
            f"frozen manifest {run_id} evaluations mismatch: "
            f"expected={contract['evaluations']!r}, actual={actual_evaluations!r}"
        )
    actual_audits = [
        {
            key: entry.get(key)
            for key in ("audit_id", "path", "evidence_label", "expected_protocol")
        }
        for entry in run.get("audits", [])
    ]
    if actual_audits != contract["audits"]:
        raise SystemExit(
            f"frozen manifest {run_id} audits mismatch: "
            f"expected={contract['audits']!r}, actual={actual_audits!r}"
        )

dataset_sha = sha256_file(dataset_path)
calibration_sha = sha256_file(calibration_path)
base_hashes = {
    seed: sha256_file(result_root / f"base_hubl_executed_25k_seed{seed}" / "latest.pt")
    for seed in (1, 10)
}
calibrated_beta = 1.2498948872089386
for run in runs:
    run_id = run["run_id"]
    seed = 10 if run_id.endswith("_seed10") else 1
    artifacts = run["expected_artifacts"]
    is_nominal = run_id.startswith("identity_nominal_beta0_")
    expected_calibration_sha = None if is_nominal else calibration_sha
    observed_artifacts = {
        "dataset_sha256": dataset_sha,
        "base_checkpoint_sha256": base_hashes[seed],
        "calibration_sha256": expected_calibration_sha,
    }
    for field, actual in observed_artifacts.items():
        if artifacts.get(field) != actual:
            raise SystemExit(
                f"frozen manifest {run_id} artifact {field} mismatch: "
                f"expected={artifacts.get(field)!r}, actual={actual!r}"
            )

    if run_id.startswith("inverse_p001_"):
        expected_method = ("inverse", "sampled_expected_q1", 0.25, 0.01, 3e-4)
    elif run_id.startswith("identity_p001_"):
        expected_method = ("identity", "sampled_expected_q1", 0.25, 0.01, 3e-4)
    elif run_id.startswith("inverse_qmean_"):
        expected_method = ("inverse", "q1_at_channel_mean", 0.25, 0.01, 3e-4)
    else:
        expected_method = ("identity", "sampled_expected_q1", 2.0, 0.64, 3.75e-5)
    baseline_transform, value_estimator, delta_max, penalty, learning_rate = expected_method
    expected_config = {
        "baseline_transform": baseline_transform,
        "value_estimator": value_estimator,
        "alpha": 1.0,
        "residual_penalty": penalty,
        "delta_max": delta_max,
        "execution_noise_samples": 8,
        "updates": 5000,
        "learning_rate": learning_rate,
        "batch_size": 256,
        "split_seed": 424242,
        "audit_fraction": 0.1,
        "train_seed": seed,
        "channel_seed": 271828 + seed,
        "model_or_calibration_beta": 0.0 if is_nominal else calibrated_beta,
        "calibration_mode": (
            "known_beta_without_pair_calibration" if is_nominal else "pair_calibration"
        ),
        "hidden_dim": 128,
        "depth": 2,
        "q_scale_epsilon": 1e-6,
        "precompute_batch_size": 8192,
        "scale_calibration_observations": 4096,
        "max_observations": None,
        "base_action_pairing": "executed_executed",
        "extensions": {},
    }
    if run.get("expected_config") != expected_config:
        raise SystemExit(
            f"frozen manifest {run_id} expected_config differs from runner contract"
        )
PY
}

preflight_outputs() {
  local seed variant run_dir
  local variants=(
    inverse_p001_k8_u5k
    identity_p001_k8_u5k
    inverse_qmean_p001_k8_u5k
    identity_wide_p064_d2_k8_u5k
    identity_nominal_beta0_p064_d2_k8_u5k
  )
  for seed in 1 10; do
    for variant in "${variants[@]}"; do
      run_dir="$(training_dir_for "${variant}" "${seed}")"
      require_absent "${run_dir}" "${run_dir}.train.console.log"
    done
  done
}

run_evaluation() {
  local run_dir="$1"
  local base_checkpoint="$2"
  local evaluation_id="$3"
  local rollout_beta="$4"
  local output_path="${run_dir}/${evaluation_id}_50.json"
  local log_path="${run_dir}/${evaluation_id}_50.console.log"
  local run_id="${run_dir##*/}"

  if [[ "${dry_run}" -eq 0 ]]; then
    require_absent "${output_path}" "${output_path}.tmp" "${log_path}"
  fi
  progress "${run_id}" "evaluation_${evaluation_id}" started
  run_logged "${log_path}" \
    "${python_bin}" evaluate_inverse_residual_adapter.py \
      --adapter-checkpoint "${run_dir}/latest.pt" \
      --base-checkpoint "${base_checkpoint}" \
      --output "${output_path}" \
      --env-name Walker2d-v4 \
      --device cuda \
      --eval-episodes "${eval_episodes}" \
      --eval-seed "${eval_seed}" \
      --eval-noise-seed "${eval_noise_seed}" \
      --eval-action-noise-beta "${rollout_beta}" \
      --reference-min-score 1.629008 \
      --reference-max-score 4592.3
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output_path}"
  fi
  progress "${run_id}" "evaluation_${evaluation_id}" completed
}

run_audit() {
  local run_dir="$1"
  local base_checkpoint="$2"
  local output_path="${run_dir}/heldout_audit_k64.json"
  local log_path="${run_dir}/heldout_audit_k64.console.log"
  local run_id="${run_dir##*/}"

  if [[ "${dry_run}" -eq 0 ]]; then
    require_absent "${output_path}" "${output_path}.tmp" "${log_path}"
  fi
  progress "${run_id}" heldout_audit_k64 started
  run_logged "${log_path}" \
    "${python_bin}" audit_inverse_residual_adapter.py \
      --adapter-checkpoint "${run_dir}/latest.pt" \
      --base-checkpoint "${base_checkpoint}" \
      --dataset "${dataset}" \
      --output "${output_path}" \
      --device cuda \
      --audit-noise-samples "${audit_noise_samples}" \
      --audit-noise-seed "${audit_noise_seed}" \
      --audit-action-noise-beta "${physical_beta}" \
      --batch-size "${batch_size}"
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output_path}"
  fi
  progress "${run_id}" heldout_audit_k64 completed
}

run_variant() {
  local seed="$1"
  local variant="$2"
  local baseline_transform="$3"
  local value_estimator="$4"
  local delta_max="$5"
  local residual_penalty="$6"
  local learning_rate="$7"
  local channel_mode="$8"
  local run_clean_evaluation="$9"
  local run_dir base_checkpoint train_log execution_noise_seed
  local channel_args=()

  run_dir="$(training_dir_for "${variant}" "${seed}")"
  base_checkpoint="$(base_checkpoint_for_seed "${seed}")"
  train_log="${run_dir}.train.console.log"
  execution_noise_seed="$((271828 + seed))"

  if [[ "${channel_mode}" == "calibrated" ]]; then
    channel_args=(--channel-calibration "${calibration}")
  elif [[ "${channel_mode}" == "nominal_beta0" ]]; then
    channel_args=(--execution-noise-beta 0.0)
  else
    die "unknown channel mode: ${channel_mode}"
  fi

  progress "${variant}_seed${seed}" training started
  run_logged "${train_log}" \
    "${python_bin}" train_inverse_residual_adapter.py \
      --dataset "${dataset}" \
      --base-checkpoint "${base_checkpoint}" \
      --output-dir "${run_dir}" \
      --device cuda \
      --train-seed "${seed}" \
      --updates "${updates}" \
      --batch-size "${batch_size}" \
      --hidden-dim "${hidden_dim}" \
      --depth "${depth}" \
      --learning-rate "${learning_rate}" \
      --alpha 1.0 \
      --residual-penalty "${residual_penalty}" \
      --delta-max "${delta_max}" \
      --baseline-transform "${baseline_transform}" \
      --value-estimator "${value_estimator}" \
      --q-scale-epsilon 1e-6 \
      "${channel_args[@]}" \
      --execution-noise-samples "${execution_noise_samples}" \
      --execution-noise-seed "${execution_noise_seed}" \
      --audit-fraction "${audit_fraction}" \
      --split-seed "${split_seed}" \
      --precompute-batch-size 8192 \
      --scale-calibration-observations "${scale_calibration_observations}" \
      --log-period 100 \
      --checkpoint-period 1000 \
      --torch-threads 2

  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${run_dir}/latest.pt"
    require_file "${run_dir}/config.json"
    require_file "${run_dir}/progress.jsonl"
    require_file "${run_dir}/summary.json"
  fi
  progress "${variant}_seed${seed}" training completed

  run_evaluation "${run_dir}" "${base_checkpoint}" confirm_beta125 "${physical_beta}"
  if [[ "${run_clean_evaluation}" == "yes" ]]; then
    run_evaluation "${run_dir}" "${base_checkpoint}" confirm_clean 0.0
  fi
  run_audit "${run_dir}" "${base_checkpoint}"
}

if [[ "${dry_run}" -eq 0 ]]; then
  require_file "${python_bin}"
  require_file "${dataset}"
  require_file "${calibration}"
  require_file "${manifest}"
  require_file "${code_dir}/train_inverse_residual_adapter.py"
  require_file "${code_dir}/evaluate_inverse_residual_adapter.py"
  require_file "${code_dir}/audit_inverse_residual_adapter.py"
  for seed in 1 10; do
    require_file "$(base_checkpoint_for_seed "${seed}")"
  done
  verify_manifest_contract
  preflight_outputs
  mkdir -p "${result_root}"
  cd "${code_dir}"
else
  echo "Static dry run only; no files, directories, models, or results will be created."
fi

progress driver confirmation_block started

for seed in 1 10; do
  run_variant "${seed}" inverse_p001_k8_u5k \
    inverse sampled_expected_q1 0.25 0.01 3e-4 calibrated yes
  run_variant "${seed}" identity_p001_k8_u5k \
    identity sampled_expected_q1 0.25 0.01 3e-4 calibrated yes
  run_variant "${seed}" inverse_qmean_p001_k8_u5k \
    inverse q1_at_channel_mean 0.25 0.01 3e-4 calibrated no
  run_variant "${seed}" identity_wide_p064_d2_k8_u5k \
    identity sampled_expected_q1 2.0 0.64 3.75e-5 calibrated no
  run_variant "${seed}" identity_nominal_beta0_p064_d2_k8_u5k \
    identity sampled_expected_q1 2.0 0.64 3.75e-5 nominal_beta0 no
done

if [[ "${dry_run}" -eq 1 ]]; then
  progress driver confirmation_block completed
  echo "Static dry run complete."
else
  progress driver confirmation_block completed
  echo "Frozen inverse-residual confirmation block completed in ${result_root}."
fi
