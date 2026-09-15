#!/usr/bin/env bash
set -euo pipefail

# One-shot Hopper-v4 cross-task replication driver for the frozen CA-OPEX
# controller. Controller hyperparameters, base-learner recipe, and protocol
# are carried over from Walker2d without task-specific tuning. This script
# intentionally has no resume mode. A real run owns one completely new
# hopper_seed<N> directory; any pre-existing directory is a hard failure.

readonly research_root="/root/hubl_research_20260914"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly code_dir="${research_root}/code"
readonly dataset="/root/hubl_backup_data/action_noise_dev/hopper_v4_beta1_seed2201.hdf5"
readonly dataset_sha256="19bff1e68edef8fdd3ef8b4911a74d0c898e7b81970ca73975b6a0c9f04b9ee1"
readonly calibration="${research_root}/results/channel_calibration/hopper_beta125_n512_seed27002_calibration.json"
readonly calibration_sha256="d7f273b39ae935bb31e62d12f41861ca216f89c8c5086899f355b6ebfd7941f0"
readonly results_root="${CA_OPEX_HOPPER_RESULTS_ROOT:-${research_root}/results/crosstask_hopper}"

readonly environment="Hopper-v4"
readonly rollout_beta="1.25"
readonly eval_episodes=50
readonly reference_min="-20.272305"
readonly reference_max="3234.3"

# Base learner configuration carried over verbatim from the frozen Walker2d
# recipe (no Hopper-specific tuning).
readonly base_updates=25000
readonly heuristic_discount="0.391843318939209"
readonly base_internal_eval_seed=9200
readonly base_internal_noise_seed=19200

seed=""
env_seed=""
noise_seed=""
grad_seed=""
dry_run=0

usage() {
  cat >&2 <<'EOF'
usage: run_ca_opex_hopper_crosstask.sh \
  --seed N --env-seed N --noise-seed N --grad-seed N [--dry-run]
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

set_once() {
  local name="$1"
  local current="$2"
  local value="$3"
  [[ -z "${current}" ]] || die "${name} was specified more than once"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
  printf '%d\n' "$((10#${value}))"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed|--env-seed|--noise-seed|--grad-seed)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --seed) seed="$(set_once --seed "${seed}" "$2")" ;;
        --env-seed) env_seed="$(set_once --env-seed "${env_seed}" "$2")" ;;
        --noise-seed) noise_seed="$(set_once --noise-seed "${noise_seed}" "$2")" ;;
        --grad-seed) grad_seed="$(set_once --grad-seed "${grad_seed}" "$2")" ;;
      esac
      shift 2
      ;;
    --dry-run)
      [[ "${dry_run}" -eq 0 ]] || die "--dry-run was specified more than once"
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${seed}" ]] || die "--seed is required"
[[ -n "${env_seed}" ]] || die "--env-seed is required"
[[ -n "${noise_seed}" ]] || die "--noise-seed is required"
[[ -n "${grad_seed}" ]] || die "--grad-seed is required"
case "${seed}" in
  0) die "training seed 0 is reserved for development and is not a confirmation seed" ;;
esac
[[ "${results_root}" == /* && "${results_root}" != "/" ]] || \
  die "results root must be an absolute non-root path"

ranges_overlap() {
  local candidate_start="$1"
  local reserved_start="$2"
  local candidate_end=$((candidate_start + eval_episodes - 1))
  local reserved_end=$((reserved_start + eval_episodes - 1))
  (( candidate_start <= reserved_end && reserved_start <= candidate_end ))
}

reject_reserved_block() {
  local namespace="$1"
  local candidate_start="$2"
  local formal_start="$3"
  local supplemental_start="$4"
  if ranges_overlap "${candidate_start}" "${formal_start}"; then
    die "${namespace} seed block ${candidate_start}..$((candidate_start + 49)) overlaps formal block ${formal_start}..$((formal_start + 49))"
  fi
  if ranges_overlap "${candidate_start}" "${supplemental_start}"; then
    die "${namespace} seed block ${candidate_start}..$((candidate_start + 49)) overlaps supplemental block ${supplemental_start}..$((supplemental_start + 49))"
  fi
}

# Namespace-specific comparisons only: equal numeric starts across distinct RNG
# namespaces are valid and intentionally not rejected. All Walker2d blocks
# (formal, supplemental, and scale-up) remain reserved so the Hopper namespace
# cannot silently reuse them.
reject_all_reserved() {
  local namespace="$1"
  local candidate="$2"
  shift 2
  local reserved
  for reserved in "$@"; do
    if ranges_overlap "${candidate}" "${reserved}"; then
      die "${namespace} seed block ${candidate}..$((candidate + 49)) overlaps reserved block ${reserved}..$((reserved + 49))"
    fi
  done
}
reject_all_reserved environment "${env_seed}" 39300 79300 131300 131400 131500
reject_all_reserved actuator "${noise_seed}" 49300 89300 231300 231400 231500
reject_all_reserved gradient "${grad_seed}" 69300 99300 331300 331400 331500

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required static input/code is missing: ${path}"
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || \
    die "frozen input SHA-256 mismatch for ${path}: expected=${expected}, actual=${actual}"
}

command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
require_file "${python_bin}"
require_file "${dataset}"
require_file "${calibration}"
require_file "${code_dir}/train_td3bc.py"
require_file "${code_dir}/evaluate_channel_opex.py"
require_file "${code_dir}/evaluate_td3bc.py"
require_file "${code_dir}/td3bc_core.py"
require_file "${code_dir}/evaluation_controls.py"
require_file "${code_dir}/inverse_residual_core.py"
require_file "${code_dir}/train_inverse_residual_adapter.py"
require_file "${code_dir}/aggregate_ca_opex_hopper_crosstask.py"
require_sha256 "${dataset}" "${dataset_sha256}"
require_sha256 "${calibration}" "${calibration_sha256}"

readonly run_root="${results_root}/hopper_seed${seed}"
readonly base_dir="${run_root}/base"
readonly base_checkpoint="${base_dir}/latest.pt"
readonly controls_dir="${run_root}/controls"
[[ ! -e "${run_root}" ]] || die "refusing to reuse or overwrite existing run root: ${run_root}"

progress() {
  local stage="$1"
  local status="$2"
  local mode="execution"
  if [[ "${dry_run}" -eq 1 ]]; then
    mode="dry_run_render_only"
  fi
  printf '%s mode=%s seed=%s stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${mode}" "${seed}" "${stage}" "${status}"
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
  [[ ! -e "${log_path}" ]] || die "refusing to overwrite log: ${log_path}"
  "$@" >"${log_path}" 2>&1
}

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Static dry run only; no files, directories, checkpoints, logs, or results will be created."
else
  mkdir -p "${results_root}"
  # Atomic ownership claim: this fails if another process creates the same root
  # between the read-only preflight above and this point.
  mkdir "${run_root}"
  mkdir "${controls_dir}"
fi

progress base_training started
run_logged "${run_root}/base_train.console.log" \
  "${python_bin}" "${code_dir}/train_td3bc.py" \
    --dataset "${dataset}" \
    --output-dir "${base_dir}" \
    --variant hubl_constant \
    --action-pairing executed_executed \
    --env-name "${environment}" \
    --device cuda \
    --train-seed "${seed}" \
    --noise-seed 7001 \
    --iid-noise-scale 0.0 \
    --episode-noise-scale 0.0 \
    --updates "${base_updates}" \
    --batch-size 256 \
    --hidden-dim 256 \
    --depth 2 \
    --actor-learning-rate 3e-4 \
    --critic-learning-rate 3e-4 \
    --discount 0.99 \
    --tau 0.005 \
    --policy-noise 0.2 \
    --noise-clip 0.5 \
    --policy-frequency 2 \
    --alpha 2.5 \
    --max-action 1.0 \
    --heuristic-discount "${heuristic_discount}" \
    --horizon-noise-scale 0.02 \
    --horizon-control-seed 104729 \
    --lcb-kappa 0.5 \
    --execution-noise-beta 1.0 \
    --execution-noise-samples 2 \
    --execution-noise-seed 314159 \
    --execution-noise-scheme resampled_antithetic \
    --execution-noise-twin-reduction min_of_expectations \
    --eval-period 25000 \
    --eval-episodes 10 \
    --eval-seed "${base_internal_eval_seed}" \
    --eval-noise-seed "${base_internal_noise_seed}" \
    --eval-action-noise-beta 0.5 \
    --log-period 1000 \
    --torch-threads 2 \
    --reference-min-score "${reference_min}" \
    --reference-max-score "${reference_max}"
if [[ "${dry_run}" -eq 0 ]]; then
  require_file "${base_checkpoint}"
  require_file "${base_dir}/config.json"
  require_file "${base_dir}/progress.jsonl"
  require_file "${base_dir}/summary.json"
fi
progress base_training completed

run_channel_arm() {
  local arm_id="$1"
  local baseline_transform="$2"
  local model_mode="$3"
  local step_size="$4"
  local gradient_samples="$5"
  local gradient_steps="$6"
  local delta_max="$7"
  local output_path="${controls_dir}/${arm_id}.json"
  local log_path="${controls_dir}/${arm_id}.console.log"
  local model_args=()
  if [[ "${model_mode}" == "calibrated" ]]; then
    model_args=(--channel-calibration "${calibration}")
  elif [[ "${model_mode}" == "nominal_beta0" ]]; then
    model_args=(--model-action-noise-beta 0)
  else
    die "unknown channel model mode: ${model_mode}"
  fi

  progress "${arm_id}" started
  run_logged "${log_path}" \
    "${python_bin}" "${code_dir}/evaluate_channel_opex.py" \
      --base-checkpoint "${base_checkpoint}" \
      "${model_args[@]}" \
      --output "${output_path}" \
      --env-name "${environment}" \
      --device cuda \
      --step-size "${step_size}" \
      --baseline-transform "${baseline_transform}" \
      --gradient-noise-samples "${gradient_samples}" \
      --gradient-steps "${gradient_steps}" \
      --gradient-noise-seed "${grad_seed}" \
      --delta-max "${delta_max}" \
      --rollout-action-noise-beta "${rollout_beta}" \
      --eval-episodes "${eval_episodes}" \
      --eval-seed "${env_seed}" \
      --eval-noise-seed "${noise_seed}" \
      --reference-min-score "${reference_min}" \
      --reference-max-score "${reference_max}"
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output_path}"
    require_file "${log_path}"
  fi
  progress "${arm_id}" completed
}

run_inverse_only() {
  local output_path="${controls_dir}/inverse_only.json"
  local log_path="${controls_dir}/inverse_only.console.log"
  progress inverse_only started
  run_logged "${log_path}" \
    "${python_bin}" "${code_dir}/evaluate_td3bc.py" \
      --checkpoint "${base_checkpoint}" \
      --output "${output_path}" \
      --env-name "${environment}" \
      --device cuda \
      --eval-episodes "${eval_episodes}" \
      --eval-seed "${env_seed}" \
      --eval-noise-seed "${noise_seed}" \
      --eval-action-noise-beta "${rollout_beta}" \
      --command-scale 1.0 \
      --command-transform uniform_mean_inverse \
      --reference-min-score "${reference_min}" \
      --reference-max-score "${reference_max}"
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output_path}"
    require_file "${log_path}"
  fi
  progress inverse_only completed
}

# The first three arms deliberately match K=8,T=2 (16 Q1 rows and two
# backwards per adapted environment step). Original OPEX retains K=1,T=1.
run_channel_arm complete inverse calibrated 0.1 8 2 0.25
run_channel_arm calibrated_identity identity calibrated 0.3 8 2 2.0
run_channel_arm nominal_k8t2 identity nominal_beta0 0.1 8 2 2.0
run_channel_arm original_opex identity nominal_beta0 0.1 1 1 2.0
run_inverse_only

progress aggregate started
run_logged "${run_root}/aggregate.console.log" \
  "${python_bin}" "${code_dir}/aggregate_ca_opex_hopper_crosstask.py" \
    --run-root "${run_root}" \
    --output "${run_root}/aggregate.json"
if [[ "${dry_run}" -eq 0 ]]; then
  require_file "${run_root}/aggregate.json"
  require_file "${run_root}/aggregate.console.log"
fi
progress aggregate completed

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Static dry run complete."
else
  echo "Hopper CA-OPEX cross-task replication completed in ${run_root}."
fi
