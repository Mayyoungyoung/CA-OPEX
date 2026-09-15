#!/usr/bin/env bash
set -euo pipefail

# One-shot beta-severity grid driver for the frozen CA-OPEX controller.
# Evaluates beta in {0.5,0.9,1.1,1.4} on pre-existing Walker2d base-policy
# checkpoints.  Per beta it generates its own 512-pair calibration (distinct
# pairs/bootstrap seeds) and evaluates four controllers on one shared 50-case
# block: complete CA-OPEX, equal-neural-budget nominal-Q (K=8,T=2), calibrated
# inverse-only, and the ordinary identity command.  K/T/eta/delta are frozen;
# nothing is retuned per beta.  No resume mode: one new directory per unit.

readonly research_root="/root/hubl_research_20260914"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly code_dir="${research_root}/code"
readonly dataset="/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5"
readonly dataset_sha256="159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a"
readonly results_root="${CA_OPEX_BETAGRID_RESULTS_ROOT:-${research_root}/results/betagrid}"

readonly environment="Walker2d-v4"
readonly eval_episodes=50
readonly reference_min="1.629008"
readonly reference_max="4592.3"

seed=""
beta=""
dry_run=0

usage() {
  cat >&2 <<'EOF'
usage: run_ca_opex_walker_betagrid.sh --seed N --beta {0.5|0.9|1.1|1.4} [--dry-run]
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value"
      [[ -z "${seed}" ]] || die "--seed was specified more than once"
      seed="$2"
      shift 2
      ;;
    --beta)
      [[ $# -ge 2 ]] || die "--beta requires a value"
      [[ -z "${beta}" ]] || die "--beta was specified more than once"
      beta="$2"
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

[[ "${seed}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ -n "${beta}" ]] || die "--beta is required"

# Frozen unit table: beta -> "pairs_seed bootstrap_seed env_start noise_start grad_start"
case "${beta}" in
  0.5)  unit="27011 28011 51400 61400 71400" ;;
  0.9)  unit="27012 28012 51500 61500 71500" ;;
  1.1)  unit="27013 28013 51600 61600 71600" ;;
  1.4)  unit="27014 28014 51700 61700 71700" ;;
  *) die "beta ${beta} is not in the frozen grid {0.5,0.9,1.1,1.4}" ;;
esac
read -r pairs_seed bootstrap_seed env_seed noise_seed grad_seed <<<"${unit}"

case "${seed}" in
  1) checkpoint="${research_root}/results/inverse_residual_confirm/base_hubl_executed_25k_seed1/latest.pt"
     checkpoint_sha256="22b47965695bfa312533741abaa1e17e3cbbc091f3c5e6ba7fc8953fbc8ca6ae" ;;
  2) checkpoint="${research_root}/results/scaleup/walker_seed2/base/latest.pt"
     checkpoint_sha256="d94b05044c9071c2a987aa28bd2dabbf04a91a7164d4215f851f37aaff06a75e" ;;
  10) checkpoint="${research_root}/results/inverse_residual_confirm/base_hubl_executed_25k_seed10/latest.pt"
     checkpoint_sha256="eba1af040aa73823ceba28048e10d3af2eef54c59f09c1e0194f27aa0ed4944d" ;;
  *) die "training seed ${seed} is not in the predeclared checkpoint set {1,2,10}" ;;
esac

readonly pairs_npz="${results_root}/calibration/beta${beta}_n512_seed${pairs_seed}.npz"
readonly calibration="${results_root}/calibration/beta${beta}_n512_seed${pairs_seed}_calibration.json"
readonly run_root="${results_root}/beta${beta}_seed${seed}"
readonly controls_dir="${run_root}/controls"
[[ "${results_root}" == /* && "${results_root}" != "/" ]] || \
  die "results root must be an absolute non-root path"
[[ ! -e "${run_root}" ]] || die "refusing to reuse or overwrite existing unit root: ${run_root}"

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

ranges_overlap() {
  local candidate_start="$1"
  local reserved_start="$2"
  local candidate_end=$((candidate_start + eval_episodes - 1))
  local reserved_end=$((reserved_start + eval_episodes - 1))
  (( candidate_start <= reserved_end && reserved_start <= candidate_end ))
}

reject_all_reserved() {
  local namespace="$1"
  local candidate="$2"
  shift 2
  local reserved
  for reserved in "$@"; do
    if [[ "${candidate}" -eq "${reserved}" ]]; then
      continue
    fi
    if ranges_overlap "${candidate}" "${reserved}"; then
      die "${namespace} block ${candidate}..$((candidate + 49)) overlaps reserved block ${reserved}..$((reserved + 49))"
    fi
  done
}

command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
require_file "${python_bin}"
require_file "${dataset}"
require_file "${checkpoint}"
require_file "${code_dir}/generate_channel_pairs.py"
require_file "${code_dir}/calibrate_uniform_channel.py"
require_file "${code_dir}/evaluate_channel_opex.py"
require_file "${code_dir}/evaluate_td3bc.py"
require_sha256 "${dataset}" "${dataset_sha256}"
require_sha256 "${checkpoint}" "${checkpoint_sha256}"

# All previously used environment/noise/gradient blocks stay reserved, plus
# every beta-grid row (including this unit's own row, which is skipped by the
# self-equality rule), so one grid entry can never silently reuse another's
# rollout cases.
reject_all_reserved environment "${env_seed}" 39300 79300 131300 131400 131500 41300 41400 41500 51400 51500 51600 51700
reject_all_reserved actuator "${noise_seed}" 49300 89300 231300 231400 231500 51300 51400 51500 61400 61500 61600 61700
reject_all_reserved gradient "${grad_seed}" 69300 99300 331300 331400 331500 61300 61400 61500 71400 71500 71600 71700

progress() {
  local stage="$1"
  local status="$2"
  local mode="execution"
  if [[ "${dry_run}" -eq 1 ]]; then
    mode="dry_run_render_only"
  fi
  printf '%s mode=%s seed=%s beta=%s stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${mode}" "${seed}" "${beta}" "${stage}" "${status}"
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
  echo "Static dry run only; no files, directories, logs, or results will be created."
else
  mkdir -p "${results_root}/calibration"
  mkdir "${run_root}"
  mkdir "${controls_dir}"
fi

progress calibration_pairs started
if [[ -f "${pairs_npz}" ]]; then
  if [[ "${dry_run}" -eq 0 ]]; then
    progress calibration_pairs existing_verified
  fi
else
  run_logged "${run_root}/calibration_pairs.console.log" \
    "${python_bin}" "${code_dir}/generate_channel_pairs.py" \
      --source "${dataset}" \
      --output "${pairs_npz}" \
      --pairs 512 \
      --beta "${beta}" \
      --seed "${pairs_seed}"
fi
progress calibration_pairs completed

progress calibration_fit started
if [[ -f "${calibration}" ]]; then
  if [[ "${dry_run}" -eq 0 ]]; then
    progress calibration_fit existing_verified
  fi
else
  run_logged "${run_root}/calibration_fit.console.log" \
    "${python_bin}" "${code_dir}/calibrate_uniform_channel.py" \
      --pairs "${pairs_npz}" \
      --output "${calibration}" \
      --bootstrap-replicates 1000 \
      --bootstrap-seed "${bootstrap_seed}"
fi
progress calibration_fit completed

run_channel_arm() {
  local arm_id="$1"
  local baseline_transform="$2"
  local model_mode="$3"
  local step_size="$4"
  local delta_max="$5"
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
      --base-checkpoint "${checkpoint}" \
      "${model_args[@]}" \
      --output "${output_path}" \
      --env-name "${environment}" \
      --device cuda \
      --step-size "${step_size}" \
      --baseline-transform "${baseline_transform}" \
      --gradient-noise-samples 8 \
      --gradient-steps 2 \
      --gradient-noise-seed "${grad_seed}" \
      --delta-max "${delta_max}" \
      --rollout-action-noise-beta "${beta}" \
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

run_command_arm() {
  local arm_id="$1"
  local transform="$2"
  local output_path="${controls_dir}/${arm_id}.json"
  local log_path="${controls_dir}/${arm_id}.console.log"
  progress "${arm_id}" started
  run_logged "${log_path}" \
    "${python_bin}" "${code_dir}/evaluate_td3bc.py" \
      --checkpoint "${checkpoint}" \
      --output "${output_path}" \
      --env-name "${environment}" \
      --device cuda \
      --eval-episodes "${eval_episodes}" \
      --eval-seed "${env_seed}" \
      --eval-noise-seed "${noise_seed}" \
      --eval-action-noise-beta "${beta}" \
      --command-scale 1.0 \
      --command-transform "${transform}" \
      --reference-min-score "${reference_min}" \
      --reference-max-score "${reference_max}"
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output_path}"
    require_file "${log_path}"
  fi
  progress "${arm_id}" completed
}

# Frozen configurations carried over unchanged: eta/K/T/delta are never reselected.
run_channel_arm complete inverse calibrated 0.1 0.25
run_channel_arm nominal_k8t2 identity nominal_beta0 0.1 2.0
run_command_arm inverse_only uniform_mean_inverse
run_command_arm identity_command identity

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Static dry run complete."
else
  echo "Beta-grid unit beta=${beta} seed=${seed} completed in ${run_root}."
fi
