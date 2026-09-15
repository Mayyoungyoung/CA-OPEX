#!/usr/bin/env bash
set -euo pipefail

readonly research_root="/root/hubl_research_20260914"
readonly code_dir="${research_root}/code"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly output_root="${research_root}/results/channel_opex_dev"
readonly base_checkpoint="${research_root}/results/channel_dev/hubl_constant_executed_beta05_25k_seed0/latest.pt"
readonly calibration="${research_root}/results/channel_calibration/beta125_n512_seed27001_calibration.json"
readonly protocol="${code_dir}/channel_opex_dev_protocol.json"

declare -A output_names=(
  ["inverse:0.01"]="inverse_t2_k8_d025_eta001_seed0_dev10"
  ["inverse:0.03"]="inverse_t2_k8_d025_eta003_seed0_dev10"
  ["inverse:0.1"]="inverse_t2_k8_d025_eta01_seed0_dev10"
  ["identity:0.01"]="identity_t2_k8_d2_eta001_seed0_dev10"
  ["identity:0.03"]="identity_t2_k8_d2_eta003_seed0_dev10"
  ["identity:0.1"]="identity_t2_k8_d2_eta01_seed0_dev10"
)

for required in "${python_bin}" "${base_checkpoint}" "${calibration}" "${protocol}"; do
  [[ -f "${required}" ]] || { echo "missing ${required}" >&2; exit 1; }
done

for anchor in inverse identity; do
  for eta in 0.01 0.03 0.1; do
    stem="${output_names[${anchor}:${eta}]}"
    for path in "${output_root}/${stem}.json" "${output_root}/${stem}.json.tmp" "${output_root}/${stem}.console.log"; do
      [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}" >&2; exit 1; }
    done
  done
done
[[ ! -e "${output_root}/selection.json" ]] || { echo "selection already exists" >&2; exit 1; }
[[ ! -e "${output_root}/selection.json.tmp" ]] || { echo "stale selection temp exists" >&2; exit 1; }

mkdir -p "${output_root}"
cd "${code_dir}"

run_one() {
  local anchor="$1"
  local eta="$2"
  local delta_max="$3"
  local stem="${output_names[${anchor}:${eta}]}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start ${stem}"
  "${python_bin}" evaluate_channel_opex.py \
    --base-checkpoint "${base_checkpoint}" \
    --channel-calibration "${calibration}" \
    --output "${output_root}/${stem}.json" \
    --env-name Walker2d-v4 \
    --device cuda \
    --step-size "${eta}" \
    --baseline-transform "${anchor}" \
    --gradient-noise-samples 8 \
    --gradient-steps 2 \
    --gradient-noise-seed 58300 \
    --delta-max "${delta_max}" \
    --rollout-action-noise-beta 1.25 \
    --eval-episodes 10 \
    --eval-seed 28300 \
    --eval-noise-seed 38300 \
    --reference-min-score 1.629008 \
    --reference-max-score 4592.3 \
    > "${output_root}/${stem}.console.log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) complete ${stem}"
}

for eta in 0.01 0.03 0.1; do
  run_one inverse "${eta}" 0.25
done
for eta in 0.01 0.03 0.1; do
  run_one identity "${eta}" 2.0
done

"${python_bin}" select_channel_opex_dev.py \
  --protocol "${protocol}" \
  --output "${output_root}/selection.json"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) CA-OPEX development selection complete"
