#!/usr/bin/env bash
set -euo pipefail

# One-shot Hopper-v4 data pipeline for the CA-OPEX cross-task replication:
# validate the downloaded D4RL source, collect a small smoke set, collect the
# full 1M-transition beta-1 dataset, then fit the frozen 512-pair beta-1.25
# calibration.  All outputs are create-only; the script never overwrites and
# never deletes a failed artifact.

readonly research_root="/root/hubl_research_20260914"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly code_dir="${research_root}/code"
readonly source_hdf5="/root/hubl_backup_data/hopper_medium-v2.hdf5"
readonly source_expected_size=152756557
readonly data_dir="/root/hubl_backup_data/action_noise_dev"
readonly smoke_hdf5="${data_dir}/hopper_v4_beta1_smoke10k.hdf5"
readonly dataset_hdf5="${data_dir}/hopper_v4_beta1_seed2201.hdf5"
readonly pairs_npz="${research_root}/results/channel_calibration/hopper_beta125_n512_seed27002.npz"
readonly calibration_json="${research_root}/results/channel_calibration/hopper_beta125_n512_seed27002_calibration.json"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required input is missing: $1"
}

require_absent() {
  [[ ! -e "$1" ]] || die "refusing to reuse or overwrite existing path: $1"
}

progress() {
  printf '%s hopper_pipeline stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" "$2"
}

require_file "${python_bin}"
require_file "${code_dir}/collect_stochastic.py"
require_file "${source_hdf5}"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

actual_size=$(stat -c %s "${source_hdf5}")
[[ "${actual_size}" == "${source_expected_size}" ]] || \
  die "source size mismatch: ${actual_size} != ${source_expected_size}"
progress source_size_ok completed
sha256sum "${source_hdf5}" | tee "${source_hdf5}.sha256"
progress source_hash_recorded completed

require_absent "${smoke_hdf5}"
progress smoke_collection started
"${python_bin}" "${code_dir}/collect_stochastic.py" \
  --source "${source_hdf5}" \
  --output "${smoke_hdf5}" \
  --env Hopper-v4 \
  --beta 1.0 \
  --transitions 10000 \
  --env-seed 2201 \
  --noise-seed 2210 \
  --policy-seed 2211 \
  --audit-seed 2212 \
  --device cpu \
  --progress-every 5000 \
  > "${smoke_hdf5}.console.log" 2>&1
require_file "${smoke_hdf5}"
sha256sum "${smoke_hdf5}" | tee "${smoke_hdf5}.sha256"
progress smoke_collection completed

require_absent "${dataset_hdf5}"
progress full_collection started
"${python_bin}" "${code_dir}/collect_stochastic.py" \
  --source "${source_hdf5}" \
  --output "${dataset_hdf5}" \
  --env Hopper-v4 \
  --beta 1.0 \
  --transitions 1000000 \
  --env-seed 2201 \
  --noise-seed 2210 \
  --policy-seed 2211 \
  --audit-seed 2212 \
  --device cpu \
  --progress-every 100000 \
  > "${dataset_hdf5}.console.log" 2>&1
require_file "${dataset_hdf5}"
sha256sum "${dataset_hdf5}" | tee "${dataset_hdf5}.sha256"
progress full_collection completed

require_absent "${pairs_npz}"
progress calibration_pairs started
"${python_bin}" "${code_dir}/generate_channel_pairs.py" \
  --source "${dataset_hdf5}" \
  --output "${pairs_npz}" \
  --pairs 512 \
  --beta 1.25 \
  --seed 27002
progress calibration_pairs completed

require_absent "${calibration_json}"
progress calibration_fit started
"${python_bin}" "${code_dir}/calibrate_uniform_channel.py" \
  --pairs "${pairs_npz}" \
  --output "${calibration_json}" \
  --bootstrap-replicates 1000 \
  --bootstrap-seed 28002
require_file "${calibration_json}"
sha256sum "${calibration_json}" | tee "${calibration_json}.sha256"
progress calibration_fit completed

echo "Hopper pipeline complete; dataset and calibration SHAs recorded."
