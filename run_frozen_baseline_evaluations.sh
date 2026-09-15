#!/usr/bin/env bash
set -euo pipefail

# Frozen, predeclared Walker2d confirmation block for non-adapter controls.
# Every output is immutable: an existing JSON aborts instead of being replaced.

research_root="/root/hubl_research_20260914"
python_bin="/root/hubl_backup_env/bin/python"
code_dir="${research_root}/code"
result_dir="${research_root}/results/inverse_residual_confirm/baseline_evals"
episodes=50
environment_seed=39300
actuator_noise_seed=49300

mkdir -p "${result_dir}"
cd "${code_dir}"

run_eval() {
  local checkpoint_path="$1"
  local output_path="$2"
  local rollout_beta="$3"
  local command_scale="$4"
  local command_transform="$5"
  if [[ -e "${output_path}" || -e "${output_path}.tmp" ]]; then
    echo "refusing to overwrite ${output_path}" >&2
    return 1
  fi
  "${python_bin}" evaluate_td3bc.py \
    --checkpoint "${checkpoint_path}" \
    --output "${output_path}" \
    --device cuda \
    --eval-episodes "${episodes}" \
    --eval-seed "${environment_seed}" \
    --eval-noise-seed "${actuator_noise_seed}" \
    --eval-action-noise-beta "${rollout_beta}" \
    --command-scale "${command_scale}" \
    --command-transform "${command_transform}" \
    > "${output_path%.json}.console.log" 2>&1
}

for training_seed in 1 10; do
  base_checkpoint="${research_root}/results/inverse_residual_confirm/base_hubl_executed_25k_seed${training_seed}/latest.pt"
  dual_checkpoint="${research_root}/results/inverse_residual_confirm/dual_hubl_executed_commanded_25k_seed${training_seed}/latest.pt"

  run_eval "${base_checkpoint}" "${result_dir}/base_identity_seed${training_seed}_beta125.json" 1.25 1.0 identity
  run_eval "${base_checkpoint}" "${result_dir}/base_scalar12_seed${training_seed}_beta125.json" 1.25 1.2 identity
  run_eval "${base_checkpoint}" "${result_dir}/base_oracle_inverse_seed${training_seed}_beta125.json" 1.25 1.0 uniform_mean_inverse
  run_eval "${dual_checkpoint}" "${result_dir}/dual_identity_seed${training_seed}_beta125.json" 1.25 1.0 identity

  run_eval "${base_checkpoint}" "${result_dir}/base_identity_seed${training_seed}_clean.json" 0.0 1.0 identity
  run_eval "${base_checkpoint}" "${result_dir}/base_scalar12_seed${training_seed}_clean.json" 0.0 1.2 identity
  run_eval "${dual_checkpoint}" "${result_dir}/dual_identity_seed${training_seed}_clean.json" 0.0 1.0 identity
done

echo "completed frozen baseline evaluations in ${result_dir}"
