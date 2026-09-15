#!/usr/bin/env bash
set -euo pipefail

# Sequential chain for the frozen beta-severity grid: 4 betas x 3 predeclared
# checkpoints = 12 create-only units, run one at a time on the single GPU.
# Each unit reuses run_ca_opex_walker_betagrid.sh, which owns its own
# fail-closed preflight, calibration generation, and per-unit logs.  On any
# failure the chain stops and the failed unit directory is preserved for
# diagnosis; re-running the chain skips completed units only if invoked again
# by the operator after inspection (this script itself is one-shot per unit
# because the unit driver refuses existing roots).

readonly research_root="/root/hubl_research_20260914"
readonly code_dir="${research_root}/code"
readonly chain_log="${research_root}/results/betagrid_chain_20260915.nohup.log"

progress() {
  printf '%s betagrid_chain stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" "$2" | tee -a "${chain_log}"
}

for beta in 0.5 0.9 1.1 1.4; do
  for seed in 1 2 10; do
    progress "unit_beta${beta}_seed${seed}" started
    bash "${code_dir}/run_ca_opex_walker_betagrid.sh" \
      --seed "${seed}" --beta "${beta}" >> "${chain_log}" 2>&1
    progress "unit_beta${beta}_seed${seed}" completed
  done
done

progress betagrid_complete completed
echo "Beta-severity grid completed; aggregate with aggregate_ca_opex_walker_betagrid.py."
