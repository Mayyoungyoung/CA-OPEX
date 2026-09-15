#!/usr/bin/env bash
set -euo pipefail

# Frozen paired evaluation block for the original-structure OPEX control and
# two stronger channel-aware OPEX controls.  The formal path is fail-closed:
# it accepts only a finalized strict manifest whose implementation/artifact
# hashes and six external contracts match this runner.  --dry-run only prints.

readonly research_root="/root/hubl_research_20260914"
readonly python_bin="/root/hubl_backup_env/bin/python"
readonly code_dir="${research_root}/code"
readonly results_root="${research_root}/results"
readonly output_dir="${results_root}/inverse_residual_confirm/external_controls"
readonly calibration="${results_root}/channel_calibration/beta125_n512_seed27001_calibration.json"
readonly manifest="${code_dir}/inverse_residual_frozen_confirmation_manifest.json"

readonly environment="Walker2d-v4"
readonly rollout_beta="1.25"
readonly episodes=50
readonly environment_seed=39300
readonly actuator_noise_seed=49300
readonly reference_min="1.629008"
readonly reference_max="4592.3"

dry_run=0
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
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

progress() {
  local run_id="$1"
  local stage="$2"
  local status="$3"
  printf '%s run_id=%s stage=%s status=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${run_id}" "${stage}" "${status}"
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

base_checkpoint_for_seed() {
  local seed="$1"
  printf '%s\n' \
    "${results_root}/inverse_residual_confirm/base_hubl_executed_25k_seed${seed}/latest.pt"
}

base_config_for_seed() {
  local seed="$1"
  printf '%s\n' \
    "${results_root}/inverse_residual_confirm/base_hubl_executed_25k_seed${seed}/config.json"
}

output_for() {
  local variant="$1"
  local seed="$2"
  case "${variant}" in
    original)
      printf '%s\n' "${output_dir}/opex_original_seed${seed}_beta125_50.json"
      ;;
    ca_inverse)
      printf '%s\n' "${output_dir}/ca_opex_inverse_seed${seed}_beta125_50.json"
      ;;
    ca_identity)
      printf '%s\n' "${output_dir}/ca_opex_identity_seed${seed}_beta125_50.json"
      ;;
    *)
      die "unknown OPEX variant: ${variant}"
      ;;
  esac
}

run_id_for() {
  local variant="$1"
  local seed="$2"
  case "${variant}" in
    original) printf '%s\n' "external_opex_original_seed${seed}" ;;
    ca_inverse) printf '%s\n' "external_ca_opex_inverse_seed${seed}" ;;
    ca_identity) printf '%s\n' "external_ca_opex_identity_seed${seed}" ;;
    *) die "unknown OPEX variant: ${variant}" ;;
  esac
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

verify_frozen_contract() {
  "${python_bin}" - \
    "${manifest}" "${code_dir}" "${results_root}" "${calibration}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
code_dir = Path(sys.argv[2]).resolve()
results_root = Path(sys.argv[3]).resolve()
calibration_path = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(code_dir))

from aggregate_inverse_residual_results import (  # noqa: E402
    IMPLEMENTATION_SOURCE_FILES,
    sha256_file,
    validate_strict_manifest_definition,
    verify_opex_development_selection,
)

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
validate_strict_manifest_definition(payload, allow_placeholders=False)
if "__FILL_" in json.dumps(payload, sort_keys=True):
    raise SystemExit("finalized manifest still contains a freeze placeholder")
if payload.get("schema_version") != "inverse-residual-manifest-v2":
    raise SystemExit("unexpected frozen manifest schema_version")
if payload.get("results_root") != "../results":
    raise SystemExit("frozen manifest results_root must resolve to the research results")
normalization = payload.get("normalization", {}).get("Walker2d-v4")
if not isinstance(normalization, dict) or {
    "reference_min_score": normalization.get("reference_min_score"),
    "reference_max_score": normalization.get("reference_max_score"),
} != {
    "reference_min_score": 1.629008,
    "reference_max_score": 4592.3,
}:
    raise SystemExit("frozen Walker2d-v4 normalization contract mismatch")

# The adapter block and the external OPEX block must both still point at the
# exact code that was present when the manifest was frozen.
for field, filename in IMPLEMENTATION_SOURCE_FILES.items():
    actual = sha256_file(code_dir / filename)
    expected = payload["frozen_implementation"].get(field)
    if actual != expected:
        raise SystemExit(
            f"frozen implementation hash mismatch for {filename}: "
            f"expected={expected!r}, actual={actual!r}"
        )

implementation_files = {
    "evaluate_sha256": "evaluate_channel_opex.py",
    "inverse_residual_core_sha256": "inverse_residual_core.py",
    "td3bc_core_sha256": "td3bc_core.py",
    "train_td3bc_sha256": "train_td3bc.py",
    "evaluation_controls_sha256": "evaluation_controls.py",
    "train_inverse_residual_adapter_sha256": "train_inverse_residual_adapter.py",
}
actual_implementation = {
    field: sha256_file(code_dir / filename)
    for field, filename in implementation_files.items()
}
calibration_sha = sha256_file(calibration_path)
calibrated_beta = 1.2498948872089386

protocol_metadata_positive = {
    "paired_environment_and_action_noise_seeds": True,
    "environment_rng_namespace": "gymnasium_env_reset",
    "actuator_noise_rng_namespace": "numpy_generator_per_episode",
    "gradient_noise_rng_namespace": (
        "torch_generator_per_episode_fresh_antithetic_per_gradient_step"
    ),
    "gradient_noise_sampling_frequency": "per_gradient_step",
    "gradient_noise_antithetic": True,
    "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling": True,
    "selection_rule": "all requested episodes retained",
}
protocol_metadata_zero = {
    **protocol_metadata_positive,
    "gradient_noise_rng_namespace": "none_beta_zero_deterministic_objective",
    "gradient_noise_sampling_frequency": "none_beta_zero",
    "gradient_noise_antithetic": False,
}

variants = {
    "original": {
        "path_prefix": "opex_original",
        "run_prefix": "external_opex_original",
        "method_id": "original_structure_opex_t1",
        "baseline_transform": "identity",
        "gradient_steps": 1,
        "K": 1,
        "model_beta": 0.0,
        "calibration_mode": "cli_known_beta_without_pair_calibration",
        "calibration_sha256": None,
        "delta_max": 2.0,
        "step_size": 0.1,
        "gradient_objective": (
            "q1_of_action_bounded_command_with_deterministic_zero_channel_noise"
        ),
        "protocol_metadata": protocol_metadata_zero,
    },
    "ca_inverse": {
        "path_prefix": "ca_opex_inverse",
        "run_prefix": "external_ca_opex_inverse",
        "method_id": "channel_aware_opex_inverse_anchor",
        "baseline_transform": "inverse",
        "gradient_steps": 2,
        "K": 8,
        "model_beta": calibrated_beta,
        "calibration_mode": "censored_uniform_plus_clip_pair_calibration",
        "calibration_sha256": calibration_sha,
        "delta_max": 0.25,
        "step_size": 0.1,
        "gradient_objective": (
            "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
            "uniform_noise_per_gradient_step"
        ),
        "protocol_metadata": protocol_metadata_positive,
    },
    "ca_identity": {
        "path_prefix": "ca_opex_identity",
        "run_prefix": "external_ca_opex_identity",
        "method_id": "channel_aware_opex_identity_anchor",
        "baseline_transform": "identity",
        "gradient_steps": 2,
        "K": 8,
        "model_beta": calibrated_beta,
        "calibration_mode": "censored_uniform_plus_clip_pair_calibration",
        "calibration_sha256": calibration_sha,
        "delta_max": 2.0,
        "step_size": 0.3,
        "gradient_objective": (
            "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
            "uniform_noise_per_gradient_step"
        ),
        "protocol_metadata": protocol_metadata_positive,
    },
}

expected_specs = {}
for variant in variants.values():
    for seed in (1, 10):
        run_id = f"{variant['run_prefix']}_seed{seed}"
        gradient_seed = 69300
        expected_config = {
            "baseline_transform": variant["baseline_transform"],
            "step_size": variant["step_size"],
            "gradient_steps": variant["gradient_steps"],
            "K": variant["K"],
            "execution_noise_samples": variant["K"],
            "model_beta": variant["model_beta"],
            "calibration_mode": variant["calibration_mode"],
            "gradient_noise_seed": gradient_seed,
            "q_reducer": "mean_q1",
            "delta_max": variant["delta_max"],
            "critic": "frozen_q1",
            "gradient_objective": variant["gradient_objective"],
            "gradient_steps_per_action": variant["gradient_steps"],
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
        }
        expected_protocol = {
            "environment": "Walker2d-v4",
            "episode_count": 50,
            "rollout_beta": 1.25,
            "environment_seed_start": 39300,
            "action_noise_seed_start": 49300,
            "gradient_noise_seed_start": gradient_seed,
            **variant["protocol_metadata"],
        }
        expected_specs[run_id] = {
            "method_id": variant["method_id"],
            "path": (
                "inverse_residual_confirm/external_controls/"
                f"{variant['path_prefix']}_seed{seed}_beta125_50.json"
            ),
            "training_seed": seed,
            "expected_config": expected_config,
            "expected_calibration_sha": variant["calibration_sha256"],
            "expected_protocol": expected_protocol,
            "expected_cost": {
                "cost_scope": "adapted_arm_deployment_controller_only",
                "base_actor_rows_per_adapted_environment_step": 1,
                "q1_rows_per_adapted_step": (
                    variant["gradient_steps"] * variant["K"]
                ),
                "q1_backward_calls_per_adapted_step": variant["gradient_steps"],
            },
        }

opex_specs = [
    spec
    for spec in payload.get("external_evaluations", [])
    if spec.get("raw_schema") == "channel_opex_v1"
]
actual_ids = [spec.get("run_id") for spec in opex_specs]
if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_specs):
    raise SystemExit(
        f"frozen OPEX run IDs mismatch: expected={sorted(expected_specs)}, "
        f"actual={sorted(actual_ids)}"
    )

for spec in opex_specs:
    run_id = spec["run_id"]
    expected = expected_specs[run_id]
    scalar_fields = {
        "method_id": expected["method_id"],
        "path": expected["path"],
        "raw_schema": "channel_opex_v1",
        "evidence_label": "confirmation",
        "training_seed": expected["training_seed"],
    }
    for field, value in scalar_fields.items():
        if spec.get(field) != value:
            raise SystemExit(
                f"frozen OPEX {run_id} {field} mismatch: "
                f"expected={value!r}, actual={spec.get(field)!r}"
            )
    if spec.get("expected_config") != expected["expected_config"]:
        raise SystemExit(f"frozen OPEX {run_id} controller contract mismatch")
    if spec.get("expected_implementation") != actual_implementation:
        raise SystemExit(f"frozen OPEX {run_id} implementation hashes mismatch")
    if spec.get("expected_cost") != expected["expected_cost"]:
        raise SystemExit(f"frozen OPEX {run_id} cost contract mismatch")

    seed = expected["training_seed"]
    checkpoint = (
        results_root
        / "inverse_residual_confirm"
        / f"base_hubl_executed_25k_seed{seed}"
        / "latest.pt"
    )
    checkpoint_config = checkpoint.with_name("config.json")
    expected_artifacts = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint_config),
        "calibration_sha256": expected["expected_calibration_sha"],
    }
    if spec.get("expected_artifacts") != expected_artifacts:
        raise SystemExit(f"frozen OPEX {run_id} artifact hashes mismatch")

    evaluations = spec.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise SystemExit(f"frozen OPEX {run_id} must have one evaluation")
    evaluation = evaluations[0]
    for field, value in {
        "evaluation_id": "confirm_beta125",
        "raw_arm": "paired_controller",
        "evidence_label": "confirmation",
    }.items():
        if evaluation.get(field) != value:
            raise SystemExit(f"frozen OPEX {run_id} evaluation {field} mismatch")
    if evaluation.get("expected_protocol") != expected["expected_protocol"]:
        raise SystemExit(f"frozen OPEX {run_id} protocol contract mismatch")
    selection = spec.get("development_selection")
    if selection is not None:
        verification = verify_opex_development_selection(
            implementation_root=code_dir,
            results_root=results_root,
            selection_spec=selection,
            expected_implementation=actual_implementation,
        )
        if verification.get("candidate_count") != 10:
            raise SystemExit(
                f"frozen OPEX {run_id} development inventory is not ten candidates"
            )
PY
}

preflight_outputs() {
  local seed variant output log
  for seed in 1 10; do
    for variant in original ca_inverse ca_identity; do
      output="$(output_for "${variant}" "${seed}")"
      log="${output%.json}.console.log"
      require_absent "${output}" "${output}.tmp" "${log}"
    done
  done
}

run_case() {
  local seed="$1"
  local variant="$2"
  local baseline_transform="$3"
  local k="$4"
  local gradient_steps="$5"
  local delta_max="$6"
  local selected_step_size="$7"
  local channel_mode="$8"
  local output run_id log checkpoint gradient_seed
  local channel_args=()

  output="$(output_for "${variant}" "${seed}")"
  run_id="$(run_id_for "${variant}" "${seed}")"
  log="${output%.json}.console.log"
  checkpoint="$(base_checkpoint_for_seed "${seed}")"
  gradient_seed="69300"
  if [[ "${channel_mode}" == "known_beta_zero" ]]; then
    channel_args=(--model-action-noise-beta 0.0)
  elif [[ "${channel_mode}" == "calibrated" ]]; then
    channel_args=(--channel-calibration "${calibration}")
  else
    die "unknown channel mode: ${channel_mode}"
  fi

  if [[ "${dry_run}" -eq 0 ]]; then
    require_absent "${output}" "${output}.tmp" "${log}"
  fi
  progress "${run_id}" paired_evaluation started
  if ! run_logged "${log}" \
    "${python_bin}" evaluate_channel_opex.py \
      --base-checkpoint "${checkpoint}" \
      "${channel_args[@]}" \
      --output "${output}" \
      --env-name "${environment}" \
      --device cuda \
      --step-size "${selected_step_size}" \
      --baseline-transform "${baseline_transform}" \
      --gradient-noise-samples "${k}" \
      --gradient-steps "${gradient_steps}" \
      --gradient-noise-seed "${gradient_seed}" \
      --delta-max "${delta_max}" \
      --rollout-action-noise-beta "${rollout_beta}" \
      --eval-episodes "${episodes}" \
      --eval-seed "${environment_seed}" \
      --eval-noise-seed "${actuator_noise_seed}" \
      --reference-min-score "${reference_min}" \
      --reference-max-score "${reference_max}"; then
    progress "${run_id}" paired_evaluation failed
    die "channel-OPEX evaluation failed; inspect ${log}"
  fi
  if [[ "${dry_run}" -eq 0 ]]; then
    require_file "${output}"
  fi
  progress "${run_id}" paired_evaluation completed
}

if [[ "${dry_run}" -eq 0 ]]; then
  require_file "${python_bin}"
  require_file "${manifest}"
  require_file "${calibration}"
  require_file "${code_dir}/aggregate_inverse_residual_results.py"
  require_file "${code_dir}/evaluate_channel_opex.py"
  require_file "${code_dir}/inverse_residual_core.py"
  require_file "${code_dir}/td3bc_core.py"
  require_file "${code_dir}/train_td3bc.py"
  require_file "${code_dir}/evaluation_controls.py"
  require_file "${code_dir}/train_inverse_residual_adapter.py"
  for seed in 1 10; do
    require_file "$(base_checkpoint_for_seed "${seed}")"
    require_file "$(base_config_for_seed "${seed}")"
  done
  verify_frozen_contract
  preflight_outputs
  mkdir -p "${output_dir}"
  cd "${code_dir}"
else
  echo "Static dry run only; no files, directories, models, or results will be created."
fi

progress driver frozen_channel_opex_block started
for seed in 1 10; do
  run_case "${seed}" original identity 1 1 2.0 0.1 known_beta_zero
  run_case "${seed}" ca_inverse inverse 8 2 0.25 0.1 calibrated
  run_case "${seed}" ca_identity identity 8 2 2.0 0.3 calibrated
done
progress driver frozen_channel_opex_block completed

if [[ "${dry_run}" -eq 1 ]]; then
  echo "Static dry run complete."
else
  echo "Frozen channel-OPEX confirmation block completed in ${output_dir}."
fi
