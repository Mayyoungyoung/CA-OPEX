import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_inverse_residual_results import (
    AggregationError,
    PRIMARY_CONFIRMATION_COMPARISON_IDS,
    _primary_comparison_training_seed_evidence,
    _recompute_opex_candidate_metrics,
    build_report,
    freeze_strict_manifest_template,
    paired_statistics,
    render_csv,
    render_markdown,
    validate_strict_manifest_definition,
    write_report,
)


REFERENCE_MIN = 1.0
REFERENCE_MAX = 101.0


def _normalized(values):
    return [(value - REFERENCE_MIN) for value in values]


def _arm(returns, *, env_seeds=(10, 11), noise_seeds=(20, 21), residual=0.0):
    values = [float(value) for value in returns]
    normalized = _normalized(values)
    mean = sum(values) / len(values)
    norm_mean = sum(normalized) / len(normalized)
    population_std = (
        sum((value - mean) ** 2 for value in values) / len(values)
    ) ** 0.5
    norm_std = (
        sum((value - norm_mean) ** 2 for value in normalized) / len(normalized)
    ) ** 0.5
    return {
        "returns": values,
        "lengths": [5 for _ in values],
        "environment_seeds": list(env_seeds),
        "action_noise_seeds": list(noise_seeds),
        "action_noise_beta": 1.25,
        "return_mean": mean,
        "return_std": population_std,
        "normalized_score_mean": norm_mean,
        "normalized_score_std": norm_std,
        "actuator_clip_fraction": 0.2,
        "command_at_bound_fraction": 0.3,
        "inverse_saturation_fraction": 0.25,
        "executed_commanded_action_mse": 0.4,
        "proposed_residual_abs_mean": residual,
        "proposed_residual_abs_max": residual * 2,
        "applied_residual_abs_mean": residual * 0.8,
    }


def _write_complete_run(root: Path, name: str, *, transform: str, seed: int = 0):
    run = root / name
    run.mkdir(parents=True)
    config = {
        "arguments": {"train_seed": seed, "updates": 50},
        "adapter": {
            "baseline_transform": transform,
            "residual_penalty": 0.01,
            "alpha": 1.0,
            "delta_max": 0.25,
            "execution_noise_samples": 8,
            "execution_noise_beta": 1.25,
        },
    }
    summary = {
        "status": "complete",
        "updates": 50,
        "wall_time_seconds": 12.5,
        "preprocessing_seconds_this_invocation": 2.0,
        "optimizer_seconds": 10.0,
        "q1_input_rows": 4096,
        "q1_rows_per_state_per_update": 16,
        "base_parameters_unchanged": True,
        "channel": {"half_vectors_drawn": 100, "draw_calls": 50},
    }
    baseline = _arm([1.0, 3.0])
    adapted = _arm([2.0, 4.0], residual=0.1)
    evaluation = {
        "status": "complete",
        "environment": "Walker2d-v4",
        "adapter_step": 50,
        "adapter_checkpoint_sha256": "adapter-hash",
        "base_checkpoint_sha256": "base-hash",
        "baseline_transform": transform,
        "channel": {
            "environment_rollout_beta": 1.25,
            "model_or_calibration_beta": 1.25,
            "beta_mismatch": False,
        },
        "evaluation_protocol": {"episode_count_per_method": 2},
        "baseline_only": baseline,
        "adapted": adapted,
        "paired": {"adapted_minus_baseline_returns": [1.0, 1.0]},
        "wall_time_seconds": 3.0,
    }
    audit = {
        "status": "complete",
        "baseline_transform": transform,
        "adapter_step": 50,
        "wall_time_seconds": 1.0,
        "audit_channel": {
            "seed": 99,
            "sample_count": 64,
            "audit_action_noise_beta": 1.25,
            "model_or_calibration_beta": 1.25,
            "beta_mismatch": False,
            "training_noise_seed": 123,
            "noise_seed_distinct_from_training_channel": True,
        },
        "dataset": {
            "selected_observations": 4,
            "selected_indices_sha256": "indices-hash",
            "state_seed": 77,
        },
        "cost": {
            "physical_action_rows": 512,
            "individual_critic_network_rows": 1024,
        },
        "metrics": {
            "adapted_minus_baseline_q1": {
                "mean": 0.4,
                "median": 0.35,
                "std": 0.1,
                "positive_fraction": 0.75,
                "min": -0.1,
                "max": 0.8,
            },
            "adapted_minus_baseline_q2": {
                "mean": 0.2,
                "median": 0.15,
                "std": 0.2,
                "positive_fraction": 0.6,
                "min": -0.2,
                "max": 0.6,
            },
            "adapted_minus_baseline_min_twin": {
                "mean": 0.3,
                "median": 0.25,
                "std": 0.2,
                "positive_fraction": 0.7,
                "min": -0.2,
                "max": 0.7,
            },
            "baseline_absolute_twin_disagreement": {
                "mean": 0.5,
                "median": 0.45,
                "std": 0.2,
                "positive_fraction": 1.0,
                "min": 0.1,
                "max": 0.9,
            },
            "adapted_absolute_twin_disagreement": {
                "mean": 0.52,
                "median": 0.47,
                "std": 0.2,
                "positive_fraction": 1.0,
                "min": 0.1,
                "max": 0.9,
            },
            "adapted_minus_baseline_twin_disagreement": {
                "mean": 0.02,
                "median": 0.01,
                "std": 0.03,
                "positive_fraction": 0.55,
                "min": -0.1,
                "max": 0.2,
            },
            "command_saturation_fraction": 0.2,
            "proposed_residual_abs": {
                "mean": 0.1,
                "median": 0.08,
                "std": 0.05,
                "positive_fraction": 1.0,
                "min": 0.01,
                "max": 0.25,
            },
        },
        "interpretation_guardrail": "diagnostic only",
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "latest.pt").write_bytes(b"checkpoint")
    (run / "progress.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "launch.log").write_text("complete\n", encoding="utf-8")
    (run.parent / f"{run.name}.train.console.log").write_text(
        "complete\n", encoding="utf-8"
    )
    (run / "eval.json").write_text(json.dumps(evaluation), encoding="utf-8")
    (run / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return run


def _write_failure(root: Path, name: str):
    run = root / name
    run.mkdir(parents=True)
    config = {
        "arguments": {"train_seed": 0, "updates": 50},
        "adapter": {
            "baseline_transform": "inverse",
            "residual_penalty": 0.01,
            "alpha": 1.0,
            "delta_max": 0.25,
            "execution_noise_samples": 8,
            "execution_noise_beta": 1.25,
        },
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "progress.jsonl").write_text("", encoding="utf-8")
    (run / "launch.log").write_text("failed before update\n", encoding="utf-8")
    return run


def _manifest(tmp_path: Path):
    results = tmp_path / "results"
    _write_complete_run(results, "inverse", transform="inverse")
    _write_complete_run(results, "direct", transform="identity")
    _write_failure(results, "failed")
    manifest = {
        "schema_version": "inverse-residual-manifest-v1",
        "evidence_scope": "fixture development",
        "results_root": "results",
        "bootstrap": {"seed": 7, "samples": 200},
        "normalization": {
            "Walker2d-v4": {
                "reference_min_score": REFERENCE_MIN,
                "reference_max_score": REFERENCE_MAX,
                "source": "fixture",
            }
        },
        "runs": [
            {
                "run_id": "inverse",
                "method_id": "inverse_residual",
                "training_dir": "inverse",
                "evidence_label": "development",
                "expected_status": "complete",
                "evaluations": [
                    {
                        "evaluation_id": "dev",
                        "path": "inverse/eval.json",
                        "evidence_label": "development",
                    }
                ],
                "audits": [
                    {
                        "audit_id": "q",
                        "path": "inverse/audit.json",
                        "evidence_label": "development",
                    }
                ],
            },
            {
                "run_id": "direct",
                "method_id": "direct_residual",
                "training_dir": "direct",
                "evidence_label": "development",
                "expected_status": "complete",
                "evaluations": [
                    {
                        "evaluation_id": "dev",
                        "path": "direct/eval.json",
                        "evidence_label": "development",
                    }
                ],
                "audits": [],
            },
            {
                "run_id": "failed",
                "method_id": "inverse_residual",
                "training_dir": "failed",
                "evidence_label": "development",
                "expected_status": "failed_preupdate",
            },
        ],
        "comparisons": [
            {
                "comparison_id": "inverse-vs-baseline",
                "evidence_label": "development",
                "target": {
                    "run_id": "inverse",
                    "evaluation_id": "dev",
                    "arm": "adapted",
                },
                "reference": {
                    "run_id": "inverse",
                    "evaluation_id": "dev",
                    "arm": "baseline_only",
                },
            },
            {
                "comparison_id": "inverse-vs-direct",
                "evidence_label": "development",
                "target": {
                    "run_id": "inverse",
                    "evaluation_id": "dev",
                    "arm": "adapted",
                },
                "reference": {
                    "run_id": "direct",
                    "evaluation_id": "dev",
                    "arm": "adapted",
                },
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _strict_manifest(tmp_path: Path):
    results = tmp_path / "strict_results"
    run = _write_complete_run(results, "inverse", transform="inverse", seed=1)
    implementation = {
        "train_sha256": "1" * 64,
        "core_sha256": "2" * 64,
        "evaluation_controls_sha256": "3" * 64,
        "td3bc_core_sha256": "4" * 64,
        "train_iql_sha256": "5" * 64,
        "evaluate_sha256": "6" * 64,
        "audit_sha256": "7" * 64,
    }
    dataset_sha = "8" * 64
    base_sha = "9" * 64
    calibration_sha = "a" * 64
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["arguments"].update(
        {
            "learning_rate": 3e-4,
            "batch_size": 256,
            "split_seed": 424242,
            "audit_fraction": 0.1,
            "precompute_batch_size": 8192,
            "scale_calibration_observations": 4096,
            "max_observations": None,
        }
    )
    config["adapter"].update(
        {
            "value_estimator": "sampled_expected_q1",
            "execution_noise_seed": 271829,
            "hidden_dim": 128,
            "depth": 2,
            "q_scale_epsilon": 1e-6,
        }
    )
    config.update(
        {
            "dataset": {
                "sha256": dataset_sha,
                "training_observation_count": 10,
            },
            "base_checkpoint_sha256": base_sha,
            "base_action_pairing": "executed_executed",
            "channel_calibration": {
                "source": "censored_uniform_plus_clip_pair_calibration",
                "calibration_sha256": calibration_sha,
            },
            "implementation": {
                key: implementation[key]
                for key in (
                    "train_sha256",
                    "core_sha256",
                    "evaluation_controls_sha256",
                    "td3bc_core_sha256",
                    "train_iql_sha256",
                )
            },
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "resumed": False,
            "resume_count": 0,
            "q1_input_rows": 50 * 256 * 2 * 8,
            "q1_rows_per_state_per_update": 16,
            "baseline_precomputation": {"observation_count": 10},
            "value_scale_calibration": {
                "observation_count": 10,
                "q1_input_rows": 10 * 8,
                "scale": 2.5,
                "method": "frozen_std_of_declared_baseline_value_estimator_q1",
            },
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    checkpoint_path = run / "latest.pt"
    torch.save(
        {
            "format": "inverse_residual_adapter_v2",
            "adapter_config": config["adapter"],
            "base_checkpoint_sha256": base_sha,
            "base_parameter_sha256": "b" * 64,
            "step": summary["updates"],
            "config": config,
        },
        checkpoint_path,
    )
    checkpoint_sha = __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest()
    eval_path = run / "eval.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    evaluation.update(
        {
            "adapter_checkpoint_sha256": checkpoint_sha,
            "base_checkpoint_sha256": base_sha,
            "implementation": {
                "train_sha256": implementation["train_sha256"],
                "core_sha256": implementation["core_sha256"],
                "evaluate_sha256": implementation["evaluate_sha256"],
            },
        }
    )
    eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
    audit_path = run / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "adapter_checkpoint_sha256": checkpoint_sha,
            "base_checkpoint_sha256": base_sha,
            "implementation": {
                "train_sha256": implementation["train_sha256"],
                "core_sha256": implementation["core_sha256"],
                "evaluate_sha256": implementation["evaluate_sha256"],
                "audit_sha256": implementation["audit_sha256"],
            },
        }
    )
    audit["dataset"].update(
        {
            "sha256": dataset_sha,
            "train_indices_sha256": "c" * 64,
            "selected_indices_sha256": "d" * 64,
            "train_audit_disjoint": True,
            "recorded_split": {"audit_episode_units_sha256": "e" * 64},
        }
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    expected_config = {
        "baseline_transform": "inverse",
        "value_estimator": "sampled_expected_q1",
        "alpha": 1.0,
        "residual_penalty": 0.01,
        "delta_max": 0.25,
        "execution_noise_samples": 8,
        "updates": 50,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "split_seed": 424242,
        "audit_fraction": 0.1,
        "train_seed": 1,
        "channel_seed": 271829,
        "model_or_calibration_beta": 1.25,
        "calibration_mode": "pair_calibration",
        "hidden_dim": 128,
        "depth": 2,
        "q_scale_epsilon": 1e-6,
        "precompute_batch_size": 8192,
        "scale_calibration_observations": 4096,
        "max_observations": None,
        "base_action_pairing": "executed_executed",
        "extensions": {},
    }
    manifest = {
        "schema_version": "inverse-residual-manifest-v2",
        "evidence_scope": "strict fixture",
        "results_root": "strict_results",
        "bootstrap": {"seed": 7, "samples": 100},
        "normalization": {
            "Walker2d-v4": {
                "reference_min_score": REFERENCE_MIN,
                "reference_max_score": REFERENCE_MAX,
                "source": "fixture",
            }
        },
        "frozen_implementation": implementation,
        "runs": [
            {
                "run_id": "inverse",
                "method_id": "inverse_sampled_main",
                "training_dir": "inverse",
                "evidence_label": "confirmation",
                "expected_status": "complete",
                "expected_config": expected_config,
                "expected_artifacts": {
                    "dataset_sha256": dataset_sha,
                    "base_checkpoint_sha256": base_sha,
                    "calibration_sha256": calibration_sha,
                    "checkpoint_format": "inverse_residual_adapter_v2",
                },
                "evaluations": [
                    {
                        "evaluation_id": "confirm",
                        "path": "inverse/eval.json",
                        "evidence_label": "confirmation",
                        "expected_protocol": {
                            "environment": "Walker2d-v4",
                            "episode_count": 2,
                            "rollout_beta": 1.25,
                            "environment_seed_start": 10,
                            "action_noise_seed_start": 20,
                        },
                    }
                ],
                "audits": [
                    {
                        "audit_id": "heldout",
                        "path": "inverse/audit.json",
                        "evidence_label": "confirmation",
                        "expected_protocol": {
                            "noise_samples": 64,
                            "noise_seed": 99,
                            "action_noise_beta": 1.25,
                            "selected_observations": 4,
                            "train_indices_sha256": "c" * 64,
                            "audit_indices_sha256": "d" * 64,
                            "audit_episode_units_sha256": "e" * 64,
                        },
                    }
                ],
            }
        ],
        "comparisons": [
            {
                "comparison_id": "effect",
                "evidence_label": "confirmation",
                "target": {
                    "run_id": "inverse",
                    "evaluation_id": "confirm",
                    "arm": "adapted",
                },
                "reference": {
                    "run_id": "inverse",
                    "evaluation_id": "confirm",
                    "arm": "baseline_only",
                },
                "require_same_base_checkpoint": True,
            }
        ],
    }
    path = tmp_path / "strict_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _append_external_td3bc(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_root = manifest_path.parent / manifest["results_root"]
    external_dir = results_root / "external_td3bc"
    external_dir.mkdir()
    checkpoint = external_dir / "latest.pt"
    checkpoint.write_bytes(b"external-td3bc-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training_config = {
        "arguments": {
            "variant": "hubl_constant",
            "action_pairing": "executed_executed",
            "train_seed": 1,
            "updates": 25000,
        }
    }
    config_path = external_dir / "config.json"
    config_path.write_text(json.dumps(training_config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    persistent = _arm([2.0, 4.0])
    clean_source = _arm([3.0, 5.0])
    clean = {
        key: value
        for key, value in clean_source.items()
        if key not in {"environment_seeds", "action_noise_seeds", "action_noise_beta"}
    }
    clean["episode_seeds"] = [10, 11]
    implementation = {
        "evaluate_td3bc_sha256": "1" * 64,
        "evaluation_controls_sha256": "2" * 64,
        "train_td3bc_sha256": "3" * 64,
    }
    raw = {
        "status": "complete",
        "environment": "Walker2d-v4",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "command_transform": "identity",
        "command_scale": 1.2,
        "command_transform_beta": 1.25,
        "implementation": implementation,
        "persistent_action_noise": persistent,
        "clean": clean,
        "wall_time_seconds": 4.0,
    }
    raw_path = external_dir / "eval.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest["external_evaluations"] = [
        {
            "run_id": "external_scalar_seed1",
            "method_id": "external_scalar12",
            "path": "external_td3bc/eval.json",
            "raw_schema": "evaluate_td3bc_v1",
            "evidence_label": "confirmation",
            "training_seed": 1,
            "expected_artifacts": {
                "raw_file_sha256": raw_sha,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_config_sha256": config_sha,
            },
            "expected_config": {
                "command_transform": "identity",
                "command_scale": 1.2,
                "command_transform_beta": 1.25,
                "checkpoint_variant": "hubl_constant",
                "checkpoint_action_pairing": "executed_executed",
                "checkpoint_train_seed": 1,
                "checkpoint_updates": 25000,
            },
            "expected_implementation": implementation,
            "evaluations": [
                {
                    "evaluation_id": "noisy",
                    "raw_arm": "persistent_action_noise",
                    "evidence_label": "confirmation",
                    "expected_protocol": {
                        "environment": "Walker2d-v4",
                        "episode_count": 2,
                        "rollout_beta": 1.25,
                        "environment_seed_start": 10,
                        "action_noise_seed_start": 20,
                    },
                },
                {
                    "evaluation_id": "clean",
                    "raw_arm": "clean",
                    "evidence_label": "confirmation",
                    "expected_protocol": {
                        "environment": "Walker2d-v4",
                        "episode_count": 2,
                        "rollout_beta": 0.0,
                        "environment_seed_start": 10,
                        "action_noise_seed_start": None,
                    },
                },
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return raw_path


def _append_external_opex(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_root = manifest_path.parent / manifest["results_root"]
    external_dir = results_root / "external_opex"
    external_dir.mkdir()
    checkpoint = external_dir / "latest.pt"
    checkpoint.write_bytes(b"external-opex-base")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path = external_dir / "config.json"
    config_path.write_text(
        json.dumps({"arguments": {"train_seed": 1}}), encoding="utf-8"
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    calibration_sha = "a" * 64
    controller = {
        "baseline_transform": "inverse",
        "step_size": 0.03,
        "gradient_steps": 2,
        "K": 8,
        "execution_noise_samples": 8,
        "model_beta": 1.25,
        "calibration_mode": "censored_uniform_plus_clip_pair_calibration",
        "gradient_noise_seed": 59300,
        "q_reducer": "mean_q1",
        "delta_max": 0.25,
        "critic": "frozen_q1",
        "gradient_objective": (
            "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
            "uniform_noise_per_gradient_step"
        ),
        "gradient_steps_per_action": 2,
        "actor_parameter_updates": 0,
        "critic_parameter_updates": 0,
    }
    baseline = _arm([1.0, 3.0])
    adapted = _arm([2.0, 4.0], residual=0.1)
    for arm in (baseline, adapted):
        arm["gradient_noise_seeds"] = [30, 31]
    implementation = {
        "evaluate_sha256": "1" * 64,
        "inverse_residual_core_sha256": "2" * 64,
        "td3bc_core_sha256": "3" * 64,
        "train_td3bc_sha256": "4" * 64,
        "evaluation_controls_sha256": "5" * 64,
        "train_inverse_residual_adapter_sha256": "6" * 64,
    }
    raw = {
        "raw_schema": "channel_opex_v1",
        "status": "complete",
        "method_id": "channel_aware_opex_inverse_anchor",
        "environment": "Walker2d-v4",
        "base_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "calibration": {"calibration_sha256": calibration_sha},
        "channel": {
            "gradient_model": "iid_uniform_additive_then_clip",
            "model_or_calibration_beta": 1.25,
            "environment_rollout_beta": 1.25,
            "beta_mismatch": False,
        },
        "controller": controller,
        "evaluation_protocol": {
            "environment": "Walker2d-v4",
            "episode_count_per_arm": 2,
            "paired_environment_and_action_noise_seeds": True,
            "environment_rng_namespace": "gymnasium_env_reset",
            "actuator_noise_rng_namespace": "numpy_generator_per_episode",
            "gradient_noise_rng_namespace": (
                "torch_generator_per_episode_fresh_antithetic_per_gradient_step"
            ),
            "gradient_noise_sampling_frequency": "per_gradient_step",
            "gradient_noise_antithetic": True,
            "gradient_noise_seed_start": 30,
            "environment_seed_start": 10,
            "action_noise_seed_start": 20,
            "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling": True,
            "selection_rule": "all requested episodes retained",
        },
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
        },
        "arms": {"baseline_only": baseline, "adapted": adapted},
        "paired": {"adapted_minus_baseline_returns": [1.0, 1.0]},
        "cost": {
            "cost_scope": "adapted_arm_deployment_controller_only",
            "environment_steps": 10,
            "q1_forward_rows": 160,
            "q1_backward_rows": 160,
            "q1_backward_calls": 20,
            "base_actor_rows": 10,
            "baseline_environment_steps": 10,
            "baseline_base_actor_rows": 10,
            "paired_evaluation_environment_steps": 20,
            "baseline_q1_rows_total": 0,
            "adapted_q1_rows_total": 160,
            "adapted_q1_rows_per_environment_step": 16,
            "adapted_backward_calls": 20,
            "wall_time_seconds": 2.0,
            "baseline_wall_time_seconds": 1.5,
            "paired_evaluation_wall_time_seconds": 5.0,
        },
        "implementation": implementation,
        "wall_time_seconds": 5.0,
    }
    raw_path = external_dir / "eval.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    manifest["external_evaluations"] = [
        {
            "run_id": "external_opex_seed1",
            "method_id": "channel_aware_opex_inverse_anchor",
            "path": "external_opex/eval.json",
            "raw_schema": "channel_opex_v1",
            "evidence_label": "confirmation",
            "training_seed": 1,
            "development_selection": {
                "selection_schema": "channel-opex-development-selection-v1",
                "protocol_path": "channel_opex_dev_protocol.json",
                "protocol_sha256": "7" * 64,
                "selection_path": "channel_opex_dev/selection.json",
                "selection_sha256": "8" * 64,
                "selector_path": "select_channel_opex_dev.py",
                "selector_sha256": "9" * 64,
                "evaluator_sha256": implementation["evaluate_sha256"],
                "anchor": "inverse",
                "selected_step_size": 0.03,
                "selected_candidate_raw_sha256": "a" * 64,
                "candidate_count": 1,
                "provenance_revalidation_after_evaluations": False,
                "pre_evaluation_protocol_sha256": None,
                "pre_evaluation_selection_sha256": None,
                "pre_evaluation_selector_sha256": None,
                "prior_protocol_path": None,
                "prior_protocol_sha256": None,
                "prior_selection_path": None,
                "prior_selection_sha256": None,
                "prior_selector_path": None,
                "prior_selector_dependency_sha256": None,
                "selected_ca_opex_confirmation_outputs_read_before_selection": None,
                "selected_adapter_confirmation_outputs_read_before_selection": None,
                "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": None,
                "global_confirmation_blindness_claim": None,
            },
            "expected_artifacts": {
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_config_sha256": config_sha,
                "calibration_sha256": calibration_sha,
            },
            "expected_config": controller,
            "expected_implementation": implementation,
            "expected_cost": {
                "cost_scope": "adapted_arm_deployment_controller_only",
                "base_actor_rows_per_adapted_environment_step": 1,
                "q1_rows_per_adapted_step": 16,
                "q1_backward_calls_per_adapted_step": 2,
            },
            "evaluations": [
                {
                    "evaluation_id": "noisy",
                    "raw_arm": "paired_controller",
                    "evidence_label": "confirmation",
                    "expected_protocol": {
                        "environment": "Walker2d-v4",
                        "episode_count": 2,
                        "rollout_beta": 1.25,
                        "environment_seed_start": 10,
                        "action_noise_seed_start": 20,
                        "gradient_noise_seed_start": 30,
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
                    },
                }
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return raw_path


def test_paired_statistics_has_t_and_fixed_seed_bootstrap_intervals():
    first = paired_statistics(
        [1.0, 1.0, 1.0],
        normalized_scale=2.0,
        bootstrap_seed=9,
        bootstrap_samples=100,
    )
    second = paired_statistics(
        [1.0, 1.0, 1.0],
        normalized_scale=2.0,
        bootstrap_seed=9,
        bootstrap_samples=100,
    )
    assert first == second
    assert first["raw"]["mean_difference"] == 1.0
    assert first["raw"]["t_95ci_low"] == 1.0
    assert first["raw"]["paired_bootstrap_95ci_high"] == 1.0
    assert first["normalized"]["mean_difference"] == 2.0


def test_opex_development_metrics_are_recomputed_from_all_ten_paired_returns():
    env_seeds = tuple(range(28300, 28310))
    noise_seeds = tuple(range(38300, 38310))
    baseline = _arm(
        range(10), env_seeds=env_seeds, noise_seeds=noise_seeds
    )
    adapted = _arm(
        range(1, 11), env_seeds=env_seeds, noise_seeds=noise_seeds
    )
    for arm in (baseline, adapted):
        arm["gradient_noise_seeds"] = list(range(58300, 58310))
    raw = {
        "normalization": {
            "reference_min_score": REFERENCE_MIN,
            "reference_max_score": REFERENCE_MAX,
        },
        "arms": {"baseline_only": baseline, "adapted": adapted},
        "paired": {
            "adapted_minus_baseline_returns": [1.0] * 10,
            "return_difference_mean": 1.0,
        },
    }
    expected_protocol = {
        "environment_seed_start": 28300,
        "action_noise_seed_start": 38300,
        "gradient_noise_seed_start": 58300,
    }
    metrics = _recompute_opex_candidate_metrics(
        raw, expected_protocol=expected_protocol
    )
    assert metrics["paired_raw_return_difference_mean"] == 1.0
    assert metrics["adapted_normalized_score_mean"] == pytest.approx(4.5)

    raw["arms"]["adapted"]["gradient_noise_seeds"][-1] += 1
    with pytest.raises(AggregationError, match="arms are not seed-paired"):
        _recompute_opex_candidate_metrics(raw, expected_protocol=expected_protocol)
    raw["arms"]["adapted"]["gradient_noise_seeds"][-1] -= 1
    for arm in raw["arms"].values():
        arm["environment_seeds"][0] += 1
    with pytest.raises(AggregationError, match="differs from protocol"):
        _recompute_opex_candidate_metrics(raw, expected_protocol=expected_protocol)
    for arm in raw["arms"].values():
        arm["environment_seeds"][0] -= 1
    raw["arms"]["adapted"]["return_mean"] += 0.5
    with pytest.raises(AggregationError, match="reported mean is not raw-derived"):
        _recompute_opex_candidate_metrics(raw, expected_protocol=expected_protocol)


def test_report_derives_means_costs_audit_and_retained_failure(tmp_path):
    report = build_report(_manifest(tmp_path))
    assert report["aggregation_status"] == "complete"
    assert len(report["runs"]) == 3
    inverse = next(run for run in report["runs"] if run["run_id"] == "inverse")
    arm = inverse["evaluations"][0]["arms"]["adapted"]
    assert arm["return_mean"] == 3.0
    assert arm["normalized_score_mean"] == 2.0
    assert arm["episode_count"] == 2
    assert arm["total_environment_steps"] == 10
    assert arm["base_actor_forward_rows"] == 10
    assert arm["adapter_mlp_forward_rows"] == 10
    baseline_arm = inverse["evaluations"][0]["arms"]["baseline_only"]
    assert baseline_arm["base_actor_forward_rows"] == 10
    assert baseline_arm["adapter_mlp_forward_rows"] == 0
    assert inverse["evaluations"][0]["wall_time_scope"] == (
        "paired_evaluation_total_only_not_attributable_to_arm"
    )
    assert inverse["training_cost"]["q1_input_rows"] == 4096
    assert inverse["training_cost"]["preprocessing_seconds_this_invocation"] == 2.0
    assert inverse["training_cost"]["preprocessing_time_scope"] == (
        "latest_invocation_only_not_lifetime"
    )
    audit = inverse["audits"][0]
    assert audit["q1_gain"]["mean"] == 0.4
    assert audit["q1_gain"]["median"] == 0.35
    assert audit["q2_gain"]["median"] == 0.15
    assert audit["min_twin_gain"]["mean"] == 0.3
    assert audit["individual_critic_network_rows"] == 1024
    failed = next(run for run in report["runs"] if run["run_id"] == "failed")
    assert failed["observed_status"] == "failed_preupdate"
    assert failed["evaluations"] == []
    assert report["training_seed_evidence"][0]["statistical_unit"] == (
        "full_pipeline_training_seed"
    )
    assert all(
        item["per_training_seed_values"][0]["training_seed"] == 0
        for item in report["training_seed_evidence"]
    )
    assert all(
        "return_mean_per_arm"
        in item["per_training_seed_values"][0]["evaluations"][0]
        for item in report["training_seed_evidence"]
    )
    assert all(
        not item["can_estimate_training_seed_variability"]
        for item in report["training_seed_evidence"]
    )


def test_report_computes_strict_paired_comparisons_without_copying_returns(tmp_path):
    report = build_report(_manifest(tmp_path))
    within = report["paired_comparisons"][0]
    assert within["episode_pairs"] == 2
    assert within["raw"]["mean_difference"] == 1.0
    assert within["normalized"]["mean_difference"] == 1.0
    assert within["statistical_unit"] == (
        "paired_evaluation_episode_for_fixed_checkpoints"
    )
    encoded = json.dumps(report)
    assert '"returns"' not in encoded


def test_seed_mismatch_prevents_cross_run_pairing(tmp_path):
    manifest_path = _manifest(tmp_path)
    evaluation_path = tmp_path / "results" / "direct" / "eval.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["adapted"]["environment_seeds"] = [10, 12]
    evaluation["baseline_only"]["environment_seeds"] = [10, 12]
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(AggregationError, match="environment seed arrays differ"):
        build_report(manifest_path)


def test_cross_file_baseline_equivalence_is_bit_exact_and_fails_on_omission(
    tmp_path,
):
    manifest_path = _manifest(tmp_path)
    base_sha = "b" * 64
    for name in ("inverse", "direct"):
        config_path = tmp_path / "results" / name / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_checkpoint_sha256"] = base_sha
        config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_equivalence_checks"] = [
        {
            "check_id": "fixture-baseline",
            "evidence_label": "development",
            "expected_baseline_semantics": "fixture identity",
            "expected_rollout_beta": 1.25,
            "expected_base_checkpoint_sha256": base_sha,
            "endpoints": [
                {"run_id": "inverse", "evaluation_id": "dev", "arm": "baseline_only"},
                {"run_id": "direct", "evaluation_id": "dev", "arm": "baseline_only"},
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = build_report(manifest_path)
    assert report["baseline_equivalence_checks"][0]["status"] == (
        "verified_elementwise_identical"
    )

    evaluation_path = tmp_path / "results" / "direct" / "eval.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    arm = evaluation["baseline_only"]
    arm.update(
        {
            "returns": [1.5, 3.0],
            "return_mean": 2.25,
            "return_std": 0.75,
            "normalized_score_mean": 1.25,
            "normalized_score_std": 0.75,
        }
    )
    evaluation["paired"]["adapted_minus_baseline_returns"] = [0.5, 1.0]
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(AggregationError, match="return arrays differ"):
        build_report(manifest_path)


def test_manifest_cannot_contain_hand_entered_result_scores(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["return_mean"] = 999.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AggregationError, match="result values must come from raw files"):
        build_report(manifest_path)


def test_csv_markdown_and_json_outputs_are_deterministic(tmp_path):
    report = build_report(_manifest(tmp_path))
    csv_first = render_csv(report)
    markdown_first = render_markdown(report)
    csv_second = render_csv(report)
    markdown_second = render_markdown(report)
    assert csv_first == csv_second
    assert markdown_first == markdown_second
    parsed_rows = list(csv.DictReader(io.StringIO(csv_first)))
    assert {row["row_type"] for row in parsed_rows} == {
        "training_run",
        "evaluation_arm",
        "q_audit",
        "paired_comparison",
    }
    assert "Episode confidence intervals" in markdown_first
    audit_row = next(row for row in parsed_rows if row["row_type"] == "q_audit")
    assert audit_row["audit_q1_gain_median"] == "0.35"
    assert audit_row["audit_q2_gain_median"] == "0.15"
    assert "Q1 gain mean/median" in markdown_first
    first_paths = write_report(report, tmp_path / "out", "aggregate")
    first_bytes = {
        key: Path(path).read_bytes() for key, path in first_paths.items()
    }
    second_paths = write_report(report, tmp_path / "out", "aggregate")
    second_bytes = {
        key: Path(path).read_bytes() for key, path in second_paths.items()
    }
    assert first_bytes == second_bytes


def test_strict_manifest_validates_full_frozen_protocol(tmp_path):
    report = build_report(_strict_manifest(tmp_path))
    assert report["manifest_validation_mode"] == "strict_fail_closed_v2"
    run = report["runs"][0]
    assert run["fail_closed_validation"]["status"] == "verified"
    assert run["audits"][0]["audit_action_noise_beta"] == pytest.approx(1.25)
    assert run["audits"][0]["independent_from_training_seed"] is True
    cost = run["training_cost"]
    assert cost["invocation_count"] == 1
    assert cost["base_actor_precompute_rows_per_invocation"] == 10
    assert cost["base_actor_precompute_rows_lifetime"] == 10
    assert cost["q_scale_q1_forward_rows_per_invocation"] == 80
    assert cost["q_scale_q1_forward_rows_lifetime"] == 80
    assert cost["optimizer_q1_forward_rows_lifetime"] == 204800
    assert cost["optimizer_q1_backward_rows_lifetime"] == 102400
    assert cost["optimizer_adapter_mlp_forward_rows_lifetime"] == 12800
    assert cost["optimizer_adapter_mlp_backward_rows_lifetime"] == 12800
    assert cost["total_q1_forward_rows_lifetime"] == 204880
    assert cost["frozen_q_scale"] == 2.5
    assert cost["preprocessing_seconds_lifetime_available"] is True
    assert "CUDA" in cost["resume_scope"]
    assert report["paired_comparisons"][0]["episode_pairs"] == 2


def test_strict_external_td3bc_is_raw_linked_and_preserves_clean_noise_semantics(
    tmp_path,
):
    manifest_path = _strict_manifest(tmp_path)
    _append_external_td3bc(manifest_path)
    report = build_report(manifest_path)
    external = report["external_evaluations"][0]
    assert external["raw_schema"] == "evaluate_td3bc_v1"
    assert external["fail_closed_validation"]["raw_file_hash_was_predeclared"] is True
    assert external["cost"]["critic_forward_rows"] == 0
    noisy, clean = external["evaluations"]
    assert noisy["arms"]["external"]["return_mean"] == 3.0
    assert clean["arms"]["external"]["action_noise_seed_first"] is None
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert sum(row["row_type"] == "external_evaluation_arm" for row in rows) == 2
    assert "Strict external controls" in render_markdown(report)


def test_strict_external_opex_validates_paired_arms_and_exact_q_cost(tmp_path):
    manifest_path = _strict_manifest(tmp_path)
    raw_path = _append_external_opex(manifest_path)
    report = build_report(manifest_path)
    external = report["external_evaluations"][0]
    assert external["raw_schema"] == "channel_opex_v1"
    assert external["fail_closed_validation"]["raw_file_hash_was_predeclared"] is False
    assert external["cost"]["q1_forward_rows"] == 160
    assert external["cost"]["environment_steps"] == 10
    assert external["cost"]["paired_evaluation_environment_steps"] == 20
    assert external["cost"]["cost_scope"] == (
        "adapted_arm_deployment_controller_only"
    )
    assert set(external["evaluations"][0]["arms"]) == {
        "baseline_only",
        "adapted",
    }
    seed_evidence = report["external_training_seed_evidence"][0]
    assert seed_evidence["statistical_unit"] == "base_policy_checkpoint_seed"
    assert seed_evidence["can_estimate_training_seed_variability"] is False
    assert seed_evidence["evaluation_arm_descriptive_means"][0][
        "n_training_seeds"
    ] == 1
    assert seed_evidence["evaluation_arm_descriptive_means"][0][
        "statistical_unit"
    ] == "base_policy_checkpoint_seed"
    csv_rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert any(
        row["row_type"] == "external_training_seed_evidence"
        for row in csv_rows
    )
    assert "External-control checkpoint evidence" in render_markdown(report)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["cost"]["q1_forward_rows"] -= 1
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AggregationError, match="external OPEX cost q1_forward_rows mismatch"):
        build_report(manifest_path)


def test_strict_external_opex_rejects_raw_controller_and_seed_mutations(tmp_path):
    config_root = tmp_path / "config"
    manifest_path = _strict_manifest(config_root)
    raw_path = _append_external_opex(manifest_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["controller"]["delta_max"] = 0.5
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AggregationError, match="controller config mismatch"):
        build_report(manifest_path)

    seed_root = tmp_path / "seed"
    manifest_path = _strict_manifest(seed_root)
    raw_path = _append_external_opex(manifest_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    for arm in raw["arms"].values():
        arm["environment_seeds"][0] += 100
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(AggregationError, match="external environment seeds differ"):
        build_report(manifest_path)


def test_strict_manifest_rejects_config_and_linkage_mismatches(tmp_path):
    manifest_path = _strict_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["expected_config"]["alpha"] = 2.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AggregationError, match=r"expected_config\.alpha mismatch"):
        build_report(manifest_path)

    manifest_path = _strict_manifest(tmp_path / "linkage")
    evaluation_path = (
        tmp_path / "linkage" / "strict_results" / "inverse" / "eval.json"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["adapter_checkpoint_sha256"] = "f" * 64
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(AggregationError, match="evaluation adapter checkpoint SHA mismatch"):
        build_report(manifest_path)


def test_strict_training_cost_validates_rows_and_resume_scopes(tmp_path):
    manifest_path = _strict_manifest(tmp_path)
    summary_path = tmp_path / "strict_results" / "inverse" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"resumed": True, "resume_count": 1})
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = build_report(manifest_path)
    cost = report["runs"][0]["training_cost"]
    assert cost["invocation_count"] == 2
    assert cost["base_actor_precompute_rows_lifetime"] == 20
    assert cost["q_scale_q1_forward_rows_lifetime"] == 160
    assert cost["total_q1_forward_rows_lifetime"] == 204960
    assert cost["preprocessing_seconds_lifetime"] is None
    assert cost["preprocessing_seconds_lifetime_available"] is False

    summary["q1_input_rows"] -= 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(AggregationError, match="optimizer Q1 forward row count mismatch"):
        build_report(manifest_path)


def test_strict_manifest_rejects_audit_protocol_and_implementation_mismatch(tmp_path):
    manifest_path = _strict_manifest(tmp_path)
    audit_path = tmp_path / "strict_results" / "inverse" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["audit_channel"]["audit_action_noise_beta"] = 0.0
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(AggregationError, match="audit frozen action_noise_beta mismatch"):
        build_report(manifest_path)

    manifest_path = _strict_manifest(tmp_path / "implementation")
    audit_path = (
        tmp_path / "implementation" / "strict_results" / "inverse" / "audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["implementation"]["core_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(AggregationError, match="audit implementation core_sha256 mismatch"):
        build_report(manifest_path)


def test_same_base_comparison_rejects_different_base_checkpoint(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["comparisons"][1]["require_same_base_checkpoint"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for name, sha in (("inverse", "1" * 64), ("direct", "2" * 64)):
        config_path = tmp_path / "results" / name / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["base_checkpoint_sha256"] = sha
        config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(AggregationError, match="different base checkpoint SHA"):
        build_report(manifest_path)


def test_frozen_confirmation_template_predeclares_five_methods_and_no_results():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "inverse_residual_frozen_confirmation_manifest.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["schema_version"] == "inverse-residual-manifest-v2"
    validate_strict_manifest_definition(template, allow_placeholders=True)
    assert len(template["runs"]) == 10
    assert {run["evidence_label"] for run in template["runs"]} == {"confirmation"}
    assert {run["training_dir"].rsplit("seed", 1)[-1] for run in template["runs"]} == {
        "1",
        "10",
    }
    assert {run["method_id"] for run in template["runs"]} == {
        "inverse_sampled_main",
        "direct_sampled_main",
        "inverse_q1_at_channel_mean",
        "direct_sampled_wide",
        "direct_sampled_nominal_beta0",
    }
    assert all(
        {item["evaluation_id"] for item in run["evaluations"]}
        == (
            {"confirm_beta125", "confirm_clean"}
            if run["method_id"] in {"inverse_sampled_main", "direct_sampled_main"}
            else {"confirm_beta125"}
        )
        for run in template["runs"]
    )
    assert all(run["audits"][0]["audit_id"] == "heldout_k64" for run in template["runs"])
    assert all(
        run["audits"][0]["expected_protocol"]["action_noise_beta"] == 1.25
        for run in template["runs"]
    )
    expected_split = {
        "selected_observations": 100060,
        "train_indices_sha256": (
            "f90181e680de975cf43be71f477462513b6e9157453abb96302f9bf46adece5d"
        ),
        "audit_indices_sha256": (
            "e12718e0aa418e07284c04363d65494ce211b3c0d0878a0d3af54f6c3b4019b3"
        ),
        "audit_episode_units_sha256": (
            "f7cf0ed8e11d6308bd79c2f522fdb50f7b62c6c590427f83a4ec142f7d1b130c"
        ),
    }
    assert all(
        all(
            run["audits"][0]["expected_protocol"][field] == value
            for field, value in expected_split.items()
        )
        for run in template["runs"]
    )
    assert all(
        comparison["require_same_base_checkpoint"] is True
        or comparison.get("comparison_class") == "different_trained_policy"
        for comparison in template["comparisons"]
    )
    assert len(template["comparisons"]) == 56
    assert template["primary_comparison_order"] == list(
        PRIMARY_CONFIRMATION_COMPARISON_IDS
    )
    assert [
        comparison["comparison_id"]
        for comparison in template["comparisons"]
        if comparison["comparison_id"] in PRIMARY_CONFIRMATION_COMPARISON_IDS
    ] == list(PRIMARY_CONFIRMATION_COMPARISON_IDS)
    primary = {
        comparison["comparison_id"]: comparison
        for comparison in template["comparisons"]
        if comparison["comparison_id"] in PRIMARY_CONFIRMATION_COMPARISON_IDS
    }
    assert all(
        comparison["require_same_base_checkpoint"] is True
        and comparison["evidence_label"] == "confirmation"
        for comparison in primary.values()
    )
    assert len(template["external_evaluations"]) == 14
    assert {item["raw_schema"] for item in template["external_evaluations"]} == {
        "evaluate_td3bc_v1",
        "channel_opex_v1",
    }
    assert all(
        len(item["expected_artifacts"]["raw_file_sha256"]) == 64
        for item in template["external_evaluations"]
        if item["raw_schema"] == "evaluate_td3bc_v1"
    )
    opex = [
        item
        for item in template["external_evaluations"]
        if item["raw_schema"] == "channel_opex_v1"
    ]
    assert len(opex) == 6
    assert {item["method_id"] for item in opex} == {
        "original_structure_opex_t1",
        "channel_aware_opex_inverse_anchor",
        "channel_aware_opex_identity_anchor",
    }
    assert {
        item["method_id"]: item["expected_config"]["step_size"] for item in opex
    } == {
        "original_structure_opex_t1": 0.1,
        "channel_aware_opex_inverse_anchor": 0.1,
        "channel_aware_opex_identity_anchor": 0.3,
    }
    assert {item["expected_config"]["gradient_noise_seed"] for item in opex} == {
        69300,
    }
    assert {
        item["evaluations"][0]["expected_protocol"]["gradient_noise_seed_start"]
        for item in opex
    } == {69300}
    assert all(
        item["expected_cost"]["cost_scope"]
        == "adapted_arm_deployment_controller_only"
        for item in opex
    )
    assert all("raw_file_sha256" not in item["expected_artifacts"] for item in opex)
    selected_opex = [item for item in opex if item.get("development_selection")]
    assert len(selected_opex) == 4
    assert all(
        item["development_selection"]["selection_schema"]
        == "channel-opex-boundary-extension-selection-revalidated-v3"
        and item["development_selection"]["candidate_count"] == 10
        and item["development_selection"][
            "selected_ca_opex_confirmation_outputs_read_before_selection"
        ]
        is False
        and item["development_selection"][
            "selected_adapter_confirmation_outputs_read_before_selection"
        ]
        is False
        and item["development_selection"][
            "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed"
        ]
        is True
        and item["development_selection"]["global_confirmation_blindness_claim"]
        is False
        for item in selected_opex
    )
    assert len(template["baseline_equivalence_checks"]) == 4
    assert all(
        len(check["endpoints"]) >= 3
        for check in template["baseline_equivalence_checks"]
    )
    nominal = next(
        run
        for run in template["runs"]
        if run["method_id"] == "direct_sampled_nominal_beta0"
    )
    assert nominal["expected_config"]["model_or_calibration_beta"] == 0.0
    assert nominal["expected_artifacts"]["calibration_sha256"] is None
    wide = next(
        run for run in template["runs"] if run["method_id"] == "direct_sampled_wide"
    )
    assert wide["expected_config"]["residual_penalty"] / (
        wide["expected_config"]["delta_max"] ** 2
    ) == pytest.approx(0.16)
    encoded = json.dumps(template)
    for forbidden in (
        '"returns"',
        '"return_mean"',
        '"normalized_score_mean"',
    ):
        assert forbidden not in encoded
    with pytest.raises(AggregationError, match="unresolved freeze placeholders"):
        build_report(template_path)


def test_primary_comparison_contract_order_and_endpoint_mutations_fail_closed():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "inverse_residual_frozen_confirmation_manifest.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["primary_comparison_order"][0:2] = reversed(
        template["primary_comparison_order"][0:2]
    )
    with pytest.raises(AggregationError, match="primary_comparison_order"):
        validate_strict_manifest_definition(template, allow_placeholders=True)

    template = json.loads(template_path.read_text(encoding="utf-8"))
    comparison = next(
        item
        for item in template["comparisons"]
        if item["comparison_id"]
        == "seed1_beta125_ca_opex_inverse_minus_ca_opex_identity"
    )
    comparison["reference"]["arm"] = "baseline_only"
    with pytest.raises(AggregationError, match="frozen controller/arm contract"):
        validate_strict_manifest_definition(template, allow_placeholders=True)

    template = json.loads(template_path.read_text(encoding="utf-8"))
    opex = next(
        item
        for item in template["external_evaluations"]
        if item["raw_schema"] == "channel_opex_v1"
    )
    opex["expected_config"]["gradient_noise_seed"] = 69301
    with pytest.raises(AggregationError, match="share gradient_noise_seed=69300"):
        validate_strict_manifest_definition(template, allow_placeholders=True)


def test_primary_comparisons_summarize_two_checkpoints_without_episode_pooling():
    comparisons = []
    raw_means = (1.0, 3.0, 2.0, 4.0, 5.0, 7.0)
    for comparison_id, raw_mean in zip(
        PRIMARY_CONFIRMATION_COMPARISON_IDS, raw_means
    ):
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "training_seed": 1 if "seed1_" in comparison_id else 10,
                "episode_pairs": 50,
                "raw": {"mean_difference": raw_mean},
                "normalized": {"mean_difference": raw_mean / 10.0},
            }
        )
    evidence = _primary_comparison_training_seed_evidence(comparisons)
    assert [item["comparison_group_id"] for item in evidence] == [
        "ca_opex_inverse_minus_ca_opex_identity",
        "ca_opex_inverse_minus_original_opex",
        "ca_opex_inverse_minus_inverse_only",
    ]
    assert [
        item["mean_of_checkpoint_mean_differences"]["raw"] for item in evidence
    ] == [2.0, 3.0, 6.0]
    assert all(item["checkpoint_seeds"] == [1, 10] for item in evidence)
    assert all(item["n_checkpoint_pipelines"] == 2 for item in evidence)
    assert all(item["training_seed_ci"] is None for item in evidence)
    assert all(
        item["episode_returns_pooled_for_training_seed_inference"] is False
        for item in evidence
    )
    report = {"primary_cross_controller_training_seed_evidence": evidence}
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    primary_rows = [
        row
        for row in rows
        if row["row_type"] == "primary_comparison_training_seed_evidence"
    ]
    assert len(primary_rows) == 3
    assert primary_rows[0]["n_checkpoint_pipelines"] == "2"
    markdown = render_markdown(report)
    assert "Primary cross-controller checkpoint summary" in markdown
    assert "not pooled into n=100" in markdown
    assert "Primary?" in markdown
    assert "no multiple-comparison correction" in markdown


def test_strict_manifest_rejects_malformed_split_sha_before_file_access():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "inverse_residual_frozen_confirmation_manifest.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["runs"][0]["audits"][0]["expected_protocol"][
        "audit_episode_units_sha256"
    ] += "0"
    with pytest.raises(AggregationError, match="audit_episode_units_sha256 is not SHA256"):
        validate_strict_manifest_definition(template, allow_placeholders=True)


def test_final_strict_validation_rejects_placeholder_tokens_in_mapping_keys():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "inverse_residual_frozen_confirmation_manifest.template.json"
    )
    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    for field in list(manifest["frozen_implementation"]):
        manifest["frozen_implementation"][field] = "a" * 64
    for external in manifest["external_evaluations"]:
        for field in external["expected_implementation"]:
            external["expected_implementation"][field] = "a" * 64
    manifest["freeze_provenance"] = {"resolved___FILL_ILLEGAL_KEY__": {}}
    with pytest.raises(AggregationError, match="unresolved freeze placeholders"):
        validate_strict_manifest_definition(manifest, allow_placeholders=False)


def test_freeze_template_hashes_inputs_before_runs_and_refuses_overwrite(tmp_path):
    source_template = (
        Path(__file__).resolve().parents[1]
        / "inverse_residual_frozen_confirmation_manifest.template.json"
    )
    template = json.loads(source_template.read_text(encoding="utf-8"))
    template["results_root"] = "results"
    adapter_run_ids = {run["run_id"] for run in template["runs"]}
    opex_spec = next(
        item
        for item in template["external_evaluations"]
        if item["run_id"] == "external_ca_opex_inverse_seed1"
    )
    opex_spec["selection_note"] = "fixture selection fixed before confirmation"
    opex_spec["development_selection"].update(
        {
            "selection_schema": "channel-opex-development-selection-v1",
            "protocol_path": "channel_opex_dev_protocol.json",
            "selection_path": "channel_opex_dev/selection.json",
            "selector_path": "select_channel_opex_dev.py",
            "candidate_count": 1,
            "provenance_revalidation_after_evaluations": False,
            "pre_evaluation_protocol_sha256": None,
            "pre_evaluation_selection_sha256": None,
            "pre_evaluation_selector_sha256": None,
            "prior_protocol_path": None,
            "prior_protocol_sha256": None,
            "prior_selection_path": None,
            "prior_selection_sha256": None,
            "prior_selector_path": None,
            "prior_selector_dependency_sha256": None,
            "selected_ca_opex_confirmation_outputs_read_before_selection": None,
            "selected_adapter_confirmation_outputs_read_before_selection": None,
            "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": None,
            "global_confirmation_blindness_claim": None,
        }
    )
    template["external_evaluations"] = [opex_spec]
    template["comparisons"] = [
        comparison
        for comparison in template["comparisons"]
        if comparison["target"]["run_id"] in adapter_run_ids
        and comparison["reference"]["run_id"] in adapter_run_ids
    ]

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifacts = {}
    for name in ("dataset", "calibration", "base1", "base10"):
        path = artifact_dir / (
            "dataset.hdf5" if name == "dataset" else f"{name}.bin"
        )
        if name == "dataset":
            import h5py
            import numpy as np

            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "observations",
                    data=np.arange(60, dtype=np.float32).reshape(20, 3),
                )
                terminals = np.zeros(20, dtype=np.bool_)
                terminals[[4, 9, 14, 19]] = True
                handle.create_dataset("terminals", data=terminals)
        else:
            path.write_bytes(name.encode("ascii"))
        artifacts[name] = (
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    from train_inverse_residual_adapter import build_observation_split

    _, _, _, split = build_observation_split(
        artifacts["dataset"][0], None, 0.1, 424242
    )
    for run in template["runs"]:
        seed = run["expected_config"]["train_seed"]
        expected = run["expected_artifacts"]
        expected["dataset_sha256"] = artifacts["dataset"][1]
        expected["base_checkpoint_sha256"] = artifacts[f"base{seed}"][1]
        if expected["calibration_sha256"] is not None:
            expected["calibration_sha256"] = artifacts["calibration"][1]
        for audit in run["audits"]:
            audit_protocol = audit["expected_protocol"]
            audit_protocol["selected_observations"] = split[
                "audit_observation_count"
            ]
            for field in (
                "train_indices_sha256",
                "audit_indices_sha256",
                "audit_episode_units_sha256",
            ):
                audit_protocol[field] = split[field]
    opex_spec["expected_artifacts"].update(
        {
            "checkpoint_sha256": artifacts["base1"][1],
            "checkpoint_config_sha256": "f" * 64,
            "calibration_sha256": artifacts["calibration"][1],
        }
    )
    implementation_root = tmp_path / "implementation"
    implementation_root.mkdir()
    implementation_names = (
        "train_inverse_residual_adapter.py",
        "inverse_residual_core.py",
        "evaluation_controls.py",
        "td3bc_core.py",
        "train_iql.py",
        "evaluate_inverse_residual_adapter.py",
        "audit_inverse_residual_adapter.py",
        "evaluate_channel_opex.py",
        "train_td3bc.py",
        "select_channel_opex_dev.py",
    )
    for filename in implementation_names:
        (implementation_root / filename).write_text(filename, encoding="utf-8")
    protocol_path = implementation_root / "channel_opex_dev_protocol.json"
    protocol_path.write_text('{"fixture":"protocol"}\n', encoding="utf-8")
    development_dir = tmp_path / "results" / "channel_opex_dev"
    development_dir.mkdir(parents=True)
    candidate_path = development_dir / "selected_inverse.json"
    candidate_path.write_text('{"fixture":"candidate"}\n', encoding="utf-8")
    evaluator_sha = hashlib.sha256(
        (implementation_root / "evaluate_channel_opex.py").read_bytes()
    ).hexdigest()
    selector_sha = hashlib.sha256(
        (implementation_root / "select_channel_opex_dev.py").read_bytes()
    ).hexdigest()
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    selection_path = development_dir / "selection.json"
    selection_payload = {
        "schema_version": "channel-opex-development-selection-v1",
        "status": "complete",
        "evidence_label": "development",
        "protocol_sha256": protocol_sha,
        "selection_script_sha256": selector_sha,
        "evaluator_sha256": evaluator_sha,
        "confirmation_data_read": False,
        "selected": {
            "inverse": {
                "step_size": 0.1,
                "raw_sha256": candidate_sha,
                "path": str(candidate_path.resolve()),
            }
        },
    }
    selection_path.write_text(json.dumps(selection_payload), encoding="utf-8")
    opex_spec["development_selection"].update(
        {
            "protocol_sha256": protocol_sha,
            "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "selector_sha256": selector_sha,
            "evaluator_sha256": evaluator_sha,
            "selected_candidate_raw_sha256": candidate_sha,
        }
    )
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    frozen_path = tmp_path / "frozen.json"
    result = freeze_strict_manifest_template(
        template_path,
        frozen_path,
        implementation_root=implementation_root,
        artifact_paths=[item[0] for item in artifacts.values()],
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["status"] == "frozen"
    assert not _placeholder_strings(frozen)
    assert "__FILL_" not in json.dumps(frozen, sort_keys=True)
    assert set(
        frozen["freeze_provenance"][
            "resolved_implementation_sources_by_filename"
        ]
    ) == {
        "train_inverse_residual_adapter.py",
        "inverse_residual_core.py",
        "evaluation_controls.py",
        "td3bc_core.py",
        "train_iql.py",
        "evaluate_inverse_residual_adapter.py",
        "audit_inverse_residual_adapter.py",
        "evaluate_channel_opex.py",
        "train_td3bc.py",
    }
    assert frozen["freeze_provenance"]["results_root_verified_absent"].endswith(
        "results"
    )
    verified_split = frozen["freeze_provenance"]["verified_observation_splits"][0]
    assert verified_split["dataset_path"] == str(artifacts["dataset"][0])
    assert verified_split["dataset_sha256"] == artifacts["dataset"][1]
    assert verified_split["max_observations"] is None
    assert verified_split["audit_fraction"] == 0.1
    assert verified_split["split_seed"] == 424242
    for field in (
        "considered_observation_count",
        "train_observation_count",
        "audit_observation_count",
        "requested_audit_fraction",
        "realized_audit_fraction",
        "audit_episode_unit_count",
        "train_indices_sha256",
        "audit_indices_sha256",
        "audit_episode_units_sha256",
    ):
        assert verified_split[field] == split[field]
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze_strict_manifest_template(
            template_path,
            frozen_path,
            implementation_root=implementation_root,
            artifact_paths=[item[0] for item in artifacts.values()],
        )
    first_training_dir = tmp_path / "results" / template["runs"][0]["training_dir"]
    first_training_dir.mkdir(parents=True)
    with pytest.raises(AggregationError, match="after confirmation output directories exist"):
        freeze_strict_manifest_template(
            template_path,
            tmp_path / "too_late.json",
            implementation_root=implementation_root,
            artifact_paths=[item[0] for item in artifacts.values()],
        )


def _placeholder_strings(value):
    if isinstance(value, dict):
        keys = [
            key
            for key in value
            if isinstance(key, str) and "__FILL_" in key
        ]
        return keys + [
            item for child in value.values() for item in _placeholder_strings(child)
        ]
    if isinstance(value, list):
        return [item for child in value for item in _placeholder_strings(child)]
    return [value] if isinstance(value, str) and "__FILL_" in value else []
