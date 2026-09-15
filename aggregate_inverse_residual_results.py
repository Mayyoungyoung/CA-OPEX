#!/usr/bin/env python3
"""Deterministically aggregate inverse-residual adapter evidence.

The manifest supplies only provenance and evidence labels.  Every score, cost,
diagnostic, and confidence interval is derived from the referenced raw training,
evaluation, or audit records.  Paired episode intervals condition on a fixed
trained checkpoint and are never presented as independent-training-seed
uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from scipy.stats import t as student_t


SCHEMA_VERSION = "inverse-residual-aggregate-v2"
LEGACY_MANIFEST_SCHEMA_VERSION = "inverse-residual-manifest-v1"
STRICT_MANIFEST_SCHEMA_VERSION = "inverse-residual-manifest-v2"
MANIFEST_SCHEMA_VERSION = STRICT_MANIFEST_SCHEMA_VERSION
SUPPORTED_MANIFEST_SCHEMAS = {
    LEGACY_MANIFEST_SCHEMA_VERSION,
    STRICT_MANIFEST_SCHEMA_VERSION,
}
EVIDENCE_LABELS = {"development", "confirmation", "implementation_smoke"}
EXPECTED_STATUSES = {"complete", "failed_preupdate"}
ADAPTER_ARMS = {"baseline_only", "adapted"}
ARMS = ADAPTER_ARMS | {"external"}
FORBIDDEN_MANIFEST_RESULT_KEYS = {
    "returns",
    "return_mean",
    "return_std",
    "normalized_score_mean",
    "normalized_score_std",
    "scores",
    "metrics",
    "paired_differences",
}
STRICT_CONFIG_FIELDS = (
    "baseline_transform",
    "value_estimator",
    "alpha",
    "residual_penalty",
    "delta_max",
    "execution_noise_samples",
    "updates",
    "learning_rate",
    "batch_size",
    "split_seed",
    "audit_fraction",
    "train_seed",
    "channel_seed",
    "model_or_calibration_beta",
    "calibration_mode",
    "hidden_dim",
    "depth",
    "q_scale_epsilon",
    "precompute_batch_size",
    "scale_calibration_observations",
    "max_observations",
    "base_action_pairing",
)
TRAIN_IMPLEMENTATION_FIELDS = (
    "train_sha256",
    "core_sha256",
    "evaluation_controls_sha256",
    "td3bc_core_sha256",
    "train_iql_sha256",
)
FROZEN_IMPLEMENTATION_FIELDS = TRAIN_IMPLEMENTATION_FIELDS + (
    "evaluate_sha256",
    "audit_sha256",
)
STRICT_ARTIFACT_FIELDS = (
    "dataset_sha256",
    "base_checkpoint_sha256",
    "calibration_sha256",
    "checkpoint_format",
)
PLACEHOLDER_PREFIX = "__FILL_"
IMPLEMENTATION_SOURCE_FILES = {
    "train_sha256": "train_inverse_residual_adapter.py",
    "core_sha256": "inverse_residual_core.py",
    "evaluation_controls_sha256": "evaluation_controls.py",
    "td3bc_core_sha256": "td3bc_core.py",
    "train_iql_sha256": "train_iql.py",
    "evaluate_sha256": "evaluate_inverse_residual_adapter.py",
    "audit_sha256": "audit_inverse_residual_adapter.py",
}
FREEZE_PLACEHOLDER_SOURCE_FILES = {
    "__FILL_FINAL_TRAIN_SHA256__": "train_inverse_residual_adapter.py",
    "__FILL_FINAL_CORE_SHA256__": "inverse_residual_core.py",
    "__FILL_FINAL_EVALUATION_CONTROLS_SHA256__": "evaluation_controls.py",
    "__FILL_FINAL_TD3BC_CORE_SHA256__": "td3bc_core.py",
    "__FILL_FINAL_TRAIN_IQL_SHA256__": "train_iql.py",
    "__FILL_FINAL_EVALUATE_SHA256__": "evaluate_inverse_residual_adapter.py",
    "__FILL_FINAL_AUDIT_SHA256__": "audit_inverse_residual_adapter.py",
    "__FILL_FINAL_OPEX_EVALUATE_SHA256__": "evaluate_channel_opex.py",
    "__FILL_FINAL_TRAIN_TD3BC_SHA256__": "train_td3bc.py",
}
EXTERNAL_RAW_SCHEMAS = {"evaluate_td3bc_v1", "channel_opex_v1"}
TD3BC_EXTERNAL_CONFIG_FIELDS = (
    "command_transform",
    "command_scale",
    "command_transform_beta",
    "checkpoint_variant",
    "checkpoint_action_pairing",
    "checkpoint_train_seed",
    "checkpoint_updates",
)
TD3BC_EXTERNAL_IMPLEMENTATION_FIELDS = (
    "evaluate_td3bc_sha256",
    "evaluation_controls_sha256",
    "train_td3bc_sha256",
)
OPEX_EXTERNAL_CONFIG_FIELDS = (
    "baseline_transform",
    "step_size",
    "gradient_steps",
    "K",
    "execution_noise_samples",
    "model_beta",
    "calibration_mode",
    "gradient_noise_seed",
    "q_reducer",
    "delta_max",
    "critic",
    "gradient_objective",
    "gradient_steps_per_action",
    "actor_parameter_updates",
    "critic_parameter_updates",
)
OPEX_EXTERNAL_IMPLEMENTATION_FIELDS = (
    "evaluate_sha256",
    "inverse_residual_core_sha256",
    "td3bc_core_sha256",
    "train_td3bc_sha256",
    "evaluation_controls_sha256",
    "train_inverse_residual_adapter_sha256",
)
OPEX_EXTERNAL_PROTOCOL_METADATA_FIELDS = (
    "paired_environment_and_action_noise_seeds",
    "environment_rng_namespace",
    "actuator_noise_rng_namespace",
    "gradient_noise_rng_namespace",
    "gradient_noise_sampling_frequency",
    "gradient_noise_antithetic",
    "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling",
    "selection_rule",
)
PRIMARY_CONFIRMATION_COMPARISON_IDS = (
    "seed1_beta125_ca_opex_inverse_minus_ca_opex_identity",
    "seed10_beta125_ca_opex_inverse_minus_ca_opex_identity",
    "seed1_beta125_ca_opex_inverse_minus_original_opex",
    "seed10_beta125_ca_opex_inverse_minus_original_opex",
    "seed1_beta125_ca_opex_inverse_effect",
    "seed10_beta125_ca_opex_inverse_effect",
)
PRIMARY_CONFIRMATION_COMPARISON_GROUPS = (
    (
        "ca_opex_inverse_minus_ca_opex_identity",
        PRIMARY_CONFIRMATION_COMPARISON_IDS[0:2],
    ),
    (
        "ca_opex_inverse_minus_original_opex",
        PRIMARY_CONFIRMATION_COMPARISON_IDS[2:4],
    ),
    (
        "ca_opex_inverse_minus_inverse_only",
        PRIMARY_CONFIRMATION_COMPARISON_IDS[4:6],
    ),
)
OPEX_SELECTION_FIELDS = (
    "selection_schema",
    "protocol_path",
    "protocol_sha256",
    "selection_path",
    "selection_sha256",
    "selector_path",
    "selector_sha256",
    "evaluator_sha256",
    "anchor",
    "selected_step_size",
    "selected_candidate_raw_sha256",
    "candidate_count",
    "provenance_revalidation_after_evaluations",
    "pre_evaluation_protocol_sha256",
    "pre_evaluation_selection_sha256",
    "pre_evaluation_selector_sha256",
    "prior_protocol_path",
    "prior_protocol_sha256",
    "prior_selection_path",
    "prior_selection_sha256",
    "prior_selector_path",
    "prior_selector_dependency_sha256",
    "selected_ca_opex_confirmation_outputs_read_before_selection",
    "selected_adapter_confirmation_outputs_read_before_selection",
    "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed",
    "global_confirmation_blindness_claim",
)


class AggregationError(ValueError):
    """Raised when provenance or comparison invariants do not hold."""


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _placeholder_locations(value: Any, location: str = "manifest") -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and PLACEHOLDER_PREFIX in key:
                found.append(f"{location}.<key:{key}>")
            found.extend(_placeholder_locations(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_locations(child, f"{location}[{index}]"))
    elif isinstance(value, str) and PLACEHOLDER_PREFIX in value:
        found.append(location)
    return found


def _replace_freeze_placeholders(
    value: Any, replacements: Mapping[str, str]
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _replace_freeze_placeholders(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_freeze_placeholders(child, replacements) for child in value
        ]
    if isinstance(value, str) and value in replacements:
        return replacements[value]
    return value


def _require_exact_keys(
    value: Any,
    required: Iterable[str],
    *,
    location: str,
    allow_extra: Iterable[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{location} must be an object")
    required_set = set(required)
    allowed = required_set | set(allow_extra)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing:
        raise AggregationError(f"{location} lacks required keys: {missing}")
    if extra:
        raise AggregationError(f"{location} has undeclared keys: {extra}")
    return value


def validate_strict_manifest_definition(
    manifest: Mapping[str, Any], *, allow_placeholders: bool
) -> None:
    """Validate the fail-closed v2 declaration before reading result files."""

    if manifest.get("schema_version") != STRICT_MANIFEST_SCHEMA_VERSION:
        raise AggregationError("strict validation requires manifest schema v2")
    if not allow_placeholders:
        unresolved = _placeholder_locations(manifest)
        if unresolved:
            preview = ", ".join(unresolved[:5])
            raise AggregationError(
                f"manifest has unresolved freeze placeholders ({preview}); "
                "fill them before aggregation"
            )
    implementation = _require_exact_keys(
        manifest.get("frozen_implementation"),
        FROZEN_IMPLEMENTATION_FIELDS,
        location="manifest.frozen_implementation",
        allow_extra=(
            "evaluate_sha256",
            "audit_sha256",
            "evaluate_td3bc_sha256",
            "evaluation_controls_external_sha256",
            "train_td3bc_sha256",
        ),
    )
    for key, value in implementation.items():
        if allow_placeholders and isinstance(value, str) and value.startswith(
            PLACEHOLDER_PREFIX
        ):
            continue
        if not _is_sha256(value):
            raise AggregationError(f"frozen_implementation.{key} is not SHA256")
    run_specs = manifest.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise AggregationError("strict manifest runs must be non-empty")
    seen_run_ids: set[str] = set()
    seen_training_dirs: set[str] = set()
    method_to_config: Dict[str, str] = {}
    config_to_method: Dict[str, str] = {}
    for index, spec in enumerate(run_specs):
        location = f"manifest.runs[{index}]"
        if not isinstance(spec, Mapping):
            raise AggregationError(f"{location} must be an object")
        for field in ("run_id", "method_id", "training_dir"):
            if not isinstance(spec.get(field), str) or not spec[field]:
                raise AggregationError(f"{location}.{field} must be a non-empty string")
        if spec["run_id"] in seen_run_ids:
            raise AggregationError(f"duplicate strict run_id {spec['run_id']!r}")
        seen_run_ids.add(spec["run_id"])
        if spec["training_dir"] in seen_training_dirs:
            raise AggregationError(
                f"duplicate strict training_dir {spec['training_dir']!r}"
            )
        seen_training_dirs.add(spec["training_dir"])
        if spec.get("evidence_label") != "confirmation":
            raise AggregationError(f"{location} must be labeled confirmation")
        if spec.get("expected_status") != "complete":
            raise AggregationError(f"{location} must predeclare expected_status=complete")
        config = _require_exact_keys(
            spec.get("expected_config"),
            STRICT_CONFIG_FIELDS,
            location=f"{location}.expected_config",
            allow_extra=("extensions",),
        )
        extensions = config.get("extensions", {})
        if not isinstance(extensions, Mapping):
            raise AggregationError(f"{location}.expected_config.extensions must be an object")
        method_config_sha = sha256_bytes(
            canonical_json(_canonical_method_config(config)).encode("utf-8")
        )
        prior_config = method_to_config.setdefault(spec["method_id"], method_config_sha)
        if prior_config != method_config_sha:
            raise AggregationError(
                f"strict method_id {spec['method_id']!r} maps to multiple configs"
            )
        prior_method = config_to_method.setdefault(method_config_sha, spec["method_id"])
        if prior_method != spec["method_id"]:
            raise AggregationError(
                f"strict identical configs use method_ids {prior_method!r} and "
                f"{spec['method_id']!r}"
            )
        artifacts = _require_exact_keys(
            spec.get("expected_artifacts"),
            STRICT_ARTIFACT_FIELDS,
            location=f"{location}.expected_artifacts",
        )
        for key in ("dataset_sha256", "base_checkpoint_sha256"):
            value = artifacts[key]
            if allow_placeholders and isinstance(value, str) and value.startswith(
                PLACEHOLDER_PREFIX
            ):
                continue
            if not _is_sha256(value):
                raise AggregationError(f"{location}.expected_artifacts.{key} is not SHA256")
        calibration_sha = artifacts["calibration_sha256"]
        if calibration_sha is not None and not (
            allow_placeholders
            and isinstance(calibration_sha, str)
            and calibration_sha.startswith(PLACEHOLDER_PREFIX)
        ) and not _is_sha256(calibration_sha):
            raise AggregationError(
                f"{location}.expected_artifacts.calibration_sha256 is not SHA256/null"
            )
        if not isinstance(artifacts["checkpoint_format"], str):
            raise AggregationError(f"{location}.checkpoint_format must be a string")
        if config["calibration_mode"] == "pair_calibration" and calibration_sha is None:
            raise AggregationError(f"{location} pair_calibration requires calibration SHA")
        if config["calibration_mode"] == "known_beta_without_pair_calibration" and calibration_sha is not None:
            raise AggregationError(f"{location} nominal known-beta run requires null calibration SHA")
        for eval_index, evaluation in enumerate(spec.get("evaluations", [])):
            if not isinstance(evaluation, Mapping):
                raise AggregationError(
                    f"{location}.evaluations[{eval_index}] must be an object"
                )
            if evaluation.get("evidence_label") != "confirmation":
                raise AggregationError(
                    f"{location}.evaluations[{eval_index}] must be confirmation"
                )
            protocol = _require_exact_keys(
                evaluation.get("expected_protocol") if isinstance(evaluation, Mapping) else None,
                (
                    "environment",
                    "episode_count",
                    "rollout_beta",
                    "environment_seed_start",
                    "action_noise_seed_start",
                ),
                location=f"{location}.evaluations[{eval_index}].expected_protocol",
                allow_extra=("environment_seeds", "action_noise_seeds"),
            )
            if int(protocol["episode_count"]) <= 0:
                raise AggregationError("expected evaluation episode_count must be positive")
            for seeds_key in ("environment_seeds", "action_noise_seeds"):
                seeds = protocol.get(seeds_key)
                if seeds is not None and (
                    not isinstance(seeds, list)
                    or len(seeds) != int(protocol["episode_count"])
                    or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
                ):
                    raise AggregationError(
                        f"{location}.evaluations[{eval_index}].{seeds_key} invalid"
                    )
        for audit_index, audit in enumerate(spec.get("audits", [])):
            if not isinstance(audit, Mapping):
                raise AggregationError(
                    f"{location}.audits[{audit_index}] must be an object"
                )
            if audit.get("evidence_label") != "confirmation":
                raise AggregationError(
                    f"{location}.audits[{audit_index}] must be confirmation"
                )
            audit_protocol = _require_exact_keys(
                audit.get("expected_protocol") if isinstance(audit, Mapping) else None,
                (
                    "noise_samples",
                    "noise_seed",
                    "action_noise_beta",
                    "selected_observations",
                    "train_indices_sha256",
                    "audit_indices_sha256",
                    "audit_episode_units_sha256",
                ),
                location=f"{location}.audits[{audit_index}].expected_protocol",
            )
            for field in (
                "train_indices_sha256",
                "audit_indices_sha256",
                "audit_episode_units_sha256",
            ):
                if not _is_sha256(audit_protocol[field]):
                    raise AggregationError(
                        f"{location}.audits[{audit_index}].{field} is not SHA256"
                    )
    external_specs = manifest.get("external_evaluations", [])
    if not isinstance(external_specs, list):
        raise AggregationError("manifest.external_evaluations must be a list")
    for index, spec in enumerate(external_specs):
        location = f"manifest.external_evaluations[{index}]"
        if not isinstance(spec, Mapping):
            raise AggregationError(f"{location} must be an object")
        for field in ("run_id", "method_id", "path", "raw_schema"):
            if not isinstance(spec.get(field), str) or not spec[field]:
                raise AggregationError(f"{location}.{field} must be a non-empty string")
        if spec["run_id"] in seen_run_ids:
            raise AggregationError(f"duplicate external run_id {spec['run_id']!r}")
        seen_run_ids.add(spec["run_id"])
        if spec.get("evidence_label") != "confirmation":
            raise AggregationError(f"{location} must be labeled confirmation")
        if isinstance(spec.get("training_seed"), bool) or not isinstance(
            spec.get("training_seed"), int
        ):
            raise AggregationError(f"{location}.training_seed must be an integer")
        raw_schema = spec["raw_schema"]
        if raw_schema not in EXTERNAL_RAW_SCHEMAS:
            raise AggregationError(f"{location} has unsupported raw_schema")
        artifact_fields = (
            (
                "raw_file_sha256",
                "checkpoint_sha256",
                "checkpoint_config_sha256",
            )
            if raw_schema == "evaluate_td3bc_v1"
            else (
                "checkpoint_sha256",
                "checkpoint_config_sha256",
                "calibration_sha256",
            )
        )
        artifacts = _require_exact_keys(
            spec.get("expected_artifacts"),
            artifact_fields,
            location=f"{location}.expected_artifacts",
        )
        for field, value in artifacts.items():
            if field == "calibration_sha256" and value is None:
                continue
            if allow_placeholders and isinstance(value, str) and value.startswith(
                PLACEHOLDER_PREFIX
            ):
                continue
            if not _is_sha256(value):
                raise AggregationError(
                    f"{location}.expected_artifacts.{field} is not SHA256"
                )
        config_fields = (
            TD3BC_EXTERNAL_CONFIG_FIELDS
            if raw_schema == "evaluate_td3bc_v1"
            else OPEX_EXTERNAL_CONFIG_FIELDS
        )
        config = _require_exact_keys(
            spec.get("expected_config"),
            config_fields,
            location=f"{location}.expected_config",
        )
        if raw_schema == "channel_opex_v1":
            calibration_sha = artifacts["calibration_sha256"]
            if config["calibration_mode"] == (
                "censored_uniform_plus_clip_pair_calibration"
            ):
                if calibration_sha is None:
                    raise AggregationError(
                        f"{location} calibrated OPEX requires calibration SHA"
                    )
            elif config["calibration_mode"] == (
                "cli_known_beta_without_pair_calibration"
            ):
                if calibration_sha is not None:
                    raise AggregationError(
                        f"{location} nominal OPEX requires null calibration SHA"
                    )
            else:
                raise AggregationError(
                    f"{location}.expected_config.calibration_mode unsupported"
                )
            original_structure = (
                config["baseline_transform"] == "identity"
                and float(config["model_beta"]) == 0.0
                and int(config["K"]) == 1
                and int(config["gradient_steps"]) == 1
                and float(config["delta_max"]) >= 2.0
            )
            expected_method_id = (
                "original_structure_opex_t1"
                if original_structure
                else f"channel_aware_opex_{config['baseline_transform']}_anchor"
            )
            if spec["method_id"] != expected_method_id:
                raise AggregationError(
                    f"{location}.method_id differs from its declared OPEX controller"
                )
            selection = spec.get("development_selection")
            if original_structure:
                if selection is not None:
                    raise AggregationError(
                        f"{location} original-structure OPEX was not dev-selected"
                    )
            else:
                selection = _require_exact_keys(
                    selection,
                    OPEX_SELECTION_FIELDS,
                    location=f"{location}.development_selection",
                )
                for field in (
                    "protocol_sha256",
                    "selection_sha256",
                    "selector_sha256",
                    "evaluator_sha256",
                    "selected_candidate_raw_sha256",
                ):
                    if not _is_sha256(selection[field]):
                        raise AggregationError(
                            f"{location}.development_selection.{field} is not SHA256"
                        )
                for field in ("protocol_path", "selection_path", "selector_path"):
                    if not isinstance(selection[field], str) or not selection[field]:
                        raise AggregationError(
                            f"{location}.development_selection.{field} invalid"
                        )
                selection_schema = selection["selection_schema"]
                if selection_schema == (
                    "channel-opex-boundary-extension-selection-revalidated-v3"
                ):
                    if selection["provenance_revalidation_after_evaluations"] is not True:
                        raise AggregationError(
                            f"{location} must disclose post-evaluation provenance repair"
                        )
                    if int(selection["candidate_count"]) != 10:
                        raise AggregationError(
                            f"{location} boundary union must retain all ten candidates"
                        )
                    for field in (
                        "pre_evaluation_protocol_sha256",
                        "pre_evaluation_selection_sha256",
                        "pre_evaluation_selector_sha256",
                        "prior_protocol_sha256",
                        "prior_selection_sha256",
                        "prior_selector_dependency_sha256",
                    ):
                        if not _is_sha256(selection[field]):
                            raise AggregationError(
                                f"{location}.development_selection.{field} is not SHA256"
                            )
                    for field in (
                        "prior_protocol_path",
                        "prior_selection_path",
                        "prior_selector_path",
                    ):
                        if not isinstance(selection[field], str) or not selection[field]:
                            raise AggregationError(
                                f"{location}.development_selection.{field} invalid"
                            )
                    chronology = {
                        "selected_ca_opex_confirmation_outputs_read_before_selection": False,
                        "selected_adapter_confirmation_outputs_read_before_selection": False,
                        "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": True,
                        "global_confirmation_blindness_claim": False,
                    }
                    for field, expected in chronology.items():
                        if selection[field] is not expected:
                            raise AggregationError(
                                f"{location}.development_selection.{field} chronology mismatch"
                            )
                elif selection_schema == "channel-opex-development-selection-v1":
                    if selection["provenance_revalidation_after_evaluations"] is not False:
                        raise AggregationError(
                            f"{location} legacy development selection chronology invalid"
                        )
                    if int(selection["candidate_count"]) <= 0:
                        raise AggregationError(
                            f"{location} development candidate_count invalid"
                        )
                    for field in (
                        "pre_evaluation_protocol_sha256",
                        "pre_evaluation_selection_sha256",
                        "pre_evaluation_selector_sha256",
                        "prior_protocol_path",
                        "prior_protocol_sha256",
                        "prior_selection_path",
                        "prior_selection_sha256",
                        "prior_selector_path",
                        "prior_selector_dependency_sha256",
                        "selected_ca_opex_confirmation_outputs_read_before_selection",
                        "selected_adapter_confirmation_outputs_read_before_selection",
                        "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed",
                        "global_confirmation_blindness_claim",
                    ):
                        if selection[field] is not None:
                            raise AggregationError(
                                f"{location} legacy selection must not invent {field}"
                            )
                else:
                    raise AggregationError(
                        f"{location}.development_selection.selection_schema unsupported"
                    )
                if selection["anchor"] != config["baseline_transform"]:
                    raise AggregationError(
                        f"{location} development-selection anchor mismatch"
                    )
                if not _close(
                    float(selection["selected_step_size"]),
                    float(config["step_size"]),
                ):
                    raise AggregationError(
                        f"{location} development-selected step size mismatch"
                    )
                expected_evaluator = spec.get("expected_implementation", {}).get(
                    "evaluate_sha256"
                )
                if not (
                    allow_placeholders
                    and isinstance(expected_evaluator, str)
                    and expected_evaluator.startswith(PLACEHOLDER_PREFIX)
                ) and selection["evaluator_sha256"] != expected_evaluator:
                    raise AggregationError(
                        f"{location} selection/evaluation implementation mismatch"
                    )
        implementation_fields = (
            TD3BC_EXTERNAL_IMPLEMENTATION_FIELDS
            if raw_schema == "evaluate_td3bc_v1"
            else OPEX_EXTERNAL_IMPLEMENTATION_FIELDS
        )
        expected_implementation = _require_exact_keys(
            spec.get("expected_implementation"),
            implementation_fields,
            location=f"{location}.expected_implementation",
        )
        for field, value in expected_implementation.items():
            if allow_placeholders and isinstance(value, str) and value.startswith(
                PLACEHOLDER_PREFIX
            ):
                continue
            if not _is_sha256(value):
                raise AggregationError(
                    f"{location}.expected_implementation.{field} is not SHA256"
                )
        seed_only_config_fields = (
            {"checkpoint_train_seed"}
            if raw_schema == "evaluate_td3bc_v1"
            else {"gradient_noise_seed"}
        )
        method_config = {
            "raw_schema": raw_schema,
            **{
                key: value
                for key, value in config.items()
                if key not in seed_only_config_fields
            },
        }
        method_config_sha = sha256_bytes(canonical_json(method_config).encode("utf-8"))
        prior_config = method_to_config.setdefault(spec["method_id"], method_config_sha)
        if prior_config != method_config_sha:
            raise AggregationError(
                f"external method_id {spec['method_id']!r} maps to multiple configs"
            )
        prior_method = config_to_method.setdefault(method_config_sha, spec["method_id"])
        if prior_method != spec["method_id"]:
            raise AggregationError(
                f"external identical configs use method_ids {prior_method!r} and "
                f"{spec['method_id']!r}"
            )
        evaluations = spec.get("evaluations")
        if not isinstance(evaluations, list) or not evaluations:
            raise AggregationError(f"{location}.evaluations must be non-empty")
        seen_external_evaluation_ids = set()
        for eval_index, evaluation in enumerate(evaluations):
            eval_location = f"{location}.evaluations[{eval_index}]"
            if not isinstance(evaluation, Mapping):
                raise AggregationError(f"{eval_location} must be an object")
            evaluation_id = evaluation.get("evaluation_id")
            if not isinstance(evaluation_id, str) or not evaluation_id:
                raise AggregationError(f"{eval_location}.evaluation_id invalid")
            if evaluation_id in seen_external_evaluation_ids:
                raise AggregationError(f"duplicate {eval_location}.evaluation_id")
            seen_external_evaluation_ids.add(evaluation_id)
            if evaluation.get("evidence_label") != "confirmation":
                raise AggregationError(f"{eval_location} must be confirmation")
            expected_raw_arm = (
                {"clean", "persistent_action_noise"}
                if raw_schema == "evaluate_td3bc_v1"
                else {"paired_controller"}
            )
            if evaluation.get("raw_arm") not in expected_raw_arm:
                raise AggregationError(f"{eval_location}.raw_arm invalid")
            protocol = _require_exact_keys(
                evaluation.get("expected_protocol"),
                (
                    "environment",
                    "episode_count",
                    "rollout_beta",
                    "environment_seed_start",
                    "action_noise_seed_start",
                )
                + (
                    OPEX_EXTERNAL_PROTOCOL_METADATA_FIELDS
                    if raw_schema == "channel_opex_v1"
                    else ()
                ),
                location=f"{eval_location}.expected_protocol",
                allow_extra=(
                    "environment_seeds",
                    "action_noise_seeds",
                    "gradient_noise_seed_start",
                    "gradient_noise_seeds",
                ),
            )
            if int(protocol["episode_count"]) <= 0:
                raise AggregationError(f"{eval_location} episode_count must be positive")
            if raw_schema == "channel_opex_v1" and not isinstance(
                protocol.get("gradient_noise_seed_start"), int
            ):
                raise AggregationError(
                    f"{eval_location}.gradient_noise_seed_start must be an integer"
                )
            if raw_schema == "channel_opex_v1":
                expected_antithetic = float(config["model_beta"]) > 0.0
                expected_sampling = (
                    "per_gradient_step" if expected_antithetic else "none_beta_zero"
                )
                expected_namespace = (
                    "torch_generator_per_episode_fresh_antithetic_per_gradient_step"
                    if expected_antithetic
                    else "none_beta_zero_deterministic_objective"
                )
                required_protocol_values = {
                    "paired_environment_and_action_noise_seeds": True,
                    "environment_rng_namespace": "gymnasium_env_reset",
                    "actuator_noise_rng_namespace": "numpy_generator_per_episode",
                    "gradient_noise_rng_namespace": expected_namespace,
                    "gradient_noise_sampling_frequency": expected_sampling,
                    "gradient_noise_antithetic": expected_antithetic,
                    "episode_local_gradient_streams_prevent_cross_episode_call_order_coupling": True,
                    "selection_rule": "all requested episodes retained",
                }
                for field, expected_value in required_protocol_values.items():
                    if protocol[field] != expected_value:
                        raise AggregationError(
                            f"{eval_location}.{field} is inconsistent with "
                            "the declared OPEX controller"
                        )
        if raw_schema == "channel_opex_v1":
            expected_cost = _require_exact_keys(
                spec.get("expected_cost"),
                (
                    "cost_scope",
                    "base_actor_rows_per_adapted_environment_step",
                    "q1_rows_per_adapted_step",
                    "q1_backward_calls_per_adapted_step",
                ),
                location=f"{location}.expected_cost",
            )
            expected_rows = int(config["gradient_steps"]) * int(
                config["execution_noise_samples"]
            )
            if expected_cost != {
                "cost_scope": "adapted_arm_deployment_controller_only",
                "base_actor_rows_per_adapted_environment_step": 1,
                "q1_rows_per_adapted_step": expected_rows,
                "q1_backward_calls_per_adapted_step": int(config["gradient_steps"]),
            }:
                raise AggregationError(f"{location}.expected_cost is inconsistent")

    comparison_ids: List[str] = []
    comparison_specs_by_id: Dict[str, Mapping[str, Any]] = {}
    for index, comparison in enumerate(manifest.get("comparisons", [])):
        if not isinstance(comparison, Mapping):
            raise AggregationError(f"manifest.comparisons[{index}] must be an object")
        comparison_id = comparison.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise AggregationError(
                f"manifest.comparisons[{index}].comparison_id must be non-empty"
            )
        if comparison_id in comparison_specs_by_id:
            raise AggregationError(f"duplicate comparison_id {comparison_id!r}")
        comparison_ids.append(comparison_id)
        comparison_specs_by_id[comparison_id] = comparison
        require_same = comparison.get("require_same_base_checkpoint")
        if not isinstance(require_same, bool):
            raise AggregationError(
                f"manifest.comparisons[{index}] must declare "
                "require_same_base_checkpoint as a boolean"
            )
        if require_same is False:
            if comparison.get("comparison_class") != "different_trained_policy":
                raise AggregationError(
                    f"manifest.comparisons[{index}] may waive same-base only for "
                    "comparison_class=different_trained_policy"
                )
            for field in (
                "expected_target_base_checkpoint_sha256",
                "expected_reference_base_checkpoint_sha256",
            ):
                if not _is_sha256(comparison.get(field)):
                    raise AggregationError(
                        f"manifest.comparisons[{index}].{field} is not SHA256"
                    )
    opex_count = sum(
        spec.get("raw_schema") == "channel_opex_v1" for spec in external_specs
    )
    if opex_count == 6:
        for spec in external_specs:
            if spec.get("raw_schema") != "channel_opex_v1":
                continue
            if spec.get("expected_config", {}).get("gradient_noise_seed") != 69300:
                raise AggregationError(
                    "formal OPEX controllers must share gradient_noise_seed=69300"
                )
            evaluations = spec.get("evaluations", [])
            if len(evaluations) != 1 or evaluations[0].get(
                "expected_protocol", {}
            ).get("gradient_noise_seed_start") != 69300:
                raise AggregationError(
                    "formal OPEX evaluations must share gradient seed block 69300..69349"
                )
        declared_primary = manifest.get("primary_comparison_order")
        if declared_primary != list(PRIMARY_CONFIRMATION_COMPARISON_IDS):
            raise AggregationError(
                "manifest.primary_comparison_order differs from the frozen "
                "CA-OPEX comparison contract"
            )
        actual_primary_order = [
            comparison_id
            for comparison_id in comparison_ids
            if comparison_id in PRIMARY_CONFIRMATION_COMPARISON_IDS
        ]
        if actual_primary_order != list(PRIMARY_CONFIRMATION_COMPARISON_IDS):
            raise AggregationError(
                "primary CA-OPEX comparisons are missing, duplicated, or out of order"
            )
        endpoint_contracts = {
            "seed1_beta125_ca_opex_inverse_minus_ca_opex_identity": (
                ("external_ca_opex_inverse_seed1", "adapted"),
                ("external_ca_opex_identity_seed1", "adapted"),
            ),
            "seed10_beta125_ca_opex_inverse_minus_ca_opex_identity": (
                ("external_ca_opex_inverse_seed10", "adapted"),
                ("external_ca_opex_identity_seed10", "adapted"),
            ),
            "seed1_beta125_ca_opex_inverse_minus_original_opex": (
                ("external_ca_opex_inverse_seed1", "adapted"),
                ("external_opex_original_seed1", "adapted"),
            ),
            "seed10_beta125_ca_opex_inverse_minus_original_opex": (
                ("external_ca_opex_inverse_seed10", "adapted"),
                ("external_opex_original_seed10", "adapted"),
            ),
            "seed1_beta125_ca_opex_inverse_effect": (
                ("external_ca_opex_inverse_seed1", "adapted"),
                ("external_ca_opex_inverse_seed1", "baseline_only"),
            ),
            "seed10_beta125_ca_opex_inverse_effect": (
                ("external_ca_opex_inverse_seed10", "adapted"),
                ("external_ca_opex_inverse_seed10", "baseline_only"),
            ),
        }
        for comparison_id in PRIMARY_CONFIRMATION_COMPARISON_IDS:
            comparison = comparison_specs_by_id[comparison_id]
            target_expected, reference_expected = endpoint_contracts[comparison_id]
            target = comparison.get("target", {})
            reference = comparison.get("reference", {})
            if (
                comparison.get("evidence_label") != "confirmation"
                or comparison.get("require_same_base_checkpoint") is not True
                or target.get("run_id") != target_expected[0]
                or target.get("evaluation_id") != "confirm_beta125"
                or target.get("arm") != target_expected[1]
                or reference.get("run_id") != reference_expected[0]
                or reference.get("evaluation_id") != "confirm_beta125"
                or reference.get("arm") != reference_expected[1]
            ):
                raise AggregationError(
                    f"primary comparison {comparison_id!r} differs from its frozen "
                    "controller/arm contract"
                )
    equivalence_checks = manifest.get("baseline_equivalence_checks", [])
    if not isinstance(equivalence_checks, list):
        raise AggregationError("manifest.baseline_equivalence_checks must be a list")
    seen_equivalence_ids: set[str] = set()
    for index, check in enumerate(equivalence_checks):
        location = f"manifest.baseline_equivalence_checks[{index}]"
        check = _require_exact_keys(
            check,
            (
                "check_id",
                "evidence_label",
                "expected_baseline_semantics",
                "expected_rollout_beta",
                "expected_base_checkpoint_sha256",
                "endpoints",
            ),
            location=location,
        )
        if not isinstance(check["check_id"], str) or not check["check_id"]:
            raise AggregationError(f"{location}.check_id invalid")
        if check["check_id"] in seen_equivalence_ids:
            raise AggregationError(f"duplicate baseline equivalence check_id")
        seen_equivalence_ids.add(check["check_id"])
        if check["evidence_label"] != "confirmation":
            raise AggregationError(f"{location} must be confirmation")
        if not isinstance(check["expected_baseline_semantics"], str) or not check[
            "expected_baseline_semantics"
        ]:
            raise AggregationError(f"{location} baseline semantics invalid")
        if not _is_sha256(check["expected_base_checkpoint_sha256"]):
            raise AggregationError(f"{location} base checkpoint SHA invalid")
        endpoints = check["endpoints"]
        if not isinstance(endpoints, list) or len(endpoints) < 2:
            raise AggregationError(f"{location} requires at least two endpoints")
        endpoint_keys = []
        for endpoint_index, endpoint in enumerate(endpoints):
            endpoint = _require_exact_keys(
                endpoint,
                ("run_id", "evaluation_id", "arm"),
                location=f"{location}.endpoints[{endpoint_index}]",
            )
            if endpoint["arm"] not in ARMS:
                raise AggregationError(f"{location} endpoint arm invalid")
            endpoint_keys.append(
                (endpoint["run_id"], endpoint["evaluation_id"], endpoint["arm"])
            )
        if len(endpoint_keys) != len(set(endpoint_keys)):
            raise AggregationError(f"{location} has duplicate endpoints")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AggregationError(f"cannot read JSON {path}: {exc}") from exc
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"JSON root must be an object: {path}")
    return value, {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _validate_manifest_has_no_results(value: Any, location: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN_MANIFEST_RESULT_KEYS:
                raise AggregationError(
                    f"{location}.{key} is forbidden: result values must come from raw files"
                )
            _validate_manifest_has_no_results(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_manifest_has_no_results(child, f"{location}[{index}]")


def _numeric_list(record: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> List[float]:
    raw = record.get(key)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise AggregationError(f"{key!r} must be a {'possibly empty ' if allow_empty else ''}list")
    try:
        values = [float(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"{key!r} must contain numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise AggregationError(f"{key!r} contains non-finite values")
    return values


def _integer_list(record: Mapping[str, Any], key: str) -> List[int]:
    raw = record.get(key)
    if not isinstance(raw, list) or not raw:
        raise AggregationError(f"{key!r} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise AggregationError(f"{key!r} must contain integer seeds")
    return list(raw)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-7)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AggregationError("percentile requires non-empty values")
    if not 0.0 <= probability <= 1.0:
        raise AggregationError("percentile probability must be in [0, 1]")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_statistics(
    differences: Sequence[float],
    *,
    normalized_scale: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    """Return paired t and fixed-seed percentile-bootstrap intervals."""

    values = [float(item) for item in differences]
    if not values:
        raise AggregationError("paired comparison contains no episodes")
    if bootstrap_samples <= 0:
        raise AggregationError("bootstrap_samples must be positive")
    count = len(values)
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if count > 1 else None
    if sample_std is None:
        t_low = t_high = mean
    else:
        critical = float(student_t.ppf(0.975, count - 1))
        half_width = critical * sample_std / math.sqrt(count)
        t_low, t_high = mean - half_width, mean + half_width
    rng = random.Random(int(bootstrap_seed))
    boot_means = [
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(int(bootstrap_samples))
    ]
    boot_low = percentile(boot_means, 0.025)
    boot_high = percentile(boot_means, 0.975)

    def scaled(value: Optional[float]) -> Optional[float]:
        return None if value is None else value * normalized_scale

    return {
        "statistical_unit": "paired_evaluation_episode_for_fixed_checkpoints",
        "episode_pairs": count,
        "positive_pairs": sum(value > 0.0 for value in values),
        "negative_pairs": sum(value < 0.0 for value in values),
        "zero_pairs": sum(value == 0.0 for value in values),
        "raw": {
            "mean_difference": mean,
            "sample_std_of_differences": sample_std,
            "t_95ci_low": t_low,
            "t_95ci_high": t_high,
            "paired_bootstrap_95ci_low": boot_low,
            "paired_bootstrap_95ci_high": boot_high,
        },
        "normalized": {
            "mean_difference": scaled(mean),
            "sample_std_of_differences": scaled(sample_std),
            "t_95ci_low": scaled(t_low),
            "t_95ci_high": scaled(t_high),
            "paired_bootstrap_95ci_low": scaled(boot_low),
            "paired_bootstrap_95ci_high": scaled(boot_high),
        },
        "bootstrap": {
            "seed": int(bootstrap_seed),
            "samples": int(bootstrap_samples),
            "method": "paired_episode_percentile",
            "confidence": 0.95,
        },
    }


def _normalization_for(
    manifest: Mapping[str, Any], environment: str
) -> Tuple[float, Dict[str, Any]]:
    normalizations = manifest.get("normalization")
    if not isinstance(normalizations, Mapping) or environment not in normalizations:
        raise AggregationError(f"manifest lacks normalization for {environment!r}")
    item = normalizations[environment]
    if not isinstance(item, Mapping):
        raise AggregationError(f"normalization for {environment!r} must be an object")
    try:
        minimum = float(item["reference_min_score"])
        maximum = float(item["reference_max_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AggregationError("normalization bounds must be numeric") from exc
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise AggregationError("normalization bounds must be finite and ordered")
    return 100.0 / (maximum - minimum), {
        "reference_min_score": minimum,
        "reference_max_score": maximum,
        "source": item.get("source"),
    }


def _resolve_results_path(results_root: Path, relative: Any, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AggregationError(f"{role} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise AggregationError(f"{role} must be relative to results_root")
    return (results_root / candidate).resolve()


def _summarize_arm(
    arm: Mapping[str, Any],
    *,
    normalization_scale: float,
    normalization_minimum: float,
    expected_episode_count: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    returns = _numeric_list(arm, "returns")
    lengths = _integer_list(arm, "lengths")
    environment_seeds = _integer_list(arm, "environment_seeds")
    action_noise_seeds = _integer_list(arm, "action_noise_seeds")
    count = len(returns)
    for name, values in (
        ("lengths", lengths),
        ("environment_seeds", environment_seeds),
        ("action_noise_seeds", action_noise_seeds),
    ):
        if len(values) != count:
            raise AggregationError(f"{name} count does not match returns")
    if expected_episode_count is not None and count != expected_episode_count:
        raise AggregationError("episode count differs from evaluation_protocol")
    if len(set(environment_seeds)) != count or len(set(action_noise_seeds)) != count:
        raise AggregationError("evaluation seed arrays must contain unique values")
    return_mean = statistics.fmean(returns)
    return_std = statistics.pstdev(returns)
    normalized = [
        (value - normalization_minimum) * normalization_scale for value in returns
    ]
    normalized_mean = statistics.fmean(normalized)
    normalized_std = statistics.pstdev(normalized)
    reported = {
        "return_mean": arm.get("return_mean"),
        "return_std": arm.get("return_std"),
        "normalized_score_mean": arm.get("normalized_score_mean"),
        "normalized_score_std": arm.get("normalized_score_std"),
    }
    recomputed = {
        "return_mean": return_mean,
        "return_std": return_std,
        "normalized_score_mean": normalized_mean,
        "normalized_score_std": normalized_std,
    }
    for key, value in recomputed.items():
        if reported[key] is None or not _close(float(reported[key]), value):
            raise AggregationError(f"reported {key} does not match raw returns")
    public = {
        "episode_count": count,
        "return_mean": return_mean,
        "return_population_std": return_std,
        "normalized_score_mean": normalized_mean,
        "normalized_score_population_std": normalized_std,
        "environment_seed_first": min(environment_seeds),
        "environment_seed_last": max(environment_seeds),
        "action_noise_seed_first": min(action_noise_seeds),
        "action_noise_seed_last": max(action_noise_seeds),
        "action_noise_beta": float(arm["action_noise_beta"]),
        "total_environment_steps": sum(lengths),
        "actuator_clip_fraction": arm.get("actuator_clip_fraction"),
        "command_at_bound_fraction": arm.get("command_at_bound_fraction"),
        "inverse_saturation_fraction": arm.get("inverse_saturation_fraction"),
        "executed_commanded_action_mse": arm.get("executed_commanded_action_mse"),
        "proposed_residual_abs_mean": arm.get("proposed_residual_abs_mean"),
        "proposed_residual_abs_max": arm.get("proposed_residual_abs_max"),
        "applied_residual_abs_mean": arm.get("applied_residual_abs_mean"),
    }
    private = {
        "returns": returns,
        "environment_seeds": environment_seeds,
        "action_noise_seeds": action_noise_seeds,
        "action_noise_beta": float(arm["action_noise_beta"]),
    }
    return public, private


def _parse_evaluation(
    path: Path,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    strict_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    evaluation, source = _read_json(path)
    if evaluation.get("status") != "complete":
        raise AggregationError(f"evaluation is not complete: {path}")
    evidence_label = spec.get("evidence_label")
    if evidence_label not in EVIDENCE_LABELS - {"implementation_smoke"}:
        raise AggregationError("evaluation evidence_label must be development or confirmation")
    environment = evaluation.get("environment")
    if not isinstance(environment, str) or not environment:
        raise AggregationError(f"evaluation lacks environment: {path}")
    scale, normalization = _normalization_for(manifest, environment)
    protocol = evaluation.get("evaluation_protocol", {})
    declared_count = protocol.get("episode_count_per_method")
    if declared_count is not None:
        declared_count = int(declared_count)
    arms_public: Dict[str, Dict[str, Any]] = {}
    arms_private: Dict[str, Dict[str, Any]] = {}
    for arm_name in sorted(ADAPTER_ARMS):
        arm = evaluation.get(arm_name)
        if not isinstance(arm, Mapping):
            raise AggregationError(f"evaluation lacks arm {arm_name!r}: {path}")
        public, private = _summarize_arm(
            arm,
            normalization_scale=scale,
            normalization_minimum=normalization["reference_min_score"],
            expected_episode_count=declared_count,
        )
        arms_public[arm_name] = public
        arms_private[arm_name] = private
    for arm_name, public in arms_public.items():
        environment_steps = int(public["total_environment_steps"])
        public["base_actor_forward_rows"] = environment_steps
        public["adapter_mlp_forward_rows"] = (
            environment_steps if arm_name == "adapted" else 0
        )
        public["controller_cost_scope"] = "per_rollout_arm_raw_length_derived"
    baseline_private = arms_private["baseline_only"]
    adapted_private = arms_private["adapted"]
    for key in ("environment_seeds", "action_noise_seeds"):
        if baseline_private[key] != adapted_private[key]:
            raise AggregationError(f"evaluation arms have different {key}")
    if baseline_private["action_noise_beta"] != adapted_private["action_noise_beta"]:
        raise AggregationError("evaluation arms have different action-noise beta")
    stored_paired = evaluation.get("paired", {}).get("adapted_minus_baseline_returns")
    computed_paired = [
        target - reference
        for target, reference in zip(
            adapted_private["returns"], baseline_private["returns"]
        )
    ]
    if not isinstance(stored_paired, list) or len(stored_paired) != len(computed_paired):
        raise AggregationError("evaluation lacks a complete stored paired difference")
    if any(not _close(float(left), right) for left, right in zip(stored_paired, computed_paired)):
        raise AggregationError("stored paired differences do not match raw returns")
    channel = evaluation.get("channel", {})
    if strict_context is not None:
        expected = spec["expected_protocol"]
        if environment != expected["environment"]:
            raise AggregationError("evaluation environment differs from frozen protocol")
        if declared_count != int(expected["episode_count"]):
            raise AggregationError("evaluation episode count differs from frozen protocol")
        if channel.get("environment_rollout_beta") != expected["rollout_beta"]:
            raise AggregationError("evaluation rollout beta differs from frozen protocol")
        expected_environment_seeds = expected.get("environment_seeds")
        if expected_environment_seeds is None:
            start = int(expected["environment_seed_start"])
            expected_environment_seeds = list(
                range(start, start + int(expected["episode_count"]))
            )
        expected_noise_seeds = expected.get("action_noise_seeds")
        if expected_noise_seeds is None:
            start = int(expected["action_noise_seed_start"])
            expected_noise_seeds = list(
                range(start, start + int(expected["episode_count"]))
            )
        for arm_name, private in arms_private.items():
            if private["environment_seeds"] != expected_environment_seeds:
                raise AggregationError(
                    f"evaluation {arm_name} environment seeds differ from frozen protocol"
                )
            if private["action_noise_seeds"] != expected_noise_seeds:
                raise AggregationError(
                    f"evaluation {arm_name} noise seeds differ from frozen protocol"
                )
        if evaluation.get("baseline_transform") != strict_context["baseline_transform"]:
            raise AggregationError("evaluation baseline transform differs from frozen config")
        if channel.get("model_or_calibration_beta") != strict_context[
            "model_or_calibration_beta"
        ]:
            raise AggregationError("evaluation model beta differs from frozen config")
        if evaluation.get("adapter_checkpoint_sha256") != strict_context[
            "checkpoint_sha256"
        ]:
            raise AggregationError("evaluation adapter checkpoint SHA mismatch")
        if evaluation.get("base_checkpoint_sha256") != strict_context[
            "base_checkpoint_sha256"
        ]:
            raise AggregationError("evaluation base checkpoint SHA mismatch")
        implementation = evaluation.get("implementation", {})
        for field in ("train_sha256", "core_sha256", "evaluate_sha256"):
            if implementation.get(field) != strict_context["implementation"][field]:
                raise AggregationError(f"evaluation implementation {field} mismatch")
    public_evaluation = {
        "evaluation_id": spec.get("evaluation_id"),
        "evidence_label": evidence_label,
        "selection_note": spec.get("selection_note"),
        "environment": environment,
        "status": "complete",
        "baseline_transform": evaluation.get("baseline_transform"),
        "adapter_step": evaluation.get("adapter_step"),
        "adapter_checkpoint_sha256": evaluation.get("adapter_checkpoint_sha256"),
        "base_checkpoint_sha256": evaluation.get("base_checkpoint_sha256"),
        "environment_rollout_beta": channel.get("environment_rollout_beta"),
        "model_or_calibration_beta": channel.get("model_or_calibration_beta"),
        "beta_mismatch": channel.get("beta_mismatch"),
        "wall_time_seconds": evaluation.get("wall_time_seconds"),
        "wall_time_scope": "paired_evaluation_total_only_not_attributable_to_arm",
        "normalization": normalization,
        "arms": arms_public,
        "raw_source": {
            **source,
            "json_pointers": {
                "baseline_returns": "/baseline_only/returns",
                "adapted_returns": "/adapted/returns",
            },
        },
    }
    private = {
        "environment": environment,
        "normalization_scale": scale,
        "arms": arms_private,
    }
    return public_evaluation, arms_private, {**source, "private": private}


def _metric_fields(metric: Any) -> Dict[str, Any]:
    if not isinstance(metric, Mapping):
        return {
            "mean": None,
            "median": None,
            "std": None,
            "positive_fraction": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": metric.get("mean"),
        "median": metric.get("median"),
        "std": metric.get("std"),
        "positive_fraction": metric.get("positive_fraction"),
        "min": metric.get("min"),
        "max": metric.get("max"),
    }


def _parse_audit(
    path: Path,
    spec: Mapping[str, Any],
    *,
    strict_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    audit, source = _read_json(path)
    if audit.get("status") != "complete":
        raise AggregationError(f"audit is not complete: {path}")
    evidence_label = spec.get("evidence_label")
    if evidence_label not in EVIDENCE_LABELS - {"implementation_smoke"}:
        raise AggregationError("audit evidence_label must be development or confirmation")
    metrics = audit.get("metrics", {})
    channel = audit.get("audit_channel", {})
    noise_seed_independent = channel.get(
        "noise_seed_distinct_from_training_channel",
        channel.get("independent_from_training_seed"),
    )
    if strict_context is not None:
        expected = spec["expected_protocol"]
        dataset = audit.get("dataset", {})
        recorded_split = dataset.get("recorded_split", {})
        checks = (
            ("noise_samples", channel.get("sample_count"), expected["noise_samples"]),
            ("noise_seed", channel.get("seed"), expected["noise_seed"]),
            (
                "action_noise_beta",
                channel.get("audit_action_noise_beta"),
                expected["action_noise_beta"],
            ),
            (
                "selected_observations",
                dataset.get("selected_observations"),
                expected["selected_observations"],
            ),
            (
                "train_indices_sha256",
                dataset.get("train_indices_sha256"),
                expected["train_indices_sha256"],
            ),
            (
                "audit_indices_sha256",
                dataset.get("selected_indices_sha256"),
                expected["audit_indices_sha256"],
            ),
            (
                "audit_episode_units_sha256",
                recorded_split.get("audit_episode_units_sha256"),
                expected["audit_episode_units_sha256"],
            ),
        )
        for field, actual, frozen in checks:
            if actual != frozen:
                raise AggregationError(f"audit frozen {field} mismatch")
        if dataset.get("sha256") != strict_context["dataset_sha256"]:
            raise AggregationError("audit dataset SHA mismatch")
        if dataset.get("train_audit_disjoint") is not True:
            raise AggregationError("audit does not certify train/audit disjointness")
        if noise_seed_independent is not True:
            raise AggregationError(
                "audit does not certify a noise seed distinct from training"
            )
        if audit.get("baseline_transform") != strict_context["baseline_transform"]:
            raise AggregationError("audit baseline transform differs from frozen config")
        if audit.get("adapter_checkpoint_sha256") != strict_context[
            "checkpoint_sha256"
        ]:
            raise AggregationError("audit adapter checkpoint SHA mismatch")
        if audit.get("base_checkpoint_sha256") != strict_context[
            "base_checkpoint_sha256"
        ]:
            raise AggregationError("audit base checkpoint SHA mismatch")
        implementation = audit.get("implementation")
        if not isinstance(implementation, Mapping):
            raise AggregationError(
                "strict audit lacks train/core/evaluate/audit implementation hashes"
            )
        for field in (
            "train_sha256",
            "core_sha256",
            "evaluate_sha256",
            "audit_sha256",
        ):
            if implementation.get(field) != strict_context["implementation"][field]:
                raise AggregationError(f"audit implementation {field} mismatch")
    public = {
        "audit_id": spec.get("audit_id"),
        "evidence_label": evidence_label,
        "selection_note": spec.get("selection_note"),
        "status": "complete",
        "baseline_transform": audit.get("baseline_transform"),
        "adapter_step": audit.get("adapter_step"),
        "wall_time_seconds": audit.get("wall_time_seconds"),
        "selected_observations": audit.get("dataset", {}).get("selected_observations"),
        "selected_indices_sha256": audit.get("dataset", {}).get("selected_indices_sha256"),
        "state_seed": audit.get("dataset", {}).get("state_seed"),
        "noise_seed": channel.get("seed"),
        "noise_samples": channel.get("sample_count"),
        "audit_action_noise_beta": channel.get("audit_action_noise_beta"),
        "model_or_calibration_beta": channel.get("model_or_calibration_beta"),
        "beta_mismatch": channel.get("beta_mismatch"),
        "independent_from_training_seed": noise_seed_independent,
        "physical_action_rows": audit.get("cost", {}).get("physical_action_rows"),
        "individual_critic_network_rows": audit.get("cost", {}).get(
            "individual_critic_network_rows"
        ),
        "q1_gain": _metric_fields(metrics.get("adapted_minus_baseline_q1")),
        "q2_gain": _metric_fields(metrics.get("adapted_minus_baseline_q2")),
        "min_twin_gain": _metric_fields(
            metrics.get("adapted_minus_baseline_min_twin")
        ),
        "baseline_twin_disagreement": _metric_fields(
            metrics.get("baseline_absolute_twin_disagreement")
        ),
        "adapted_twin_disagreement": _metric_fields(
            metrics.get("adapted_absolute_twin_disagreement")
        ),
        "twin_disagreement_change": _metric_fields(
            metrics.get("adapted_minus_baseline_twin_disagreement")
        ),
        "command_saturation_fraction": metrics.get("command_saturation_fraction"),
        "proposed_residual_abs": _metric_fields(metrics.get("proposed_residual_abs")),
        "interpretation_guardrail": audit.get("interpretation_guardrail"),
        "raw_source": {**source, "json_pointer": "/metrics"},
    }
    return public, source


def _path_source(path: Path, role: str) -> Dict[str, Any]:
    if not path.exists():
        return {"role": role, "path": str(path), "status": "missing"}
    return {
        "role": role,
        "path": str(path.resolve()),
        "status": "read",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_method_config(expected: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove repeat-specific RNG ids while retaining every scientific knob."""

    return {
        key: value
        for key, value in expected.items()
        if key not in {"train_seed", "channel_seed"}
    }


def _lookup_extension(config: Mapping[str, Any], dotted_path: str) -> Any:
    if not isinstance(dotted_path, str) or "." not in dotted_path:
        raise AggregationError(
            "expected_config.extensions keys must be dotted raw-config paths"
        )
    root, *parts = dotted_path.split(".")
    if root not in {"adapter", "arguments", "channel_calibration", "observation_split"}:
        raise AggregationError(f"unsupported expected extension root {root!r}")
    value: Any = config.get(root)
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            raise AggregationError(f"raw config lacks declared extension {dotted_path!r}")
        value = value[part]
    return value


def _actual_strict_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    args = config.get("arguments")
    adapter = config.get("adapter")
    calibration = config.get("channel_calibration")
    if not isinstance(args, Mapping) or not isinstance(adapter, Mapping):
        raise AggregationError("strict run config lacks arguments/adapter mappings")
    if not isinstance(calibration, Mapping):
        raise AggregationError("strict run config lacks channel_calibration mapping")
    source = calibration.get("source")
    calibration_modes = {
        "censored_uniform_plus_clip_pair_calibration": "pair_calibration",
        "cli_known_beta_without_pair_calibration": "known_beta_without_pair_calibration",
    }
    if source not in calibration_modes:
        raise AggregationError(f"unsupported channel calibration source {source!r}")
    return {
        "baseline_transform": adapter.get("baseline_transform"),
        "value_estimator": adapter.get("value_estimator"),
        "alpha": adapter.get("alpha"),
        "residual_penalty": adapter.get("residual_penalty"),
        "delta_max": adapter.get("delta_max"),
        "execution_noise_samples": adapter.get("execution_noise_samples"),
        "updates": args.get("updates"),
        "learning_rate": args.get("learning_rate"),
        "batch_size": args.get("batch_size"),
        "split_seed": args.get("split_seed"),
        "audit_fraction": args.get("audit_fraction"),
        "train_seed": args.get("train_seed"),
        "channel_seed": adapter.get("execution_noise_seed"),
        "model_or_calibration_beta": adapter.get("execution_noise_beta"),
        "calibration_mode": calibration_modes[source],
        "hidden_dim": adapter.get("hidden_dim"),
        "depth": adapter.get("depth"),
        "q_scale_epsilon": adapter.get("q_scale_epsilon"),
        "precompute_batch_size": args.get("precompute_batch_size"),
        "scale_calibration_observations": args.get(
            "scale_calibration_observations"
        ),
        "max_observations": args.get("max_observations"),
        "base_action_pairing": config.get("base_action_pairing"),
    }


def _load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise AggregationError(
            "strict aggregation requires torch to inspect checkpoint format/config"
        ) from exc
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise AggregationError(f"cannot inspect adapter checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AggregationError("adapter checkpoint root must be a mapping")
    return {
        "format": payload.get("format"),
        "adapter_config": payload.get("adapter_config"),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "base_parameter_sha256": payload.get("base_parameter_sha256"),
        "step": payload.get("step"),
        "config": payload.get("config"),
    }


def _validate_strict_run_artifacts(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    frozen_implementation: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_config = spec["expected_config"]
    actual_config = _actual_strict_config(config)
    for field in STRICT_CONFIG_FIELDS:
        if actual_config[field] != expected_config[field]:
            raise AggregationError(
                f"run {spec['run_id']} expected_config.{field} mismatch: "
                f"{actual_config[field]!r} != {expected_config[field]!r}"
            )
    for dotted_path, expected_value in expected_config.get("extensions", {}).items():
        actual_value = _lookup_extension(config, dotted_path)
        if actual_value != expected_value:
            raise AggregationError(
                f"run {spec['run_id']} extension {dotted_path!r} mismatch"
            )
    artifacts = spec["expected_artifacts"]
    dataset_sha = config.get("dataset", {}).get("sha256")
    base_sha = config.get("base_checkpoint_sha256")
    calibration_sha = config.get("channel_calibration", {}).get("calibration_sha256")
    for field, actual in (
        ("dataset_sha256", dataset_sha),
        ("base_checkpoint_sha256", base_sha),
        ("calibration_sha256", calibration_sha),
    ):
        if actual != artifacts[field]:
            raise AggregationError(
                f"run {spec['run_id']} expected_artifacts.{field} mismatch"
            )
    implementation = config.get("implementation")
    if not isinstance(implementation, Mapping):
        raise AggregationError(f"run {spec['run_id']} lacks implementation manifest")
    for field in TRAIN_IMPLEMENTATION_FIELDS:
        if implementation.get(field) != frozen_implementation[field]:
            raise AggregationError(
                f"run {spec['run_id']} implementation {field} mismatch"
            )
    checkpoint = _load_checkpoint_metadata(checkpoint_path)
    if checkpoint["format"] != artifacts["checkpoint_format"]:
        raise AggregationError(f"run {spec['run_id']} checkpoint format mismatch")
    if checkpoint["adapter_config"] != config.get("adapter"):
        raise AggregationError(f"run {spec['run_id']} checkpoint/config adapter mismatch")
    if checkpoint["base_checkpoint_sha256"] != artifacts["base_checkpoint_sha256"]:
        raise AggregationError(f"run {spec['run_id']} checkpoint base SHA mismatch")
    if checkpoint["step"] != summary.get("updates"):
        raise AggregationError(f"run {spec['run_id']} checkpoint/summary step mismatch")
    if checkpoint["config"] != config:
        raise AggregationError(f"run {spec['run_id']} embedded/disk config mismatch")
    method_config = _canonical_method_config(expected_config)
    method_config_sha = sha256_bytes(canonical_json(method_config).encode("utf-8"))
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint["format"],
        "method_config_sha256": method_config_sha,
        "verified_expected_config": expected_config,
        "verified_expected_artifacts": artifacts,
    }


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregationError(f"{field} must be a non-negative integer")
    return int(value)


def _adapter_training_cost(
    *,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    """Extract cost without conflating per-invocation and lifetime scopes."""

    args = config.get("arguments", {})
    adapter = config.get("adapter", {})
    baseline = summary.get("baseline_precomputation", {})
    value_scale = summary.get("value_scale_calibration", {})
    public: Dict[str, Any] = {
        "updates_completed": summary.get("updates"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "wall_time_scope": (
            "cumulative_across_resume_invocations_including_preprocessing_and_optimizer"
            if strict
            else "legacy_unspecified"
        ),
        "preprocessing_seconds_this_invocation": summary.get(
            "preprocessing_seconds_this_invocation",
            summary.get("preprocessing_seconds"),
        ),
        "preprocessing_time_scope": (
            "latest_invocation_only_not_lifetime"
            if "preprocessing_seconds_this_invocation" in summary
            else "legacy_unspecified"
        ),
        "optimizer_seconds": summary.get("optimizer_seconds"),
        "optimizer_time_scope": (
            "cumulative_across_resume_invocations" if strict else "legacy_unspecified"
        ),
        "invocation_count": None,
        "preprocessing_recomputed_each_invocation": None,
        "preprocessing_seconds_lifetime": None,
        "preprocessing_seconds_lifetime_available": None,
        "base_actor_precompute_rows_per_invocation": None,
        "base_actor_precompute_rows_lifetime": None,
        "q_scale_q1_forward_rows_per_invocation": None,
        "q_scale_q1_forward_rows_lifetime": None,
        "q1_input_rows": summary.get("q1_input_rows"),
        "optimizer_q1_forward_rows_lifetime": summary.get("q1_input_rows"),
        "optimizer_q1_backward_rows_lifetime": None,
        "optimizer_adapter_mlp_forward_rows_lifetime": None,
        "optimizer_adapter_mlp_backward_rows_lifetime": None,
        "total_q1_forward_rows_lifetime": None,
        "q1_rows_per_state_per_update": summary.get(
            "q1_rows_per_state_per_update"
        ),
        "frozen_q_scale": value_scale.get("scale")
        if isinstance(value_scale, Mapping)
        else None,
        "q_scale_method": value_scale.get("method")
        if isinstance(value_scale, Mapping)
        else None,
        "half_vectors_drawn": summary.get("channel", {}).get(
            "half_vectors_drawn"
        ),
        "channel_draw_calls": summary.get("channel", {}).get("draw_calls"),
        "resume_scope": (
            "state_complete_same_runtime_continuation; CPU bitwise regression "
            "verified; CUDA uninterrupted-versus-resumed bitwise equality unverified"
            if strict
            else "legacy_unspecified"
        ),
    }
    if not strict:
        return public

    if not isinstance(args, Mapping) or not isinstance(adapter, Mapping):
        raise AggregationError("strict training cost lacks arguments or adapter config")
    if not isinstance(baseline, Mapping) or not isinstance(value_scale, Mapping):
        raise AggregationError(
            "strict training cost lacks baseline/value-scale preprocessing metadata"
        )
    updates = _nonnegative_integer(summary.get("updates"), field="summary.updates")
    batch_size = _nonnegative_integer(args.get("batch_size"), field="batch_size")
    samples = _nonnegative_integer(
        adapter.get("execution_noise_samples"), field="execution_noise_samples"
    )
    if batch_size == 0 or samples == 0:
        raise AggregationError("strict batch size and execution-noise samples must be positive")
    resume_count = _nonnegative_integer(
        summary.get("resume_count"), field="summary.resume_count"
    )
    invocation_count = resume_count + 1
    if summary.get("resumed") is not (resume_count > 0):
        raise AggregationError("summary resumed/resume_count accounting mismatch")
    train_count = _nonnegative_integer(
        config.get("dataset", {}).get("training_observation_count"),
        field="dataset.training_observation_count",
    )
    precompute_rows = _nonnegative_integer(
        baseline.get("observation_count"),
        field="baseline_precomputation.observation_count",
    )
    if precompute_rows != train_count:
        raise AggregationError(
            "base actor precompute rows differ from the recorded training partition"
        )
    scale_observations = _nonnegative_integer(
        value_scale.get("observation_count"),
        field="value_scale_calibration.observation_count",
    )
    expected_scale_observations = min(
        _nonnegative_integer(
            args.get("scale_calibration_observations"),
            field="scale_calibration_observations",
        ),
        train_count,
    )
    if scale_observations != expected_scale_observations:
        raise AggregationError("value-scale observation row count mismatch")
    q_scale_rows = _nonnegative_integer(
        value_scale.get("q1_input_rows"),
        field="value_scale_calibration.q1_input_rows",
    )
    if q_scale_rows != scale_observations * samples:
        raise AggregationError("value-scale Q1 row count mismatch")
    optimizer_forward_rows = _nonnegative_integer(
        summary.get("q1_input_rows"), field="summary.q1_input_rows"
    )
    expected_optimizer_forward = updates * batch_size * 2 * samples
    if optimizer_forward_rows != expected_optimizer_forward:
        raise AggregationError("optimizer Q1 forward row count mismatch")
    expected_rows_per_state = 2 * samples
    if _nonnegative_integer(
        summary.get("q1_rows_per_state_per_update"),
        field="summary.q1_rows_per_state_per_update",
    ) != expected_rows_per_state:
        raise AggregationError("Q1 rows per state/update mismatch")
    optimizer_backward_rows = updates * batch_size * samples
    adapter_mlp_rows = updates * batch_size
    preprocessing_seconds = summary.get("preprocessing_seconds_this_invocation")
    if (
        isinstance(preprocessing_seconds, bool)
        or not isinstance(preprocessing_seconds, (int, float))
        or not math.isfinite(float(preprocessing_seconds))
        or float(preprocessing_seconds) < 0.0
    ):
        raise AggregationError(
            "strict summary preprocessing_seconds_this_invocation invalid"
        )
    for field in ("wall_time_seconds", "optimizer_seconds"):
        value = summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise AggregationError(f"strict summary {field} invalid")
    scale = value_scale.get("scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise AggregationError("strict frozen Q scale is invalid")
    q_scale_rows_lifetime = invocation_count * q_scale_rows
    precompute_rows_lifetime = invocation_count * precompute_rows
    public.update(
        {
            "invocation_count": invocation_count,
            "preprocessing_recomputed_each_invocation": True,
            "preprocessing_seconds_lifetime": (
                float(preprocessing_seconds) if invocation_count == 1 else None
            ),
            "preprocessing_seconds_lifetime_available": invocation_count == 1,
            "base_actor_precompute_rows_per_invocation": precompute_rows,
            "base_actor_precompute_rows_lifetime": precompute_rows_lifetime,
            "q_scale_q1_forward_rows_per_invocation": q_scale_rows,
            "q_scale_q1_forward_rows_lifetime": q_scale_rows_lifetime,
            "optimizer_q1_forward_rows_lifetime": optimizer_forward_rows,
            "optimizer_q1_backward_rows_lifetime": optimizer_backward_rows,
            "optimizer_adapter_mlp_forward_rows_lifetime": adapter_mlp_rows,
            "optimizer_adapter_mlp_backward_rows_lifetime": adapter_mlp_rows,
            "total_q1_forward_rows_lifetime": (
                optimizer_forward_rows + q_scale_rows_lifetime
            ),
        }
    )
    return public


def _parse_run(
    results_root: Path,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    for key in ("run_id", "method_id", "training_dir", "evidence_label", "expected_status"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise AggregationError(f"run spec requires non-empty {key}")
    if spec["evidence_label"] not in EVIDENCE_LABELS:
        raise AggregationError(f"unsupported evidence label {spec['evidence_label']!r}")
    if spec["expected_status"] not in EXPECTED_STATUSES:
        raise AggregationError(f"unsupported expected status {spec['expected_status']!r}")
    run_dir = _resolve_results_path(results_root, spec["training_dir"], "training_dir")
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "latest.pt"
    progress_path = run_dir / "progress.jsonl"
    strict = manifest.get("schema_version") == STRICT_MANIFEST_SCHEMA_VERSION
    sibling_console_path = run_dir.parent / f"{run_dir.name}.train.console.log"
    legacy_launch_path = run_dir / "launch.log"
    training_log_path = (
        sibling_console_path
        if strict or sibling_console_path.exists()
        else legacy_launch_path
    )
    sources: List[Dict[str, Any]] = []
    config, config_source = _read_json(config_path)
    config_source["role"] = "training_config"
    sources.append(config_source)
    args = config.get("arguments", {})
    adapter_config = config.get("adapter", {})
    training_seed = args.get("train_seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise AggregationError(f"run {spec['run_id']} lacks integer train_seed")
    expected_status = spec["expected_status"]
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        summary, summary_source = _read_json(summary_path)
        summary_source["role"] = "training_summary"
        sources.append(summary_source)
    else:
        sources.append(_path_source(summary_path, "training_summary"))
    sources.append(_path_source(progress_path, "training_progress"))
    training_log_source = _path_source(training_log_path, "training_console_log")
    training_log_source["layout"] = (
        "sibling_run_dir_dot_train_dot_console_dot_log"
        if training_log_path == sibling_console_path
        else "legacy_in_run_launch_log_fallback"
    )
    sources.append(training_log_source)
    checkpoint_source = _path_source(checkpoint_path, "adapter_checkpoint")
    sources.append(checkpoint_source)
    if expected_status == "complete":
        if summary.get("status") != "complete" or not checkpoint_path.is_file():
            raise AggregationError(
                f"run {spec['run_id']} is declared complete but raw artifacts are incomplete"
            )
        if strict and not sibling_console_path.is_file():
            raise AggregationError(
                f"run {spec['run_id']} lacks sibling training console log"
            )
        observed_status = "complete"
    else:
        if summary.get("status") == "complete" or checkpoint_path.exists():
            raise AggregationError(
                f"run {spec['run_id']} is declared failed_preupdate but has completed artifacts"
            )
        progress_nonempty = progress_path.exists() and progress_path.stat().st_size > 0
        if progress_nonempty:
            raise AggregationError(
                f"run {spec['run_id']} failed_preupdate but progress is non-empty"
            )
        observed_status = "failed_preupdate"
    strict_validation: Optional[Dict[str, Any]] = None
    if strict and expected_status == "complete":
        strict_validation = _validate_strict_run_artifacts(
            spec=spec,
            config=config,
            summary=summary,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=str(checkpoint_source.get("sha256")),
            frozen_implementation=manifest["frozen_implementation"],
        )
    public: Dict[str, Any] = {
        "run_id": spec["run_id"],
        "method_id": spec["method_id"],
        "evidence_label": spec["evidence_label"],
        "selection_note": spec.get("selection_note"),
        "expected_status": expected_status,
        "observed_status": observed_status,
        "training_dir": str(run_dir),
        "training_seed": training_seed,
        "baseline_transform": adapter_config.get("baseline_transform"),
        "residual_penalty": adapter_config.get("residual_penalty"),
        "alpha": adapter_config.get("alpha"),
        "delta_max": adapter_config.get("delta_max"),
        "noise_samples": adapter_config.get("execution_noise_samples"),
        "model_or_calibration_beta": adapter_config.get("execution_noise_beta"),
        "updates_requested": args.get("updates"),
        "training_cost": _adapter_training_cost(
            summary=summary,
            config=config,
            strict=strict and expected_status == "complete",
        ),
        "base_parameters_unchanged": summary.get("base_parameters_unchanged"),
        "fail_closed_validation": (
            {
                "status": "verified",
                **strict_validation,
            }
            if strict_validation is not None
            else {
                "status": "legacy_manifest_not_fail_closed"
                if not strict
                else "not_applicable_failed_preupdate"
            }
        ),
        "evaluations": [],
        "audits": [],
        "sources": sources,
    }
    private_evaluations: Dict[str, Dict[str, Any]] = {}
    if expected_status == "failed_preupdate":
        if spec.get("evaluations") or spec.get("audits"):
            raise AggregationError("failed_preupdate run cannot declare evaluations or audits")
        return public, private_evaluations, sources
    for evaluation_spec in spec.get("evaluations", []):
        if not isinstance(evaluation_spec, Mapping):
            raise AggregationError("evaluation spec must be an object")
        evaluation_id = evaluation_spec.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise AggregationError("evaluation spec requires evaluation_id")
        if evaluation_id in private_evaluations:
            raise AggregationError(f"duplicate evaluation_id {evaluation_id!r}")
        path = _resolve_results_path(results_root, evaluation_spec.get("path"), "evaluation path")
        parsed, private_arms, source_private = _parse_evaluation(
            path,
            evaluation_spec,
            manifest,
            strict_context=(
                {
                    "baseline_transform": strict_validation[
                        "verified_expected_config"
                    ]["baseline_transform"],
                    "model_or_calibration_beta": strict_validation[
                        "verified_expected_config"
                    ]["model_or_calibration_beta"],
                    "checkpoint_sha256": strict_validation["checkpoint_sha256"],
                    "base_checkpoint_sha256": strict_validation[
                        "verified_expected_artifacts"
                    ]["base_checkpoint_sha256"],
                    "implementation": {
                        **manifest["frozen_implementation"],
                    },
                }
                if strict_validation is not None
                else None
            ),
        )
        parsed["run_id"] = spec["run_id"]
        public["evaluations"].append(parsed)
        source = {key: value for key, value in source_private.items() if key != "private"}
        source["role"] = f"evaluation:{evaluation_id}"
        sources.append(source)
        private_evaluations[evaluation_id] = {
            "environment": source_private["private"]["environment"],
            "normalization_scale": source_private["private"]["normalization_scale"],
            "training_seed": training_seed,
            "base_checkpoint_sha256": config.get("base_checkpoint_sha256"),
            "arms": private_arms,
        }
    for audit_spec in spec.get("audits", []):
        if not isinstance(audit_spec, Mapping):
            raise AggregationError("audit spec must be an object")
        audit_id = audit_spec.get("audit_id")
        if not isinstance(audit_id, str) or not audit_id:
            raise AggregationError("audit spec requires audit_id")
        path = _resolve_results_path(results_root, audit_spec.get("path"), "audit path")
        parsed, source = _parse_audit(
            path,
            audit_spec,
            strict_context=(
                {
                    "baseline_transform": strict_validation[
                        "verified_expected_config"
                    ]["baseline_transform"],
                    "checkpoint_sha256": strict_validation["checkpoint_sha256"],
                    "base_checkpoint_sha256": strict_validation[
                        "verified_expected_artifacts"
                    ]["base_checkpoint_sha256"],
                    "dataset_sha256": strict_validation[
                        "verified_expected_artifacts"
                    ]["dataset_sha256"],
                    "implementation": {
                        **manifest["frozen_implementation"],
                    },
                }
                if strict_validation is not None
                else None
            ),
        )
        parsed["run_id"] = spec["run_id"]
        public["audits"].append(parsed)
        sources.append({**source, "role": f"audit:{audit_id}"})
    return public, private_evaluations, sources


def _external_expected_seeds(
    protocol: Mapping[str, Any], key: str, start_key: str
) -> Optional[List[int]]:
    declared = protocol.get(key)
    if declared is not None:
        if not isinstance(declared, list) or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in declared
        ):
            raise AggregationError(f"external protocol {key} must contain integers")
        return list(declared)
    start = protocol.get(start_key)
    if start is None:
        return None
    return list(range(int(start), int(start) + int(protocol["episode_count"])))


def _summarize_external_arm(
    arm: Mapping[str, Any],
    *,
    environment_seed_key: str,
    action_noise_seed_key: Optional[str],
    action_noise_beta: float,
    normalization_scale: float,
    normalization_minimum: float,
    expected_protocol: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    returns = _numeric_list(arm, "returns")
    lengths = _integer_list(arm, "lengths")
    environment_seeds = _integer_list(arm, environment_seed_key)
    count = len(returns)
    if len(lengths) != count or len(environment_seeds) != count:
        raise AggregationError("external arm return/length/environment-seed counts differ")
    if len(set(environment_seeds)) != count:
        raise AggregationError("external environment seed array contains duplicates")
    action_noise_seeds: Optional[List[int]] = None
    if action_noise_seed_key is not None:
        action_noise_seeds = _integer_list(arm, action_noise_seed_key)
        if len(action_noise_seeds) != count or len(set(action_noise_seeds)) != count:
            raise AggregationError("external action-noise seed array invalid")
    if count != int(expected_protocol["episode_count"]):
        raise AggregationError("external episode count differs from frozen protocol")
    expected_environment_seeds = _external_expected_seeds(
        expected_protocol, "environment_seeds", "environment_seed_start"
    )
    expected_noise_seeds = _external_expected_seeds(
        expected_protocol, "action_noise_seeds", "action_noise_seed_start"
    )
    if environment_seeds != expected_environment_seeds:
        raise AggregationError("external environment seeds differ from frozen protocol")
    if action_noise_seeds != expected_noise_seeds:
        raise AggregationError("external action-noise seeds differ from frozen protocol")
    if not _close(action_noise_beta, float(expected_protocol["rollout_beta"])):
        raise AggregationError("external rollout beta differs from frozen protocol")
    return_mean = statistics.fmean(returns)
    return_std = statistics.pstdev(returns)
    normalized = [
        (value - normalization_minimum) * normalization_scale for value in returns
    ]
    normalized_mean = statistics.fmean(normalized)
    normalized_std = statistics.pstdev(normalized)
    for field, recomputed in (
        ("return_mean", return_mean),
        ("return_std", return_std),
        ("normalized_score_mean", normalized_mean),
        ("normalized_score_std", normalized_std),
    ):
        reported = arm.get(field)
        if reported is None or not _close(float(reported), recomputed):
            raise AggregationError(f"external reported {field} differs from raw returns")
    public = {
        "episode_count": count,
        "return_mean": return_mean,
        "return_population_std": return_std,
        "normalized_score_mean": normalized_mean,
        "normalized_score_population_std": normalized_std,
        "environment_seed_first": min(environment_seeds),
        "environment_seed_last": max(environment_seeds),
        "action_noise_seed_first": (
            min(action_noise_seeds) if action_noise_seeds is not None else None
        ),
        "action_noise_seed_last": (
            max(action_noise_seeds) if action_noise_seeds is not None else None
        ),
        "action_noise_beta": float(action_noise_beta),
        "total_environment_steps": sum(lengths),
        "actuator_clip_fraction": arm.get(
            "action_clip_fraction", arm.get("actuator_clip_fraction")
        ),
        "command_at_bound_fraction": arm.get(
            "transformed_command_at_bound_fraction",
            arm.get("command_at_bound_fraction"),
        ),
        "inverse_saturation_fraction": arm.get(
            "command_transform_saturation_fraction",
            arm.get("inverse_saturation_fraction"),
        ),
        "executed_commanded_action_mse": arm.get("executed_commanded_action_mse"),
        "proposed_residual_abs_mean": arm.get("proposed_residual_abs_mean"),
        "proposed_residual_abs_max": arm.get("proposed_residual_abs_max"),
        "applied_residual_abs_mean": arm.get("applied_residual_abs_mean"),
    }
    private = {
        "returns": returns,
        "environment_seeds": environment_seeds,
        "action_noise_seeds": action_noise_seeds,
        "action_noise_beta": float(action_noise_beta),
    }
    return public, private


def _external_checkpoint_sources(
    *,
    checkpoint_path_value: Any,
    raw_path: Path,
    expected_artifacts: Mapping[str, Any],
) -> Tuple[Path, Mapping[str, Any], List[Dict[str, Any]]]:
    if not isinstance(checkpoint_path_value, str) or not checkpoint_path_value:
        raise AggregationError("external raw file lacks checkpoint path")
    checkpoint_path = Path(checkpoint_path_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (raw_path.parent / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise AggregationError(f"external checkpoint is missing: {checkpoint_path}")
    checkpoint_source = _path_source(checkpoint_path, "external_checkpoint")
    if checkpoint_source.get("sha256") != expected_artifacts["checkpoint_sha256"]:
        raise AggregationError("external checkpoint SHA differs from frozen manifest")
    config_path = checkpoint_path.parent / "config.json"
    training_config, config_source = _read_json(config_path)
    config_source["role"] = "external_checkpoint_config"
    if config_source["sha256"] != expected_artifacts["checkpoint_config_sha256"]:
        raise AggregationError("external checkpoint config SHA differs from frozen manifest")
    return checkpoint_path, training_config, [checkpoint_source, config_source]


def _parse_external_evaluation(
    results_root: Path,
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    schema = spec["raw_schema"]
    path = _resolve_results_path(results_root, spec["path"], "external evaluation path")
    raw, source = _read_json(path)
    source["role"] = f"external_evaluation:{spec['run_id']}"
    artifacts = spec["expected_artifacts"]
    if schema == "evaluate_td3bc_v1" and source["sha256"] != artifacts[
        "raw_file_sha256"
    ]:
        raise AggregationError("external raw evaluation SHA differs from frozen manifest")
    if raw.get("status") != "complete":
        raise AggregationError(f"external evaluation is not complete: {path}")
    expected_config = spec["expected_config"]
    expected_implementation = spec["expected_implementation"]
    environment = raw.get("environment")
    scale, normalization = _normalization_for(manifest, environment)
    minimum = normalization["reference_min_score"]

    if schema == "evaluate_td3bc_v1":
        checkpoint_path_value = raw.get("checkpoint")
        if raw.get("checkpoint_sha256") != artifacts["checkpoint_sha256"]:
            raise AggregationError("external raw/checkpoint SHA linkage mismatch")
        _, training_config, checkpoint_sources = _external_checkpoint_sources(
            checkpoint_path_value=checkpoint_path_value,
            raw_path=path,
            expected_artifacts=artifacts,
        )
        arguments = training_config.get("arguments", {})
        actual_config = {
            "command_transform": raw.get("command_transform"),
            "command_scale": raw.get("command_scale"),
            "command_transform_beta": raw.get("command_transform_beta"),
            "checkpoint_variant": arguments.get("variant"),
            "checkpoint_action_pairing": arguments.get("action_pairing"),
            "checkpoint_train_seed": arguments.get("train_seed"),
            "checkpoint_updates": arguments.get("updates"),
        }
        if actual_config != expected_config:
            raise AggregationError("external TD3BC controller/checkpoint config mismatch")
        if int(spec["training_seed"]) != arguments.get("train_seed"):
            raise AggregationError("external training seed differs from checkpoint config")
        if raw.get("implementation") != expected_implementation:
            raise AggregationError("external TD3BC implementation hashes mismatch")
        public_evaluations = []
        private_evaluations: Dict[str, Dict[str, Any]] = {}
        for evaluation_spec in spec["evaluations"]:
            raw_arm_name = evaluation_spec["raw_arm"]
            arm = raw.get(raw_arm_name)
            if not isinstance(arm, Mapping):
                raise AggregationError(f"external raw arm {raw_arm_name!r} is missing")
            protocol = evaluation_spec["expected_protocol"]
            if environment != protocol["environment"]:
                raise AggregationError("external environment differs from frozen protocol")
            if raw_arm_name == "persistent_action_noise":
                environment_seed_key = "environment_seeds"
                action_noise_seed_key = "action_noise_seeds"
                action_noise_beta = float(arm.get("action_noise_beta"))
            else:
                environment_seed_key = "episode_seeds"
                action_noise_seed_key = None
                action_noise_beta = 0.0
            public_arm, private_arm = _summarize_external_arm(
                arm,
                environment_seed_key=environment_seed_key,
                action_noise_seed_key=action_noise_seed_key,
                action_noise_beta=action_noise_beta,
                normalization_scale=scale,
                normalization_minimum=minimum,
                expected_protocol=protocol,
            )
            evaluation_id = evaluation_spec["evaluation_id"]
            public_evaluations.append(
                {
                    "evaluation_id": evaluation_id,
                    "evidence_label": "confirmation",
                    "selection_note": evaluation_spec.get("selection_note"),
                    "environment": environment,
                    "status": "complete",
                    "raw_arm": raw_arm_name,
                    "environment_rollout_beta": action_noise_beta,
                    "wall_time_seconds": raw.get("wall_time_seconds"),
                    "normalization": normalization,
                    "arms": {"external": public_arm},
                    "raw_source": {
                        **source,
                        "json_pointer": f"/{raw_arm_name}/returns",
                    },
                }
            )
            private_evaluations[evaluation_id] = {
                "environment": environment,
                "normalization_scale": scale,
                "training_seed": int(spec["training_seed"]),
                "base_checkpoint_sha256": artifacts["checkpoint_sha256"],
                "arms": {"external": private_arm},
            }
        public_cost = {
            "wall_time_seconds": raw.get("wall_time_seconds"),
            "environment_steps": sum(
                evaluation["arms"]["external"]["total_environment_steps"]
                for evaluation in public_evaluations
            ),
            "critic_forward_rows": 0,
            "critic_backward_rows": 0,
            "cost_scope": "rollout_only_no_online_critic",
        }
        sources = [source, *checkpoint_sources]
    else:
        if raw.get("raw_schema") != "channel_opex_v1":
            raise AggregationError("external OPEX raw schema marker mismatch")
        base = raw.get("base_checkpoint")
        if not isinstance(base, Mapping):
            raise AggregationError("external OPEX lacks base_checkpoint mapping")
        if base.get("sha256") != artifacts["checkpoint_sha256"]:
            raise AggregationError("external OPEX base checkpoint linkage mismatch")
        calibration = raw.get("calibration")
        if not isinstance(calibration, Mapping) or calibration.get(
            "calibration_sha256"
        ) != artifacts["calibration_sha256"]:
            raise AggregationError("external OPEX calibration SHA linkage mismatch")
        _, training_config, checkpoint_sources = _external_checkpoint_sources(
            checkpoint_path_value=base.get("path"),
            raw_path=path,
            expected_artifacts=artifacts,
        )
        if training_config.get("arguments", {}).get("train_seed") != int(
            spec["training_seed"]
        ):
            raise AggregationError("external OPEX training seed differs from base config")
        if raw.get("controller") != expected_config:
            raise AggregationError("external OPEX controller config mismatch")
        if raw.get("method_id") != spec["method_id"]:
            raise AggregationError("external OPEX method_id mismatch")
        if raw.get("implementation") != expected_implementation:
            raise AggregationError("external OPEX implementation hashes mismatch")
        if len(spec["evaluations"]) != 1:
            raise AggregationError("external OPEX requires exactly one paired evaluation")
        evaluation_spec = spec["evaluations"][0]
        protocol = evaluation_spec["expected_protocol"]
        if environment != protocol["environment"]:
            raise AggregationError("external OPEX environment differs from frozen protocol")
        raw_protocol = raw.get("evaluation_protocol")
        if not isinstance(raw_protocol, Mapping):
            raise AggregationError("external OPEX lacks evaluation_protocol mapping")
        for field in (
            "environment",
            "episode_count_per_arm",
            "environment_seed_start",
            "action_noise_seed_start",
            "gradient_noise_seed_start",
            *OPEX_EXTERNAL_PROTOCOL_METADATA_FIELDS,
        ):
            expected_field = (
                "episode_count" if field == "episode_count_per_arm" else field
            )
            if raw_protocol.get(field) != protocol[expected_field]:
                raise AggregationError(
                    f"external OPEX evaluation protocol {field} mismatch"
                )
        raw_channel = raw.get("channel")
        if not isinstance(raw_channel, Mapping):
            raise AggregationError("external OPEX lacks channel mapping")
        expected_beta_mismatch = not _close(
            float(expected_config["model_beta"]), float(protocol["rollout_beta"])
        )
        expected_channel = {
            "gradient_model": "iid_uniform_additive_then_clip",
            "model_or_calibration_beta": float(expected_config["model_beta"]),
            "environment_rollout_beta": float(protocol["rollout_beta"]),
            "beta_mismatch": expected_beta_mismatch,
        }
        if dict(raw_channel) != expected_channel:
            raise AggregationError("external OPEX channel/model beta protocol mismatch")
        expected_raw_normalization = {
            "reference_min_score": float(normalization["reference_min_score"]),
            "reference_max_score": float(normalization["reference_max_score"]),
        }
        if raw.get("normalization") != expected_raw_normalization:
            raise AggregationError("external OPEX normalization protocol mismatch")
        public_arms: Dict[str, Dict[str, Any]] = {}
        private_arms: Dict[str, Dict[str, Any]] = {}
        for arm_name in ("baseline_only", "adapted"):
            arm = raw.get("arms", {}).get(arm_name)
            if not isinstance(arm, Mapping):
                raise AggregationError(f"external OPEX lacks {arm_name} arm")
            public_arm, private_arm = _summarize_external_arm(
                arm,
                environment_seed_key="environment_seeds",
                action_noise_seed_key="action_noise_seeds",
                action_noise_beta=float(arm.get("action_noise_beta")),
                normalization_scale=scale,
                normalization_minimum=minimum,
                expected_protocol=protocol,
            )
            public_arms[arm_name] = public_arm
            private_arms[arm_name] = private_arm
        expected_gradient_seeds = _external_expected_seeds(
            protocol, "gradient_noise_seeds", "gradient_noise_seed_start"
        )
        for arm_name in ("baseline_only", "adapted"):
            gradient_seeds = _integer_list(
                raw["arms"][arm_name], "gradient_noise_seeds"
            )
            if gradient_seeds != expected_gradient_seeds:
                raise AggregationError(
                    f"external OPEX {arm_name} gradient-noise seeds mismatch"
                )
        stored = raw.get("paired", {}).get("adapted_minus_baseline_returns")
        computed = [
            left - right
            for left, right in zip(
                private_arms["adapted"]["returns"],
                private_arms["baseline_only"]["returns"],
            )
        ]
        if not isinstance(stored, list) or len(stored) != len(computed) or any(
            not _close(float(left), right) for left, right in zip(stored, computed)
        ):
            raise AggregationError("external OPEX stored paired differences mismatch")
        cost = raw.get("cost")
        cost = _require_exact_keys(
            cost,
            (
                "cost_scope",
                "environment_steps",
                "q1_forward_rows",
                "q1_backward_rows",
                "q1_backward_calls",
                "base_actor_rows",
                "baseline_environment_steps",
                "baseline_base_actor_rows",
                "paired_evaluation_environment_steps",
                "baseline_q1_rows_total",
                "adapted_q1_rows_total",
                "adapted_q1_rows_per_environment_step",
                "adapted_backward_calls",
                "wall_time_seconds",
                "baseline_wall_time_seconds",
                "paired_evaluation_wall_time_seconds",
            ),
            location="external OPEX cost",
        )
        adapted_steps = public_arms["adapted"]["total_environment_steps"]
        baseline_steps = public_arms["baseline_only"]["total_environment_steps"]
        total_steps = adapted_steps + baseline_steps
        expected_cost = spec["expected_cost"]
        cost_checks = {
            "environment_steps": adapted_steps,
            "base_actor_rows": adapted_steps
            * int(expected_cost["base_actor_rows_per_adapted_environment_step"]),
            "baseline_environment_steps": baseline_steps,
            "baseline_base_actor_rows": baseline_steps,
            "paired_evaluation_environment_steps": total_steps,
            "baseline_q1_rows_total": 0,
            "adapted_q1_rows_total": adapted_steps
            * int(expected_cost["q1_rows_per_adapted_step"]),
            "adapted_q1_rows_per_environment_step": int(
                expected_cost["q1_rows_per_adapted_step"]
            ),
            "q1_forward_rows": adapted_steps
            * int(expected_cost["q1_rows_per_adapted_step"]),
            "q1_backward_rows": adapted_steps
            * int(expected_cost["q1_rows_per_adapted_step"]),
            "q1_backward_calls": adapted_steps
            * int(expected_cost["q1_backward_calls_per_adapted_step"]),
            "adapted_backward_calls": adapted_steps
            * int(expected_cost["q1_backward_calls_per_adapted_step"]),
        }
        for field, expected_value in cost_checks.items():
            if cost.get(field) != expected_value:
                raise AggregationError(f"external OPEX cost {field} mismatch")
        if cost.get("cost_scope", expected_cost["cost_scope"]) != expected_cost[
            "cost_scope"
        ]:
            raise AggregationError("external OPEX cost scope mismatch")
        for field in (
            "wall_time_seconds",
            "baseline_wall_time_seconds",
            "paired_evaluation_wall_time_seconds",
        ):
            if not isinstance(cost.get(field), (int, float)) or not math.isfinite(
                float(cost[field])
            ) or float(cost[field]) < 0.0:
                raise AggregationError(f"external OPEX {field} cost invalid")
        if float(cost["paired_evaluation_wall_time_seconds"]) + 1e-12 < (
            float(cost["wall_time_seconds"])
            + float(cost["baseline_wall_time_seconds"])
        ):
            raise AggregationError(
                "external OPEX paired wall time is smaller than its two rollout arms"
            )
        if not _close(
            float(raw.get("wall_time_seconds")),
            float(cost["paired_evaluation_wall_time_seconds"]),
        ):
            raise AggregationError(
                "external OPEX top-level wall time differs from paired cost"
            )
        evaluation_id = evaluation_spec["evaluation_id"]
        public_evaluations = [
            {
                "evaluation_id": evaluation_id,
                "evidence_label": "confirmation",
                "selection_note": evaluation_spec.get("selection_note"),
                "environment": environment,
                "status": "complete",
                "raw_arm": "paired_controller",
                "environment_rollout_beta": float(protocol["rollout_beta"]),
                "wall_time_seconds": cost.get("wall_time_seconds"),
                "normalization": normalization,
                "arms": public_arms,
                "raw_source": {
                    **source,
                    "json_pointers": {
                        "baseline_returns": "/arms/baseline_only/returns",
                        "adapted_returns": "/arms/adapted/returns",
                    },
                },
            }
        ]
        private_evaluations = {
            evaluation_id: {
                "environment": environment,
                "normalization_scale": scale,
                "training_seed": int(spec["training_seed"]),
                "base_checkpoint_sha256": artifacts["checkpoint_sha256"],
                "arms": private_arms,
            }
        }
        public_cost = {**cost, "cost_scope": expected_cost["cost_scope"]}
        sources = [source, *checkpoint_sources]

    seed_only_config_fields = (
        {"checkpoint_train_seed"}
        if schema == "evaluate_td3bc_v1"
        else {"gradient_noise_seed"}
    )
    method_config = {
        "raw_schema": schema,
        **{
            key: value
            for key, value in expected_config.items()
            if key not in seed_only_config_fields
        },
    }
    return (
        {
            "run_id": spec["run_id"],
            "method_id": spec["method_id"],
            "source_kind": "external_evaluation",
            "raw_schema": schema,
            "evidence_label": "confirmation",
            "selection_note": spec.get("selection_note"),
            "development_selection": spec.get("development_selection"),
            "training_seed": int(spec["training_seed"]),
            "observed_status": "complete",
            "checkpoint_sha256": artifacts["checkpoint_sha256"],
            "fail_closed_validation": {
                "status": "verified",
                "raw_file_sha256": source["sha256"],
                "raw_file_hash_was_predeclared": schema == "evaluate_td3bc_v1",
                "checkpoint_config_sha256": artifacts["checkpoint_config_sha256"],
                "method_config_sha256": sha256_bytes(
                    canonical_json(method_config).encode("utf-8")
                ),
            },
            "controller_config": expected_config,
            "cost": public_cost,
            "evaluations": public_evaluations,
            "sources": sources,
        },
        private_evaluations,
        sources,
    )


def _endpoint(
    endpoint: Mapping[str, Any],
    private_runs: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        run_id = endpoint["run_id"]
        evaluation_id = endpoint["evaluation_id"]
        arm = endpoint["arm"]
    except KeyError as exc:
        raise AggregationError("comparison endpoint requires run_id/evaluation_id/arm") from exc
    if arm not in ARMS:
        raise AggregationError(f"unsupported comparison arm {arm!r}")
    try:
        evaluation = private_runs[run_id][evaluation_id]
    except KeyError as exc:
        raise AggregationError(
            f"comparison endpoint not found: {run_id}/{evaluation_id}"
        ) from exc
    return {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "arm": arm,
    }, {
        "environment": evaluation["environment"],
        "normalization_scale": evaluation["normalization_scale"],
        "training_seed": evaluation.get("training_seed"),
        "base_checkpoint_sha256": evaluation.get("base_checkpoint_sha256"),
        **evaluation["arms"][arm],
    }


def _build_comparison(
    spec: Mapping[str, Any],
    private_runs: Mapping[str, Mapping[str, Any]],
    run_labels: Mapping[str, str],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    comparison_id = spec.get("comparison_id")
    evidence_label = spec.get("evidence_label")
    if not isinstance(comparison_id, str) or not comparison_id:
        raise AggregationError("comparison requires comparison_id")
    if evidence_label not in EVIDENCE_LABELS - {"implementation_smoke"}:
        raise AggregationError("comparison evidence_label must be development or confirmation")
    target_public, target = _endpoint(spec.get("target", {}), private_runs)
    reference_public, reference = _endpoint(spec.get("reference", {}), private_runs)
    if run_labels[target_public["run_id"]] != evidence_label or run_labels[
        reference_public["run_id"]
    ] != evidence_label:
        raise AggregationError("comparison evidence label differs from constituent run")
    if target["environment"] != reference["environment"]:
        raise AggregationError("paired comparison environments differ")
    if target["environment_seeds"] != reference["environment_seeds"]:
        raise AggregationError("paired comparison environment seed arrays differ")
    if target["action_noise_beta"] != 0.0 or reference["action_noise_beta"] != 0.0:
        if target["action_noise_seeds"] != reference["action_noise_seeds"]:
            raise AggregationError("paired comparison action-noise seed arrays differ")
    if target["action_noise_beta"] != reference["action_noise_beta"]:
        raise AggregationError("paired comparison rollout betas differ")
    if target["normalization_scale"] != reference["normalization_scale"]:
        raise AggregationError("paired comparison normalizations differ")
    if spec.get("require_same_base_checkpoint") is True:
        if target["training_seed"] != reference["training_seed"]:
            raise AggregationError(
                "same-base mechanism comparison has different training seeds"
            )
        if not target["base_checkpoint_sha256"] or (
            target["base_checkpoint_sha256"]
            != reference["base_checkpoint_sha256"]
        ):
            raise AggregationError(
                "same-base mechanism comparison has different base checkpoint SHA"
            )
    elif spec.get("require_same_base_checkpoint") is False:
        if target["training_seed"] != reference["training_seed"]:
            raise AggregationError(
                "different-policy comparison has different training seeds"
            )
        if target["base_checkpoint_sha256"] != spec.get(
            "expected_target_base_checkpoint_sha256"
        ) or reference["base_checkpoint_sha256"] != spec.get(
            "expected_reference_base_checkpoint_sha256"
        ):
            raise AggregationError(
                "different-policy comparison checkpoint SHA differs from frozen relation"
            )
    differences = [
        left - right for left, right in zip(target["returns"], reference["returns"])
    ]
    result = paired_statistics(
        differences,
        normalized_scale=target["normalization_scale"],
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "comparison_id": comparison_id,
        "evidence_label": evidence_label,
        "training_seed": target["training_seed"],
        "is_primary_confirmation_comparison": (
            comparison_id in PRIMARY_CONFIRMATION_COMPARISON_IDS
        ),
        "selection_note": spec.get("selection_note"),
        "require_same_base_checkpoint": spec.get("require_same_base_checkpoint"),
        "comparison_class": spec.get("comparison_class"),
        "target": target_public,
        "reference": reference_public,
        "environment": target["environment"],
        "status": "complete",
        **result,
    }


def _build_baseline_equivalence_check(
    spec: Mapping[str, Any],
    private_runs: Mapping[str, Mapping[str, Any]],
    run_labels: Mapping[str, str],
) -> Dict[str, Any]:
    resolved = [_endpoint(endpoint, private_runs) for endpoint in spec["endpoints"]]
    public_endpoints = [item[0] for item in resolved]
    private_endpoints = [item[1] for item in resolved]
    reference = private_endpoints[0]
    expected_beta = float(spec["expected_rollout_beta"])
    expected_base = spec["expected_base_checkpoint_sha256"]
    for public, endpoint in zip(public_endpoints, private_endpoints):
        if run_labels[public["run_id"]] != spec["evidence_label"]:
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} evidence label mismatch"
            )
        if endpoint["base_checkpoint_sha256"] != expected_base:
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} base checkpoint mismatch"
            )
        if not _close(float(endpoint["action_noise_beta"]), expected_beta):
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} rollout beta mismatch"
            )
        if endpoint["environment"] != reference["environment"] or endpoint[
            "environment_seeds"
        ] != reference["environment_seeds"] or endpoint[
            "action_noise_seeds"
        ] != reference["action_noise_seeds"]:
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} seed/protocol mismatch"
            )
        if endpoint["training_seed"] != reference["training_seed"]:
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} training seed mismatch"
            )
        if endpoint["returns"] != reference["returns"]:
            raise AggregationError(
                f"baseline equivalence {spec['check_id']} return arrays differ"
            )
    return {
        "check_id": spec["check_id"],
        "evidence_label": spec["evidence_label"],
        "expected_baseline_semantics": spec["expected_baseline_semantics"],
        "expected_rollout_beta": expected_beta,
        "expected_base_checkpoint_sha256": expected_base,
        "status": "verified_elementwise_identical",
        "episode_count": len(reference["returns"]),
        "endpoints": public_endpoints,
        "statistical_test": None,
        "note": (
            "Deterministic cross-file integrity invariant; identical episodes are "
            "not additional training repeats and no significance test is applicable."
        ),
    }


def _training_seed_evidence(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["observed_status"] == "complete" and run["evidence_label"] != "implementation_smoke":
            groups[(run["method_id"], run["evidence_label"])].append(run)
    result = []
    for method_id, evidence_label in sorted(groups):
        rows = groups[(method_id, evidence_label)]
        external_only = all(
            row.get("source_kind") == "external_evaluation" for row in rows
        )
        seeds = [row["training_seed"] for row in rows]
        duplicate = len(set(seeds)) != len(seeds)
        per_training_seed = []
        for row in sorted(rows, key=lambda item: (item["training_seed"], item["run_id"])):
            evaluation_values = []
            for evaluation in row.get("evaluations", []):
                evaluation_values.append(
                    {
                        "evaluation_id": evaluation.get("evaluation_id"),
                        "evidence_label": evaluation.get("evidence_label"),
                        "episode_count_per_arm": {
                            arm_name: arm.get("episode_count")
                            for arm_name, arm in sorted(evaluation.get("arms", {}).items())
                        },
                        "return_mean_per_arm": {
                            arm_name: arm.get("return_mean")
                            for arm_name, arm in sorted(evaluation.get("arms", {}).items())
                        },
                        "normalized_score_mean_per_arm": {
                            arm_name: arm.get("normalized_score_mean")
                            for arm_name, arm in sorted(evaluation.get("arms", {}).items())
                        },
                    }
                )
            per_training_seed.append(
                {
                    "training_seed": row["training_seed"],
                    "run_id": row["run_id"],
                    "evaluations": evaluation_values,
                }
            )
        evaluation_arm_values: MutableMapping[
            Tuple[str, str], List[Dict[str, Any]]
        ] = defaultdict(list)
        for seed_record in per_training_seed:
            for evaluation in seed_record["evaluations"]:
                evaluation_id = evaluation["evaluation_id"]
                for arm_name, return_mean in evaluation[
                    "return_mean_per_arm"
                ].items():
                    evaluation_arm_values[(evaluation_id, arm_name)].append(
                        {
                            "training_seed": seed_record["training_seed"],
                            "run_id": seed_record["run_id"],
                            "return_mean": return_mean,
                            "normalized_score_mean": evaluation[
                                "normalized_score_mean_per_arm"
                            ][arm_name],
                        }
                    )
        descriptive_summaries = []
        for (evaluation_id, arm_name), values in sorted(
            evaluation_arm_values.items()
        ):
            values = sorted(values, key=lambda item: item["training_seed"])
            unique_value_seeds = {item["training_seed"] for item in values}
            complete = (
                len(values) == len(unique_value_seeds)
                and all(item["return_mean"] is not None for item in values)
                and all(item["normalized_score_mean"] is not None for item in values)
            )
            descriptive_summaries.append(
                {
                    "evaluation_id": evaluation_id,
                    "arm": arm_name,
                    "statistical_unit": (
                        "base_policy_checkpoint_seed"
                        if external_only
                        else "full_pipeline_training_seed"
                    ),
                    "n_training_seeds": len(unique_value_seeds),
                    "training_seeds": sorted(unique_value_seeds),
                    "per_training_seed_values": values,
                    "descriptive_return_mean_across_training_seeds": (
                        statistics.fmean(item["return_mean"] for item in values)
                        if complete
                        else None
                    ),
                    "descriptive_normalized_score_mean_across_training_seeds": (
                        statistics.fmean(
                            item["normalized_score_mean"] for item in values
                        )
                        if complete
                        else None
                    ),
                    "training_seed_confidence_interval": None,
                    "note": (
                        "Descriptive mean of one rollout mean per independent "
                        "training-seed checkpoint; episode returns are not pooled."
                    ),
                }
            )
        result.append(
            {
                "method_id": method_id,
                "evidence_label": evidence_label,
                "statistical_unit": (
                    "base_policy_checkpoint_seed"
                    if external_only
                    else "full_pipeline_training_seed"
                ),
                "training_seeds": seeds,
                "n_training_seeds": len(set(seeds)),
                "duplicate_seed_run": duplicate,
                "can_estimate_training_seed_variability": (
                    not external_only
                    and len(set(seeds)) >= 3
                    and not duplicate
                ),
                "has_multiple_descriptive_checkpoint_seeds": (
                    len(set(seeds)) >= 2 and not duplicate
                ),
                "can_estimate_base_checkpoint_seed_variability": (
                    external_only and len(set(seeds)) >= 2 and not duplicate
                ),
                "training_seed_confidence_interval": None,
                "per_training_seed_values": per_training_seed,
                "evaluation_arm_descriptive_means": descriptive_summaries,
                "note": (
                    (
                        "These are base-policy checkpoint seeds, not independently "
                        "trained post-hoc-controller pipelines. "
                        if external_only
                        else ""
                    )
                    + "Episode-level confidence intervals condition on these "
                    "checkpoints and do not increase the checkpoint-seed count."
                ),
            }
        )
    return result


def _primary_comparison_training_seed_evidence(
    comparisons: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize the six frozen primary comparisons at checkpoint-seed level."""

    by_id = {item["comparison_id"]: item for item in comparisons}
    if not all(
        comparison_id in by_id
        for comparison_id in PRIMARY_CONFIRMATION_COMPARISON_IDS
    ):
        return []
    result = []
    for group_id, comparison_ids in PRIMARY_CONFIRMATION_COMPARISON_GROUPS:
        items = [by_id[comparison_id] for comparison_id in comparison_ids]
        training_seeds = [item["training_seed"] for item in items]
        if training_seeds != [1, 10] or len(set(training_seeds)) != 2:
            raise AggregationError(
                f"primary comparison group {group_id!r} must retain seeds [1, 10]"
            )
        per_seed = [
            {
                "training_seed": item["training_seed"],
                "comparison_id": item["comparison_id"],
                "episode_pairs": item["episode_pairs"],
                "raw_mean_difference": item["raw"]["mean_difference"],
                "normalized_mean_difference": item["normalized"][
                    "mean_difference"
                ],
            }
            for item in items
        ]
        result.append(
            {
                "comparison_group_id": group_id,
                "comparison_ids": list(comparison_ids),
                "evidence_label": "confirmation",
                "statistical_unit": "base_policy_checkpoint_seed",
                "training_seeds": training_seeds,
                "n_training_seeds": 2,
                "per_training_seed_values": per_seed,
                "descriptive_raw_mean_difference_across_training_seeds": (
                    statistics.fmean(
                        item["raw_mean_difference"] for item in per_seed
                    )
                ),
                "descriptive_normalized_mean_difference_across_training_seeds": (
                    statistics.fmean(
                        item["normalized_mean_difference"] for item in per_seed
                    )
                ),
                "training_seed_confidence_interval": None,
                "episode_returns_pooled_for_training_seed_inference": False,
                "checkpoint_seeds": training_seeds,
                "n_checkpoint_pipelines": 2,
                "per_checkpoint_mean_differences": per_seed,
                "mean_of_checkpoint_mean_differences": {
                    "raw": statistics.fmean(
                        item["raw_mean_difference"] for item in per_seed
                    ),
                    "normalized": statistics.fmean(
                        item["normalized_mean_difference"] for item in per_seed
                    ),
                },
                "positive_checkpoint_count": sum(
                    item["raw_mean_difference"] > 0.0 for item in per_seed
                ),
                "training_seed_ci": None,
                "inference_note": (
                    "Two base-policy checkpoint seeds only. Each checkpoint reuses "
                    "the same 50 predeclared rollout-case seed pairs; these are not "
                    "100 independent training/pipeline repeats. Descriptive only."
                ),
                "note": (
                    "n_train=2 descriptive mean of the two checkpoint-level paired "
                    "differences. The 50 paired episodes per checkpoint are not "
                    "pooled into 100 training repeats, and no training-seed CI or "
                    "significance claim is made."
                ),
            }
        )
    return result


def build_report(
    manifest_path: Path,
    *,
    results_root_override: Optional[Path] = None,
    bootstrap_seed_override: Optional[int] = None,
    bootstrap_samples_override: Optional[int] = None,
) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest, manifest_source = _read_json(manifest_path)
    _validate_manifest_has_no_results(manifest)
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in SUPPORTED_MANIFEST_SCHEMAS:
        raise AggregationError("unsupported manifest schema_version")
    strict = manifest_schema == STRICT_MANIFEST_SCHEMA_VERSION
    if strict:
        validate_strict_manifest_definition(manifest, allow_placeholders=False)
    root_spec = manifest.get("results_root")
    if results_root_override is None:
        if not isinstance(root_spec, str) or not root_spec:
            raise AggregationError("manifest requires results_root")
        results_root = (manifest_path.parent / root_spec).resolve()
    else:
        results_root = results_root_override.resolve()
    bootstrap = manifest.get("bootstrap", {})
    seed = int(
        bootstrap_seed_override
        if bootstrap_seed_override is not None
        else bootstrap.get("seed", 20260914)
    )
    samples = int(
        bootstrap_samples_override
        if bootstrap_samples_override is not None
        else bootstrap.get("samples", 10000)
    )
    if samples <= 0:
        raise AggregationError("bootstrap samples must be positive")
    run_specs = manifest.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise AggregationError("manifest runs must be a non-empty list")
    public_runs: List[Dict[str, Any]] = []
    public_external_evaluations: List[Dict[str, Any]] = []
    private_runs: Dict[str, Dict[str, Any]] = {}
    sources = [{**manifest_source, "role": "manifest"}]
    run_labels: Dict[str, str] = {}
    for spec in run_specs:
        if not isinstance(spec, Mapping):
            raise AggregationError("run spec must be an object")
        run_id = spec.get("run_id")
        if run_id in private_runs:
            raise AggregationError(f"duplicate run_id {run_id!r}")
        public, private, run_sources = _parse_run(results_root, spec, manifest)
        public_runs.append(public)
        private_runs[public["run_id"]] = private
        run_labels[public["run_id"]] = public["evidence_label"]
        sources.extend(run_sources)
    for spec in manifest.get("external_evaluations", []):
        if not strict:
            raise AggregationError(
                "external_evaluations are supported only by fail-closed manifest v2"
            )
        public, private, external_sources = _parse_external_evaluation(
            results_root, spec, manifest
        )
        if public["run_id"] in private_runs:
            raise AggregationError(f"duplicate external run_id {public['run_id']!r}")
        public_external_evaluations.append(public)
        private_runs[public["run_id"]] = private
        run_labels[public["run_id"]] = public["evidence_label"]
        sources.extend(external_sources)
    if strict:
        method_to_config: Dict[str, str] = {}
        config_to_method: Dict[str, str] = {}
        for run in public_runs:
            if run["observed_status"] != "complete":
                continue
            method_id = run["method_id"]
            config_sha = run["fail_closed_validation"]["method_config_sha256"]
            previous_config = method_to_config.setdefault(method_id, config_sha)
            if previous_config != config_sha:
                raise AggregationError(
                    f"method_id {method_id!r} maps to multiple expected configs"
                )
            previous_method = config_to_method.setdefault(config_sha, method_id)
            if previous_method != method_id:
                raise AggregationError(
                    f"identical expected configs use two method_ids: "
                    f"{previous_method!r}, {method_id!r}"
                )
    comparisons = []
    seen_comparisons = set()
    for spec in manifest.get("comparisons", []):
        if not isinstance(spec, Mapping):
            raise AggregationError("comparison spec must be an object")
        comparison = _build_comparison(
            spec,
            private_runs,
            run_labels,
            bootstrap_seed=seed,
            bootstrap_samples=samples,
        )
        if comparison["comparison_id"] in seen_comparisons:
            raise AggregationError("duplicate comparison_id")
        seen_comparisons.add(comparison["comparison_id"])
        comparisons.append(comparison)
    baseline_equivalence_checks = [
        _build_baseline_equivalence_check(spec, private_runs, run_labels)
        for spec in manifest.get("baseline_equivalence_checks", [])
    ]
    external_training_seed_evidence = _training_seed_evidence(
        public_external_evaluations
    )
    primary_training_seed_evidence = (
        _primary_comparison_training_seed_evidence(comparisons)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation_status": "complete",
        "manifest": {
            **manifest_source,
            "schema_version": manifest.get("schema_version"),
            "results_root": str(results_root),
        },
        "manifest_validation_mode": (
            "strict_fail_closed_v2" if strict else "legacy_v1_not_fail_closed"
        ),
        "evidence_scope": manifest.get("evidence_scope"),
        "statistical_units": {
            "training": (
                "Independent training seed. A fixed-policy episode is not an "
                "independent training repeat."
            ),
            "paired_interval": (
                "Paired environment/noise-seed episode for two fixed checkpoints; "
                "does not quantify training-seed uncertainty."
            ),
        },
        "bootstrap": {
            "seed": seed,
            "samples": samples,
            "confidence": 0.95,
            "method": "paired_episode_percentile",
        },
        "runs": public_runs,
        "training_seed_evidence": _training_seed_evidence(public_runs),
        "external_evaluations": public_external_evaluations,
        "external_training_seed_evidence": external_training_seed_evidence,
        "paired_comparisons": comparisons,
        "primary_comparison_order": [
            item["comparison_id"]
            for item in comparisons
            if item["comparison_id"] in PRIMARY_CONFIRMATION_COMPARISON_IDS
        ],
        "primary_cross_controller_training_seed_evidence": (
            primary_training_seed_evidence
        ),
        "baseline_equivalence_checks": baseline_equivalence_checks,
        "input_files": sources,
    }


CSV_FIELDS = [
    "row_type",
    "evidence_label",
    "run_id",
    "method_id",
    "training_seed",
    "observed_status",
    "source_kind",
    "raw_schema",
    "checkpoint_sha256",
    "evaluation_id",
    "arm",
    "episode_count",
    "return_mean",
    "normalized_score_mean",
    "actuator_clip_fraction",
    "command_at_bound_fraction",
    "inverse_saturation_fraction",
    "proposed_residual_abs_mean",
    "applied_residual_abs_mean",
    "total_environment_steps",
    "evaluation_base_actor_forward_rows",
    "evaluation_adapter_mlp_forward_rows",
    "evaluation_controller_cost_scope",
    "training_wall_time_seconds",
    "training_wall_time_scope",
    "training_invocation_count",
    "training_preprocessing_seconds_this_invocation",
    "training_preprocessing_time_scope",
    "training_preprocessing_seconds_lifetime",
    "training_preprocessing_seconds_lifetime_available",
    "training_optimizer_seconds",
    "training_optimizer_time_scope",
    "training_q1_input_rows",
    "training_base_actor_precompute_rows_per_invocation",
    "training_base_actor_precompute_rows_lifetime",
    "training_q_scale_q1_forward_rows_per_invocation",
    "training_q_scale_q1_forward_rows_lifetime",
    "training_optimizer_q1_forward_rows_lifetime",
    "training_optimizer_q1_backward_rows_lifetime",
    "training_optimizer_adapter_mlp_forward_rows_lifetime",
    "training_optimizer_adapter_mlp_backward_rows_lifetime",
    "training_total_q1_forward_rows_lifetime",
    "training_frozen_q_scale",
    "training_q_scale_method",
    "training_resume_scope",
    "evaluation_wall_time_seconds",
    "evaluation_wall_time_scope",
    "external_environment_steps",
    "external_baseline_environment_steps",
    "external_paired_evaluation_environment_steps",
    "external_q1_forward_rows",
    "external_q1_backward_rows",
    "external_base_actor_rows",
    "external_baseline_base_actor_rows",
    "external_deployment_wall_time_seconds",
    "external_baseline_wall_time_seconds",
    "external_paired_evaluation_wall_time_seconds",
    "external_cost_scope",
    "audit_id",
    "audit_noise_samples",
    "audit_action_noise_beta",
    "audit_physical_action_rows",
    "audit_individual_critic_network_rows",
    "audit_q1_gain_mean",
    "audit_q1_gain_median",
    "audit_q2_gain_mean",
    "audit_q2_gain_median",
    "audit_min_twin_gain_mean",
    "audit_min_twin_gain_median",
    "audit_baseline_twin_disagreement_mean",
    "audit_baseline_twin_disagreement_median",
    "audit_adapted_twin_disagreement_mean",
    "audit_adapted_twin_disagreement_median",
    "audit_twin_disagreement_change_mean",
    "audit_twin_disagreement_change_median",
    "audit_command_saturation_fraction",
    "audit_proposed_residual_abs_mean",
    "audit_proposed_residual_abs_median",
    "comparison_id",
    "comparison_group_id",
    "is_primary_confirmation_comparison",
    "target",
    "reference",
    "episode_pairs",
    "positive_pairs",
    "raw_mean_difference",
    "raw_t_95ci_low",
    "raw_t_95ci_high",
    "raw_bootstrap_95ci_low",
    "raw_bootstrap_95ci_high",
    "normalized_mean_difference",
    "normalized_t_95ci_low",
    "normalized_t_95ci_high",
    "normalized_bootstrap_95ci_low",
    "normalized_bootstrap_95ci_high",
    "statistical_unit",
    "n_training_seeds",
    "training_seeds",
    "per_training_seed_values",
    "descriptive_return_mean_across_training_seeds",
    "descriptive_normalized_score_mean_across_training_seeds",
    "descriptive_raw_mean_difference_across_training_seeds",
    "descriptive_normalized_mean_difference_across_training_seeds",
    "training_seed_confidence_interval",
    "episode_returns_pooled_for_training_seed_inference",
    "checkpoint_seeds",
    "n_checkpoint_pipelines",
    "per_checkpoint_mean_differences",
    "mean_of_checkpoint_mean_differences",
    "positive_checkpoint_count",
    "training_seed_ci",
    "inference_note",
    "equivalence_check_id",
    "equivalence_semantics",
    "equivalence_episode_count",
    "equivalence_status",
    "equivalence_endpoints",
]


def _endpoint_label(endpoint: Mapping[str, Any]) -> str:
    return f"{endpoint.get('run_id')}/{endpoint.get('evaluation_id')}/{endpoint.get('arm')}"


def tabular_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in report.get("runs", []):
        base = {
            "evidence_label": run.get("evidence_label"),
            "run_id": run.get("run_id"),
            "method_id": run.get("method_id"),
            "training_seed": run.get("training_seed"),
            "observed_status": run.get("observed_status"),
            "training_wall_time_seconds": run.get("training_cost", {}).get(
                "wall_time_seconds"
            ),
            "training_wall_time_scope": run.get("training_cost", {}).get(
                "wall_time_scope"
            ),
            "training_invocation_count": run.get("training_cost", {}).get(
                "invocation_count"
            ),
            "training_preprocessing_seconds_this_invocation": run.get(
                "training_cost", {}
            ).get("preprocessing_seconds_this_invocation"),
            "training_preprocessing_time_scope": run.get("training_cost", {}).get(
                "preprocessing_time_scope"
            ),
            "training_preprocessing_seconds_lifetime": run.get(
                "training_cost", {}
            ).get("preprocessing_seconds_lifetime"),
            "training_preprocessing_seconds_lifetime_available": run.get(
                "training_cost", {}
            ).get("preprocessing_seconds_lifetime_available"),
            "training_optimizer_seconds": run.get("training_cost", {}).get(
                "optimizer_seconds"
            ),
            "training_optimizer_time_scope": run.get("training_cost", {}).get(
                "optimizer_time_scope"
            ),
            "training_q1_input_rows": run.get("training_cost", {}).get(
                "optimizer_q1_forward_rows_lifetime"
            ),
            "training_base_actor_precompute_rows_per_invocation": run.get(
                "training_cost", {}
            ).get("base_actor_precompute_rows_per_invocation"),
            "training_base_actor_precompute_rows_lifetime": run.get(
                "training_cost", {}
            ).get("base_actor_precompute_rows_lifetime"),
            "training_q_scale_q1_forward_rows_per_invocation": run.get(
                "training_cost", {}
            ).get("q_scale_q1_forward_rows_per_invocation"),
            "training_q_scale_q1_forward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("q_scale_q1_forward_rows_lifetime"),
            "training_optimizer_q1_forward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("optimizer_q1_forward_rows_lifetime"),
            "training_optimizer_q1_backward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("optimizer_q1_backward_rows_lifetime"),
            "training_optimizer_adapter_mlp_forward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("optimizer_adapter_mlp_forward_rows_lifetime"),
            "training_optimizer_adapter_mlp_backward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("optimizer_adapter_mlp_backward_rows_lifetime"),
            "training_total_q1_forward_rows_lifetime": run.get(
                "training_cost", {}
            ).get("total_q1_forward_rows_lifetime"),
            "training_frozen_q_scale": run.get("training_cost", {}).get(
                "frozen_q_scale"
            ),
            "training_q_scale_method": run.get("training_cost", {}).get(
                "q_scale_method"
            ),
            "training_resume_scope": run.get("training_cost", {}).get(
                "resume_scope"
            ),
        }
        rows.append({**base, "row_type": "training_run"})
        for evaluation in run.get("evaluations", []):
            for arm_name in sorted(evaluation.get("arms", {})):
                arm = evaluation["arms"][arm_name]
                rows.append(
                    {
                        **base,
                        "row_type": "evaluation_arm",
                        "evidence_label": evaluation.get("evidence_label"),
                        "evaluation_id": evaluation.get("evaluation_id"),
                        "arm": arm_name,
                        "episode_count": arm.get("episode_count"),
                        "return_mean": arm.get("return_mean"),
                        "normalized_score_mean": arm.get("normalized_score_mean"),
                        "actuator_clip_fraction": arm.get("actuator_clip_fraction"),
                        "command_at_bound_fraction": arm.get("command_at_bound_fraction"),
                        "inverse_saturation_fraction": arm.get(
                            "inverse_saturation_fraction"
                        ),
                        "proposed_residual_abs_mean": arm.get(
                            "proposed_residual_abs_mean"
                        ),
                        "applied_residual_abs_mean": arm.get(
                            "applied_residual_abs_mean"
                        ),
                        "total_environment_steps": arm.get("total_environment_steps"),
                        "evaluation_base_actor_forward_rows": arm.get(
                            "base_actor_forward_rows"
                        ),
                        "evaluation_adapter_mlp_forward_rows": arm.get(
                            "adapter_mlp_forward_rows"
                        ),
                        "evaluation_controller_cost_scope": arm.get(
                            "controller_cost_scope"
                        ),
                        "evaluation_wall_time_seconds": evaluation.get(
                            "wall_time_seconds"
                        ),
                        "evaluation_wall_time_scope": evaluation.get(
                            "wall_time_scope"
                        ),
                    }
                )
        for audit in run.get("audits", []):
            rows.append(
                {
                    **base,
                    "row_type": "q_audit",
                    "evidence_label": audit.get("evidence_label"),
                    "audit_id": audit.get("audit_id"),
                    "audit_noise_samples": audit.get("noise_samples"),
                    "audit_action_noise_beta": audit.get("audit_action_noise_beta"),
                    "audit_physical_action_rows": audit.get("physical_action_rows"),
                    "audit_individual_critic_network_rows": audit.get(
                        "individual_critic_network_rows"
                    ),
                    "audit_q1_gain_mean": audit.get("q1_gain", {}).get("mean"),
                    "audit_q1_gain_median": audit.get("q1_gain", {}).get("median"),
                    "audit_q2_gain_mean": audit.get("q2_gain", {}).get("mean"),
                    "audit_q2_gain_median": audit.get("q2_gain", {}).get("median"),
                    "audit_min_twin_gain_mean": audit.get("min_twin_gain", {}).get(
                        "mean"
                    ),
                    "audit_min_twin_gain_median": audit.get(
                        "min_twin_gain", {}
                    ).get("median"),
                    "audit_baseline_twin_disagreement_mean": audit.get(
                        "baseline_twin_disagreement", {}
                    ).get("mean"),
                    "audit_baseline_twin_disagreement_median": audit.get(
                        "baseline_twin_disagreement", {}
                    ).get("median"),
                    "audit_adapted_twin_disagreement_mean": audit.get(
                        "adapted_twin_disagreement", {}
                    ).get("mean"),
                    "audit_adapted_twin_disagreement_median": audit.get(
                        "adapted_twin_disagreement", {}
                    ).get("median"),
                    "audit_twin_disagreement_change_mean": audit.get(
                        "twin_disagreement_change", {}
                    ).get("mean"),
                    "audit_twin_disagreement_change_median": audit.get(
                        "twin_disagreement_change", {}
                    ).get("median"),
                    "audit_command_saturation_fraction": audit.get(
                        "command_saturation_fraction"
                    ),
                    "audit_proposed_residual_abs_mean": audit.get(
                        "proposed_residual_abs", {}
                    ).get("mean"),
                    "audit_proposed_residual_abs_median": audit.get(
                        "proposed_residual_abs", {}
                    ).get("median"),
                }
            )
    for run in report.get("external_evaluations", []):
        cost = run.get("cost", {})
        base = {
            "row_type": "external_evaluation_arm",
            "source_kind": "external_evaluation",
            "raw_schema": run.get("raw_schema"),
            "evidence_label": run.get("evidence_label"),
            "run_id": run.get("run_id"),
            "method_id": run.get("method_id"),
            "training_seed": run.get("training_seed"),
            "observed_status": run.get("observed_status"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "external_environment_steps": cost.get("environment_steps"),
            "external_baseline_environment_steps": cost.get(
                "baseline_environment_steps"
            ),
            "external_paired_evaluation_environment_steps": cost.get(
                "paired_evaluation_environment_steps"
            ),
            "external_q1_forward_rows": cost.get("q1_forward_rows"),
            "external_q1_backward_rows": cost.get("q1_backward_rows"),
            "external_base_actor_rows": cost.get("base_actor_rows"),
            "external_baseline_base_actor_rows": cost.get(
                "baseline_base_actor_rows"
            ),
            "external_deployment_wall_time_seconds": cost.get("wall_time_seconds"),
            "external_baseline_wall_time_seconds": cost.get(
                "baseline_wall_time_seconds"
            ),
            "external_paired_evaluation_wall_time_seconds": cost.get(
                "paired_evaluation_wall_time_seconds"
            ),
            "external_cost_scope": cost.get("cost_scope"),
        }
        for evaluation in run.get("evaluations", []):
            for arm_name, arm in sorted(evaluation.get("arms", {}).items()):
                rows.append(
                    {
                        **base,
                        "evaluation_id": evaluation.get("evaluation_id"),
                        "arm": arm_name,
                        "episode_count": arm.get("episode_count"),
                        "return_mean": arm.get("return_mean"),
                        "normalized_score_mean": arm.get("normalized_score_mean"),
                        "actuator_clip_fraction": arm.get("actuator_clip_fraction"),
                        "command_at_bound_fraction": arm.get(
                            "command_at_bound_fraction"
                        ),
                        "inverse_saturation_fraction": arm.get(
                            "inverse_saturation_fraction"
                        ),
                        "proposed_residual_abs_mean": arm.get(
                            "proposed_residual_abs_mean"
                        ),
                        "applied_residual_abs_mean": arm.get(
                            "applied_residual_abs_mean"
                        ),
                        "total_environment_steps": arm.get(
                            "total_environment_steps"
                        ),
                        "evaluation_wall_time_seconds": evaluation.get(
                            "wall_time_seconds"
                        ),
                    }
                )
    for comparison in report.get("paired_comparisons", []):
        raw = comparison["raw"]
        normalized = comparison["normalized"]
        rows.append(
            {
                "row_type": "paired_comparison",
                "evidence_label": comparison.get("evidence_label"),
                "comparison_id": comparison.get("comparison_id"),
                "training_seed": comparison.get("training_seed"),
                "is_primary_confirmation_comparison": comparison.get(
                    "is_primary_confirmation_comparison"
                ),
                "target": _endpoint_label(comparison["target"]),
                "reference": _endpoint_label(comparison["reference"]),
                "episode_pairs": comparison.get("episode_pairs"),
                "positive_pairs": comparison.get("positive_pairs"),
                "raw_mean_difference": raw.get("mean_difference"),
                "raw_t_95ci_low": raw.get("t_95ci_low"),
                "raw_t_95ci_high": raw.get("t_95ci_high"),
                "raw_bootstrap_95ci_low": raw.get("paired_bootstrap_95ci_low"),
                "raw_bootstrap_95ci_high": raw.get("paired_bootstrap_95ci_high"),
                "normalized_mean_difference": normalized.get("mean_difference"),
                "normalized_t_95ci_low": normalized.get("t_95ci_low"),
                "normalized_t_95ci_high": normalized.get("t_95ci_high"),
                "normalized_bootstrap_95ci_low": normalized.get(
                    "paired_bootstrap_95ci_low"
                ),
                "normalized_bootstrap_95ci_high": normalized.get(
                    "paired_bootstrap_95ci_high"
                ),
            }
        )
    for item in report.get("external_training_seed_evidence", []):
        for summary in item.get("evaluation_arm_descriptive_means", []):
            rows.append(
                {
                    "row_type": "external_training_seed_evidence",
                    "evidence_label": item.get("evidence_label"),
                    "method_id": item.get("method_id"),
                    "evaluation_id": summary.get("evaluation_id"),
                    "arm": summary.get("arm"),
                    "statistical_unit": summary.get("statistical_unit"),
                    "n_training_seeds": summary.get("n_training_seeds"),
                    "training_seeds": canonical_json(
                        summary.get("training_seeds")
                    ),
                    "per_training_seed_values": canonical_json(
                        summary.get("per_training_seed_values")
                    ),
                    "descriptive_return_mean_across_training_seeds": summary.get(
                        "descriptive_return_mean_across_training_seeds"
                    ),
                    "descriptive_normalized_score_mean_across_training_seeds": summary.get(
                        "descriptive_normalized_score_mean_across_training_seeds"
                    ),
                    "training_seed_confidence_interval": summary.get(
                        "training_seed_confidence_interval"
                    ),
                }
            )
    for item in report.get(
        "primary_cross_controller_training_seed_evidence", []
    ):
        rows.append(
            {
                "row_type": "primary_comparison_training_seed_evidence",
                "evidence_label": item.get("evidence_label"),
                "comparison_group_id": item.get("comparison_group_id"),
                "statistical_unit": item.get("statistical_unit"),
                "n_training_seeds": item.get("n_training_seeds"),
                "training_seeds": canonical_json(item.get("training_seeds")),
                "per_training_seed_values": canonical_json(
                    item.get("per_training_seed_values")
                ),
                "descriptive_raw_mean_difference_across_training_seeds": item.get(
                    "descriptive_raw_mean_difference_across_training_seeds"
                ),
                "descriptive_normalized_mean_difference_across_training_seeds": item.get(
                    "descriptive_normalized_mean_difference_across_training_seeds"
                ),
                "training_seed_confidence_interval": item.get(
                    "training_seed_confidence_interval"
                ),
                "episode_returns_pooled_for_training_seed_inference": item.get(
                    "episode_returns_pooled_for_training_seed_inference"
                ),
                "checkpoint_seeds": canonical_json(item.get("checkpoint_seeds")),
                "n_checkpoint_pipelines": item.get("n_checkpoint_pipelines"),
                "per_checkpoint_mean_differences": canonical_json(
                    item.get("per_checkpoint_mean_differences")
                ),
                "mean_of_checkpoint_mean_differences": canonical_json(
                    item.get("mean_of_checkpoint_mean_differences")
                ),
                "positive_checkpoint_count": item.get(
                    "positive_checkpoint_count"
                ),
                "training_seed_ci": item.get("training_seed_ci"),
                "inference_note": item.get("inference_note"),
            }
        )
    for check in report.get("baseline_equivalence_checks", []):
        rows.append(
            {
                "row_type": "baseline_equivalence_check",
                "evidence_label": check.get("evidence_label"),
                "equivalence_check_id": check.get("check_id"),
                "equivalence_semantics": check.get("expected_baseline_semantics"),
                "equivalence_episode_count": check.get("episode_count"),
                "equivalence_status": check.get("status"),
                "equivalence_endpoints": ";".join(
                    _endpoint_label(endpoint) for endpoint in check.get("endpoints", [])
                ),
            }
        )
    return rows


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in tabular_rows(report):
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    return buffer.getvalue()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CA-OPEX frozen confirmation and adapter controls",
        "",
        f"Evidence scope: {_fmt(report.get('evidence_scope'))}",
        "",
        "> Episode confidence intervals are paired fixed-checkpoint rollout intervals. "
        "They are not training-seed confidence intervals.",
        "",
        "## Runs",
        "",
        "| Run | Method | Label | Train seed | Status | Updates | Wall s (cumulative) | Total Q1 forward rows | Q1 backward rows | Frozen Q scale |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in report.get("runs", []):
        cost = run.get("training_cost", {})
        lines.append(
            "| "
            + " | ".join(
                _fmt(item)
                for item in (
                    run.get("run_id"),
                    run.get("method_id"),
                    run.get("evidence_label"),
                    run.get("training_seed"),
                    run.get("observed_status"),
                    cost.get("updates_completed"),
                    cost.get("wall_time_seconds"),
                    cost.get("total_q1_forward_rows_lifetime"),
                    cost.get("optimizer_q1_backward_rows_lifetime"),
                    cost.get("frozen_q_scale"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Adapter costs distinguish base-actor preprocessing, frozen Q-scale "
            "calibration, optimizer Q1 forwards, and optimizer Q1 backward rows. "
            "Wall and optimizer time are cumulative across resume invocations. "
            "Preprocessing seconds cover only the latest invocation; preprocessing "
            "rows are multiplied by the recorded invocation count because every "
            "resume recomputes them. State-complete resume is implemented, but "
            "uninterrupted-versus-resumed bitwise equality is verified only on CPU, "
            "not CUDA.",
            "",
            "| Run | Invocations | Base actor rows/invocation | Base actor rows lifetime | Q-scale Q1 rows/invocation | Q-scale Q1 rows lifetime | Optimizer Q1 forward | Optimizer Q1 backward | Adapter MLP forward | Adapter MLP backward | Preprocess s (latest invocation) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report.get("runs", []):
        cost = run.get("training_cost", {})
        lines.append(
            "| "
            + " | ".join(
                _fmt(item)
                for item in (
                    run.get("run_id"),
                    cost.get("invocation_count"),
                    cost.get("base_actor_precompute_rows_per_invocation"),
                    cost.get("base_actor_precompute_rows_lifetime"),
                    cost.get("q_scale_q1_forward_rows_per_invocation"),
                    cost.get("q_scale_q1_forward_rows_lifetime"),
                    cost.get("optimizer_q1_forward_rows_lifetime"),
                    cost.get("optimizer_q1_backward_rows_lifetime"),
                    cost.get("optimizer_adapter_mlp_forward_rows_lifetime"),
                    cost.get("optimizer_adapter_mlp_backward_rows_lifetime"),
                    cost.get("preprocessing_seconds_this_invocation"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evaluation means",
            "",
            "| Run/evaluation | Label | Arm | Episodes | Raw mean | Normalized mean | Clip | Bound | Residual | Env/base-actor rows | Adapter MLP rows |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in report.get("runs", []):
        for evaluation in run.get("evaluations", []):
            for arm_name in sorted(evaluation.get("arms", {})):
                arm = evaluation["arms"][arm_name]
                lines.append(
                    "| "
                    + " | ".join(
                        _fmt(item)
                        for item in (
                            f"{run['run_id']}/{evaluation['evaluation_id']}",
                            evaluation.get("evidence_label"),
                            arm_name,
                            arm.get("episode_count"),
                            arm.get("return_mean"),
                            arm.get("normalized_score_mean"),
                            arm.get("actuator_clip_fraction"),
                            arm.get("command_at_bound_fraction"),
                            arm.get("applied_residual_abs_mean"),
                            arm.get("base_actor_forward_rows"),
                            arm.get("adapter_mlp_forward_rows"),
                        )
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "Adapter rollout base-actor and adapter-MLP rows are derived exactly "
            "from each arm's raw episode lengths. The evaluator records only one "
            "paired total wall time, so no per-arm wall time is imputed.",
        ]
    )
    lines.extend(
        [
            "",
            "## Cross-file baseline integrity",
            "",
            "| Check | Semantics | Episodes | Endpoints | Status |",
            "|---|---|---:|---|---|",
        ]
    )
    for check in report.get("baseline_equivalence_checks", []):
        lines.append(
            "| "
            + " | ".join(
                (
                    _fmt(check.get("check_id")),
                    _fmt(check.get("expected_baseline_semantics")),
                    _fmt(check.get("episode_count")),
                    _fmt(
                        "; ".join(
                            _endpoint_label(endpoint)
                            for endpoint in check.get("endpoints", [])
                        )
                    ),
                    _fmt(check.get("status")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Strict external controls",
            "",
            "| Run/evaluation | Method | Schema | Seed | Arm | Episodes | Raw mean | Normalized mean | Beta | Arm env steps | Deployment env steps | Paired eval env steps | Q1 forward rows | Q1 backward rows | Cost scope |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in report.get("external_evaluations", []):
        cost = run.get("cost", {})
        for evaluation in run.get("evaluations", []):
            for arm_name, arm in sorted(evaluation.get("arms", {}).items()):
                lines.append(
                    "| "
                    + " | ".join(
                        _fmt(item)
                        for item in (
                            f"{run['run_id']}/{evaluation['evaluation_id']}",
                            run.get("method_id"),
                            run.get("raw_schema"),
                            run.get("training_seed"),
                            arm_name,
                            arm.get("episode_count"),
                            arm.get("return_mean"),
                            arm.get("normalized_score_mean"),
                            arm.get("action_noise_beta"),
                            arm.get("total_environment_steps"),
                            cost.get("environment_steps"),
                            cost.get("paired_evaluation_environment_steps"),
                            cost.get("q1_forward_rows"),
                            cost.get("q1_backward_rows"),
                            cost.get("cost_scope"),
                        )
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "> The six predeclared primary comparisons are flagged below and "
            "summarized first at checkpoint-seed level in the next table. "
            "Other episode-level intervals are secondary descriptions; no "
            "multiple-comparison correction is applied.",
            "",
            "| Comparison | Primary? | Label | Pairs | Positive | Raw Δ | Raw t 95% | Raw bootstrap 95% | Norm Δ | Norm t 95% | Norm bootstrap 95% |",
            "|---|---|---|---:|---:|---:|---|---|---:|---|---|",
        ]
    )
    for item in report.get("paired_comparisons", []):
        raw = item["raw"]
        norm = item["normalized"]
        lines.append(
            "| "
            + " | ".join(
                (
                    _fmt(item.get("comparison_id")),
                    _fmt(item.get("is_primary_confirmation_comparison")),
                    _fmt(item.get("evidence_label")),
                    _fmt(item.get("episode_pairs")),
                    _fmt(item.get("positive_pairs")),
                    _fmt(raw.get("mean_difference")),
                    f"[{_fmt(raw.get('t_95ci_low'))}, {_fmt(raw.get('t_95ci_high'))}]",
                    f"[{_fmt(raw.get('paired_bootstrap_95ci_low'))}, {_fmt(raw.get('paired_bootstrap_95ci_high'))}]",
                    _fmt(norm.get("mean_difference")),
                    f"[{_fmt(norm.get('t_95ci_low'))}, {_fmt(norm.get('t_95ci_high'))}]",
                    f"[{_fmt(norm.get('paired_bootstrap_95ci_low'))}, {_fmt(norm.get('paired_bootstrap_95ci_high'))}]",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary cross-controller checkpoint summary",
            "",
            "> Each row is descriptive over exactly two base-policy checkpoint "
            "seeds (1 and 10). The same 50 rollout-case seed pairs are crossed "
            "with both checkpoints; episodes are not pooled into n=100 and no "
            "training-seed confidence interval or significance claim is made.",
            "",
            "| Comparison family | Checkpoint seeds | n checkpoints | Per-checkpoint raw Δ | Mean raw Δ | Mean normalized Δ | Positive checkpoints | Training-seed CI |",
            "|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in report.get(
        "primary_cross_controller_training_seed_evidence", []
    ):
        per_checkpoint = "; ".join(
            f"seed{value['training_seed']}={_fmt(value['raw_mean_difference'])}"
            for value in item.get("per_checkpoint_mean_differences", [])
        )
        checkpoint_mean = item.get("mean_of_checkpoint_mean_differences", {})
        lines.append(
            "| "
            + " | ".join(
                _fmt(value)
                for value in (
                    item.get("comparison_group_id"),
                    canonical_json(item.get("checkpoint_seeds")),
                    item.get("n_checkpoint_pipelines"),
                    per_checkpoint,
                    checkpoint_mean.get("raw"),
                    checkpoint_mean.get("normalized"),
                    item.get("positive_checkpoint_count"),
                    item.get("training_seed_ci"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## External-control checkpoint evidence",
            "",
            "| Method/evaluation/arm | Checkpoint seeds | n checkpoints | Per-checkpoint rollout means | Descriptive raw mean | Descriptive normalized mean | Training-seed CI |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for item in report.get("external_training_seed_evidence", []):
        for summary in item.get("evaluation_arm_descriptive_means", []):
            per_checkpoint = "; ".join(
                f"seed{value['training_seed']}={_fmt(value['return_mean'])}"
                for value in summary.get("per_training_seed_values", [])
            )
            lines.append(
                "| "
                + " | ".join(
                    _fmt(value)
                    for value in (
                        f"{item['method_id']}/{summary['evaluation_id']}/{summary['arm']}",
                        canonical_json(summary.get("training_seeds")),
                        summary.get("n_training_seeds"),
                        per_checkpoint,
                        summary.get(
                            "descriptive_return_mean_across_training_seeds"
                        ),
                        summary.get(
                            "descriptive_normalized_score_mean_across_training_seeds"
                        ),
                        summary.get("training_seed_confidence_interval"),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Frozen-Q audits",
            "",
            "| Run/audit | Label | States | K | Beta | Q1 gain mean/median | Q2 gain mean/median | Min-twin mean/median | Disagreement Δ mean/median | Physical rows | Critic rows |",
            "|---|---|---:|---:|---:|---|---|---|---|---:|---:|",
        ]
    )
    for run in report.get("runs", []):
        for audit in run.get("audits", []):
            lines.append(
                "| "
                + " | ".join(
                    _fmt(item)
                    for item in (
                        f"{run['run_id']}/{audit['audit_id']}",
                        audit.get("evidence_label"),
                        audit.get("selected_observations"),
                        audit.get("noise_samples"),
                        audit.get("audit_action_noise_beta"),
                        f"{_fmt(audit.get('q1_gain', {}).get('mean'))}/{_fmt(audit.get('q1_gain', {}).get('median'))}",
                        f"{_fmt(audit.get('q2_gain', {}).get('mean'))}/{_fmt(audit.get('q2_gain', {}).get('median'))}",
                        f"{_fmt(audit.get('min_twin_gain', {}).get('mean'))}/{_fmt(audit.get('min_twin_gain', {}).get('median'))}",
                        f"{_fmt(audit.get('twin_disagreement_change', {}).get('mean'))}/{_fmt(audit.get('twin_disagreement_change', {}).get('median'))}",
                        audit.get("physical_action_rows"),
                        audit.get("individual_critic_network_rows"),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Statistical-unit audit",
            "",
            "| Method | Label | Unit | Seeds | n | Multiple seeds (descriptive only) | Training-seed CI |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for item in report.get("training_seed_evidence", []):
        lines.append(
            "| "
            + " | ".join(
                _fmt(value)
                for value in (
                    item.get("method_id"),
                    item.get("evidence_label"),
                    item.get("statistical_unit"),
                    canonical_json(item.get("training_seeds")),
                    item.get("n_training_seeds"),
                    item.get("has_multiple_descriptive_checkpoint_seeds"),
                    item.get("training_seed_confidence_interval"),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_new(path: Path, content: str) -> None:
    """Atomically publish a new file without ever replacing an existing file."""

    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite frozen manifest: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_matching_recorded_path(
    recorded: Any, expected: Path, *, role: str
) -> None:
    if not isinstance(recorded, str) or not recorded:
        raise AggregationError(f"{role} path is missing")
    if Path(recorded).resolve() != expected.resolve():
        raise AggregationError(f"{role} path differs from the frozen manifest")


def _verified_source(path: Path, expected_sha256: str, *, role: str) -> Dict[str, Any]:
    if not path.is_file():
        raise AggregationError(f"{role} missing: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise AggregationError(f"{role} SHA256 mismatch")
    return {
        "role": role,
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _recompute_opex_candidate_metrics(
    raw: Mapping[str, Any],
    *,
    expected_protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    arms = raw.get("arms")
    normalization = raw.get("normalization")
    if not isinstance(arms, Mapping) or not isinstance(normalization, Mapping):
        raise AggregationError("OPEX development raw lacks arms or normalization")
    try:
        reference_min = float(normalization["reference_min_score"])
        reference_max = float(normalization["reference_max_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AggregationError("OPEX development normalization is invalid") from exc
    if (
        not math.isfinite(reference_min)
        or not math.isfinite(reference_max)
        or reference_max <= reference_min
    ):
        raise AggregationError("OPEX development normalization bounds are invalid")
    parsed: Dict[str, Dict[str, Any]] = {}
    for arm_name in ("baseline_only", "adapted"):
        arm = arms.get(arm_name)
        if not isinstance(arm, Mapping):
            raise AggregationError(f"OPEX development raw lacks {arm_name} arm")
        returns = arm.get("returns")
        env_seeds = arm.get("environment_seeds")
        noise_seeds = arm.get("action_noise_seeds")
        gradient_seeds = arm.get("gradient_noise_seeds")
        if (
            not isinstance(returns, list)
            or len(returns) != 10
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in returns
            )
        ):
            raise AggregationError(
                f"OPEX development {arm_name} must retain ten finite returns"
            )
        for field, seeds in (
            ("environment_seeds", env_seeds),
            ("action_noise_seeds", noise_seeds),
            ("gradient_noise_seeds", gradient_seeds),
        ):
            if (
                not isinstance(seeds, list)
                or len(seeds) != len(returns)
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            ):
                raise AggregationError(
                    f"OPEX development {arm_name} {field} is invalid"
                )
        return_mean = statistics.fmean(float(value) for value in returns)
        normalized_mean = (
            100.0 * (return_mean - reference_min) / (reference_max - reference_min)
        )
        if not _close(return_mean, float(arm.get("return_mean"))) or not _close(
            normalized_mean, float(arm.get("normalized_score_mean"))
        ):
            raise AggregationError(
                f"OPEX development {arm_name} reported mean is not raw-derived"
            )
        parsed[arm_name] = {
            "returns": [float(value) for value in returns],
            "environment_seeds": list(env_seeds),
            "action_noise_seeds": list(noise_seeds),
            "gradient_noise_seeds": list(gradient_seeds),
            "return_mean": return_mean,
            "normalized_score_mean": normalized_mean,
        }
    if (
        parsed["baseline_only"]["environment_seeds"]
        != parsed["adapted"]["environment_seeds"]
        or parsed["baseline_only"]["action_noise_seeds"]
        != parsed["adapted"]["action_noise_seeds"]
        or parsed["baseline_only"]["gradient_noise_seeds"]
        != parsed["adapted"]["gradient_noise_seeds"]
    ):
        raise AggregationError("OPEX development arms are not seed-paired")
    if expected_protocol is not None:
        expected_seed_lists = {
            "environment_seeds": list(
                range(
                    int(expected_protocol["environment_seed_start"]),
                    int(expected_protocol["environment_seed_start"]) + 10,
                )
            ),
            "action_noise_seeds": list(
                range(
                    int(expected_protocol["action_noise_seed_start"]),
                    int(expected_protocol["action_noise_seed_start"]) + 10,
                )
            ),
            "gradient_noise_seeds": list(
                range(
                    int(expected_protocol["gradient_noise_seed_start"]),
                    int(expected_protocol["gradient_noise_seed_start"]) + 10,
                )
            ),
        }
        for arm_name in ("baseline_only", "adapted"):
            for field, expected_seeds in expected_seed_lists.items():
                if parsed[arm_name][field] != expected_seeds:
                    raise AggregationError(
                        f"OPEX development {arm_name} {field} differs from protocol"
                    )
    differences = [
        adapted - baseline
        for adapted, baseline in zip(
            parsed["adapted"]["returns"], parsed["baseline_only"]["returns"]
        )
    ]
    paired = raw.get("paired")
    if not isinstance(paired, Mapping):
        raise AggregationError("OPEX development raw lacks paired record")
    reported_differences = paired.get("adapted_minus_baseline_returns")
    if (
        not isinstance(reported_differences, list)
        or len(reported_differences) != len(differences)
        or any(
            not _close(observed, expected)
            for observed, expected in zip(reported_differences, differences)
        )
    ):
        raise AggregationError("OPEX development paired differences are not raw-derived")
    paired_mean = statistics.fmean(differences)
    if not _close(paired_mean, float(paired.get("return_difference_mean"))):
        raise AggregationError("OPEX development paired mean is not raw-derived")
    return {
        "adapted_normalized_score_mean": parsed["adapted"][
            "normalized_score_mean"
        ],
        "baseline_normalized_score_mean": parsed["baseline_only"][
            "normalized_score_mean"
        ],
        "paired_raw_return_difference_mean": paired_mean,
    }


def verify_opex_development_selection(
    *,
    implementation_root: Path,
    results_root: Path,
    selection_spec: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed on the complete CA-OPEX development-selection chain.

    This is the canonical preflight used by the manifest freezer and is public
    so the confirmation runner can invoke exactly the same checks.  The v3
    branch inventories every candidate raw record, not only the selected two.
    """

    implementation_root = implementation_root.resolve()
    results_root = results_root.resolve()
    protocol_path = _resolve_results_path(
        implementation_root,
        selection_spec["protocol_path"],
        "OPEX development protocol",
    )
    selection_path = _resolve_results_path(
        results_root,
        selection_spec["selection_path"],
        "OPEX development selection",
    )
    selector_path = _resolve_results_path(
        implementation_root,
        selection_spec["selector_path"],
        "OPEX development selector",
    )
    evaluator_path = implementation_root / "evaluate_channel_opex.py"
    provenance_sources = [
        _verified_source(
            protocol_path,
            selection_spec["protocol_sha256"],
            role="opex_development_protocol",
        ),
        _verified_source(
            selection_path,
            selection_spec["selection_sha256"],
            role="opex_development_selection",
        ),
        _verified_source(
            selector_path,
            selection_spec["selector_sha256"],
            role="opex_development_selector",
        ),
        _verified_source(
            evaluator_path,
            selection_spec["evaluator_sha256"],
            role="opex_development_evaluator",
        ),
    ]
    selection_payload, _ = _read_json(selection_path)
    schema = selection_payload.get("schema_version")
    if schema != selection_spec["selection_schema"]:
        raise AggregationError("freeze OPEX development selection schema mismatch")

    if schema == "channel-opex-development-selection-v1":
        if (
            selection_payload.get("status") != "complete"
            or selection_payload.get("evidence_label") != "development"
            or selection_payload.get("confirmation_data_read") is not False
            or selection_payload.get("protocol_sha256")
            != selection_spec["protocol_sha256"]
            or selection_payload.get("selection_script_sha256")
            != selection_spec["selector_sha256"]
            or selection_payload.get("evaluator_sha256")
            != selection_spec["evaluator_sha256"]
        ):
            raise AggregationError(
                "freeze OPEX legacy development selection metadata mismatch"
            )
        selected = selection_payload.get("selected", {}).get(
            selection_spec["anchor"]
        )
        if (
            not isinstance(selected, Mapping)
            or not _close(
                float(selected.get("step_size")),
                float(selection_spec["selected_step_size"]),
            )
            or selected.get("raw_sha256")
            != selection_spec["selected_candidate_raw_sha256"]
        ):
            raise AggregationError(
                "freeze OPEX selected candidate differs from manifest"
            )
        selected_path = Path(str(selected.get("path"))).resolve()
        try:
            selected_path.relative_to(results_root)
        except ValueError as exc:
            raise AggregationError(
                "freeze OPEX selected raw path escapes results_root"
            ) from exc
        selected_source = _verified_source(
            selected_path,
            selection_spec["selected_candidate_raw_sha256"],
            role=(
                f"opex_development_candidate:{selection_spec['anchor']}:"
                f"{float(selection_spec['selected_step_size']):g}"
            ),
        )
        return {
            "schema_version": schema,
            "candidate_count": 1,
            "provenance_sources": provenance_sources,
            "candidate_sources": [selected_source],
        }

    if schema != "channel-opex-boundary-extension-selection-revalidated-v3":
        raise AggregationError("unsupported OPEX development selection schema")
    expected_chronology = {
        "selected_ca_opex_confirmation_outputs_read_before_selection": False,
        "selected_adapter_confirmation_outputs_read_before_selection": False,
        "external_baseline_arms_on_confirmation_seeds_preexisting_and_observed": True,
        "global_confirmation_blindness_claim": False,
    }
    if (
        selection_payload.get("status") != "complete"
        or selection_payload.get("evidence_label") != "development"
        or selection_payload.get("provenance_revalidation_after_evaluations")
        is not True
        or selection_payload.get("protocol_sha256")
        != selection_spec["protocol_sha256"]
        or selection_payload.get("selector_sha256")
        != selection_spec["selector_sha256"]
        or selection_payload.get("evaluator_sha256")
        != selection_spec["evaluator_sha256"]
    ):
        raise AggregationError(
            "freeze OPEX revalidated development selection metadata mismatch"
        )
    for field, expected in expected_chronology.items():
        if selection_spec[field] is not expected or selection_payload.get(field) is not expected:
            raise AggregationError(f"freeze OPEX chronology mismatch for {field}")
    _require_matching_recorded_path(
        selection_payload.get("protocol_path"),
        protocol_path,
        role="OPEX amended protocol",
    )

    pre_protocol_path = _resolve_results_path(
        implementation_root,
        "channel_opex_boundary_extension_protocol_pre_evaluation.json",
        "OPEX pre-evaluation protocol",
    )
    pre_selection_path = _resolve_results_path(
        results_root,
        "channel_opex_dev_extension/selection.json",
        "OPEX pre-evaluation selection",
    )
    pre_selector_path = _resolve_results_path(
        implementation_root,
        "select_channel_opex_boundary_extension_pre_evaluation.py",
        "OPEX pre-evaluation selector",
    )
    prior_protocol_path = _resolve_results_path(
        implementation_root,
        selection_spec["prior_protocol_path"],
        "OPEX original-grid protocol",
    )
    prior_selection_path = _resolve_results_path(
        results_root,
        selection_spec["prior_selection_path"],
        "OPEX original-grid selection",
    )
    prior_selector_path = _resolve_results_path(
        implementation_root,
        selection_spec["prior_selector_path"],
        "OPEX original-grid selector",
    )
    chain = (
        (
            pre_protocol_path,
            selection_spec["pre_evaluation_protocol_sha256"],
            "opex_pre_evaluation_protocol",
        ),
        (
            pre_selection_path,
            selection_spec["pre_evaluation_selection_sha256"],
            "opex_pre_evaluation_selection",
        ),
        (
            pre_selector_path,
            selection_spec["pre_evaluation_selector_sha256"],
            "opex_pre_evaluation_selector",
        ),
        (
            prior_protocol_path,
            selection_spec["prior_protocol_sha256"],
            "opex_original_grid_protocol",
        ),
        (
            prior_selection_path,
            selection_spec["prior_selection_sha256"],
            "opex_original_grid_selection",
        ),
        (
            prior_selector_path,
            selection_spec["prior_selector_dependency_sha256"],
            "opex_original_grid_selector",
        ),
    )
    provenance_sources.extend(
        _verified_source(path, digest, role=role)
        for path, digest, role in chain
    )
    for field, path in (
        ("pre_evaluation_protocol_path", pre_protocol_path),
        ("pre_evaluation_selection_path", pre_selection_path),
        ("pre_evaluation_selector_path", pre_selector_path),
        ("prior_selection_path", prior_selection_path),
    ):
        _require_matching_recorded_path(
            selection_payload.get(field), path, role=f"OPEX {field}"
        )
    for field in (
        "pre_evaluation_protocol_sha256",
        "pre_evaluation_selection_sha256",
        "pre_evaluation_selector_sha256",
        "prior_selection_sha256",
        "prior_selector_dependency_sha256",
    ):
        if selection_payload.get(field) != selection_spec[field]:
            raise AggregationError(f"freeze OPEX {field} mismatch")

    protocol_payload, _ = _read_json(protocol_path)
    if (
        protocol_payload.get("schema_version")
        != "channel-opex-boundary-extension-provenance-amendment-v3"
        or protocol_payload.get("evidence_label") != "development"
        or protocol_payload.get("protocol_created_after_extension_evaluations")
        is not True
        or protocol_payload.get("provenance_revalidation_after_evaluations")
        is not True
    ):
        raise AggregationError("freeze OPEX amended protocol metadata mismatch")
    protocol_chronology = protocol_payload.get("confirmation_chronology")
    if not isinstance(protocol_chronology, Mapping):
        raise AggregationError("freeze OPEX amended protocol lacks chronology")
    for field, expected in expected_chronology.items():
        if protocol_chronology.get(field) is not expected:
            raise AggregationError(f"freeze OPEX protocol chronology mismatch for {field}")
    pre_record = protocol_payload.get("pre_evaluation_record")
    if not isinstance(pre_record, Mapping):
        raise AggregationError("freeze OPEX protocol lacks pre-evaluation record")
    expected_pre = {
        "protocol_sha256": selection_spec["pre_evaluation_protocol_sha256"],
        "selector_sha256": selection_spec["pre_evaluation_selector_sha256"],
        "selection_sha256": selection_spec["pre_evaluation_selection_sha256"],
    }
    for field, expected in expected_pre.items():
        if pre_record.get(field) != expected:
            raise AggregationError(f"freeze OPEX protocol pre-record {field} mismatch")
    for field, path in (
        ("protocol_path", pre_protocol_path),
        ("selector_path", pre_selector_path),
        ("selection_path", pre_selection_path),
    ):
        recorded_path = Path(str(pre_record.get(field)))
        if not recorded_path.is_absolute():
            recorded_path = protocol_path.parent / recorded_path
        if recorded_path.resolve() != path:
            raise AggregationError(f"freeze OPEX protocol pre-record {field} mismatch")
    protocol_implementation = protocol_payload.get("implementation")
    if (
        not isinstance(protocol_implementation, Mapping)
        or protocol_implementation.get("evaluate_channel_opex_sha256")
        != selection_spec["evaluator_sha256"]
        or protocol_implementation.get("select_boundary_extension_sha256")
        != selection_spec["selector_sha256"]
        or protocol_implementation.get("select_channel_opex_dev_dependency_sha256")
        != selection_spec["prior_selector_dependency_sha256"]
    ):
        raise AggregationError("freeze OPEX amended protocol implementation mismatch")
    prior_grid = protocol_payload.get("prior_grid")
    if not isinstance(prior_grid, Mapping):
        raise AggregationError("freeze OPEX protocol lacks original-grid record")
    for field, expected in (
        ("protocol_sha256", selection_spec["prior_protocol_sha256"]),
        ("selection_sha256", selection_spec["prior_selection_sha256"]),
        (
            "selection_script_sha256",
            selection_spec["prior_selector_dependency_sha256"],
        ),
    ):
        if prior_grid.get(field) != expected:
            raise AggregationError(f"freeze OPEX original-grid {field} mismatch")
    for field, path in (
        ("protocol_path", prior_protocol_path),
        ("selection_path", prior_selection_path),
    ):
        recorded_path = Path(str(prior_grid.get(field)))
        if not recorded_path.is_absolute():
            recorded_path = protocol_path.parent / recorded_path
        if recorded_path.resolve() != path:
            raise AggregationError(f"freeze OPEX original-grid {field} mismatch")

    pre_selection, _ = _read_json(pre_selection_path)
    if (
        pre_selection.get("schema_version")
        != "channel-opex-boundary-extension-selection-v1"
        or pre_selection.get("status") != "complete"
        or pre_selection.get("confirmation_data_read") is not False
        or pre_selection.get("protocol_sha256")
        != selection_spec["pre_evaluation_protocol_sha256"]
        or pre_selection.get("selector_sha256")
        != selection_spec["pre_evaluation_selector_sha256"]
        or pre_selection.get("prior_selection_sha256")
        != selection_spec["prior_selection_sha256"]
    ):
        raise AggregationError("freeze OPEX pre-evaluation selection chain mismatch")
    prior_selection, _ = _read_json(prior_selection_path)
    if (
        prior_selection.get("schema_version")
        != "channel-opex-development-selection-v1"
        or prior_selection.get("status") != "complete"
        or prior_selection.get("confirmation_data_read") is not False
        or prior_selection.get("protocol_sha256")
        != selection_spec["prior_protocol_sha256"]
        or prior_selection.get("selection_script_sha256")
        != selection_spec["prior_selector_dependency_sha256"]
        or prior_selection.get("evaluator_sha256")
        != selection_spec["evaluator_sha256"]
    ):
        raise AggregationError("freeze OPEX original-grid selection chain mismatch")

    method_contract = protocol_payload.get("method")
    base_contract = protocol_payload.get("base_checkpoint")
    calibration_contract = protocol_payload.get("channel_calibration")
    rollout_contract = protocol_payload.get("rollout_protocol")
    if not all(
        isinstance(item, Mapping)
        for item in (
            method_contract,
            base_contract,
            calibration_contract,
            rollout_contract,
        )
    ):
        raise AggregationError("freeze OPEX amended protocol lacks method artifacts")
    expected_delta = method_contract.get("delta_max_linf_by_anchor")
    if not isinstance(expected_delta, Mapping):
        raise AggregationError("freeze OPEX protocol lacks per-anchor trust regions")
    protocol_model_beta = calibration_contract.get(
        "estimated_beta", calibration_contract.get("beta")
    )
    if not isinstance(protocol_model_beta, (int, float)):
        raise AggregationError("freeze OPEX protocol lacks calibrated beta")

    candidates = selection_payload.get("candidates")
    if not isinstance(candidates, Mapping) or set(candidates) != {"inverse", "identity"}:
        raise AggregationError("freeze OPEX selection must contain both candidate anchors")
    expected_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    candidate_sources: List[Dict[str, Any]] = []
    candidate_keys: set[Tuple[str, str]] = set()
    selected_map = selection_payload.get("selected")
    if not isinstance(selected_map, Mapping) or set(selected_map) != {
        "inverse",
        "identity",
    }:
        raise AggregationError("freeze OPEX selection must select both anchors")
    for anchor in ("inverse", "identity"):
        records = candidates.get(anchor)
        if not isinstance(records, list) or len(records) != len(expected_grid):
            raise AggregationError(f"freeze OPEX {anchor} candidate inventory incomplete")
        observed_grid = sorted(float(record.get("step_size")) for record in records)
        if any(
            not _close(observed, expected)
            for observed, expected in zip(observed_grid, expected_grid)
        ):
            raise AggregationError(f"freeze OPEX {anchor} candidate grid mismatch")
        for record in records:
            if not isinstance(record, Mapping) or not _is_sha256(record.get("raw_sha256")):
                raise AggregationError(f"freeze OPEX {anchor} candidate record invalid")
            raw_path = Path(str(record.get("path"))).resolve()
            try:
                raw_path.relative_to(results_root)
            except ValueError as exc:
                raise AggregationError(
                    f"freeze OPEX {anchor} candidate escapes results_root"
                ) from exc
            source = _verified_source(
                raw_path,
                record["raw_sha256"],
                role=f"opex_development_candidate:{anchor}:{float(record['step_size']):g}",
            )
            key = (str(raw_path), record["raw_sha256"])
            if key in candidate_keys:
                raise AggregationError("freeze OPEX candidate path/hash is duplicated")
            candidate_keys.add(key)
            raw, _ = _read_json(raw_path)
            controller = raw.get("controller")
            if (
                raw.get("raw_schema") != "channel_opex_v1"
                or raw.get("status") != "complete"
                or raw.get("implementation") != expected_implementation
                or not isinstance(controller, Mapping)
                or controller.get("baseline_transform") != anchor
                or not _close(
                    float(controller.get("step_size")), float(record["step_size"])
                )
            ):
                raise AggregationError(
                    f"freeze OPEX {anchor} candidate raw/config/implementation mismatch"
                )
            raw_base = raw.get("base_checkpoint")
            raw_calibration = raw.get("calibration")
            raw_channel = raw.get("channel")
            raw_protocol = raw.get("evaluation_protocol")
            raw_normalization = raw.get("normalization")
            if not all(
                isinstance(item, Mapping)
                for item in (
                    raw_base,
                    raw_calibration,
                    raw_channel,
                    raw_protocol,
                    raw_normalization,
                )
            ):
                raise AggregationError(
                    f"freeze OPEX {anchor} candidate lacks protocol metadata"
                )
            exact_controller = {
                "gradient_steps": method_contract["gradient_steps"],
                "gradient_steps_per_action": method_contract["gradient_steps"],
                "K": method_contract["gradient_noise_samples"],
                "execution_noise_samples": method_contract[
                    "gradient_noise_samples"
                ],
                "q_reducer": method_contract["q_reducer"],
                "delta_max": expected_delta[anchor],
                "model_beta": protocol_model_beta,
                "calibration_mode": "censored_uniform_plus_clip_pair_calibration",
                "critic": "frozen_q1",
                "gradient_objective": (
                    "mean_k_q1_of_clipped_command_plus_fresh_antithetic_"
                    "uniform_noise_per_gradient_step"
                ),
                "actor_parameter_updates": 0,
                "critic_parameter_updates": 0,
            }
            for field, expected in exact_controller.items():
                observed = controller.get(field)
                numeric = isinstance(expected, (int, float)) and not isinstance(
                    expected, bool
                )
                matched = (
                    not isinstance(observed, bool)
                    and isinstance(observed, (int, float))
                    and _close(float(observed), float(expected))
                    if numeric
                    else observed == expected
                )
                if not matched:
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate controller {field} mismatch"
                    )
            if (
                base_contract.get("training_seed") != 0
                or
                raw_base.get("sha256") != base_contract.get("sha256")
                or raw_calibration.get("calibration_sha256")
                != calibration_contract.get("sha256")
                or not _close(
                    float(raw_calibration.get("beta")),
                    float(protocol_model_beta),
                )
                or not _close(
                    float(raw_channel.get("model_or_calibration_beta")),
                    float(protocol_model_beta),
                )
                or not _close(
                    float(raw_channel.get("environment_rollout_beta")),
                    float(rollout_contract.get("environment_rollout_beta")),
                )
            ):
                raise AggregationError(
                    f"freeze OPEX {anchor} candidate base/calibration/channel mismatch"
                )
            protocol_checks = {
                "environment": rollout_contract["environment"],
                "episode_count_per_arm": rollout_contract[
                    "episode_count_per_arm"
                ],
                "environment_seed_start": rollout_contract[
                    "environment_seed_start"
                ],
                "action_noise_seed_start": rollout_contract[
                    "action_noise_seed_start"
                ],
                "gradient_noise_seed_start": rollout_contract[
                    "gradient_noise_seed_start"
                ],
                "paired_environment_and_action_noise_seeds": True,
                "gradient_noise_sampling_frequency": "per_gradient_step",
                "gradient_noise_antithetic": True,
            }
            for field, expected in protocol_checks.items():
                if raw_protocol.get(field) != expected:
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate rollout {field} mismatch"
                    )
            if raw.get("environment") != rollout_contract["environment"]:
                raise AggregationError(
                    f"freeze OPEX {anchor} candidate environment mismatch"
                )
            for field, protocol_field in (
                ("reference_min_score", "reference_min_score"),
                ("reference_max_score", "reference_max_score"),
            ):
                if not _close(
                    float(raw_normalization.get(field)),
                    float(rollout_contract.get(protocol_field)),
                ):
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate normalization mismatch"
                    )
            for arm_name in ("baseline_only", "adapted"):
                if not _close(
                    float(raw.get("arms", {}).get(arm_name, {}).get("action_noise_beta")),
                    float(rollout_contract["environment_rollout_beta"]),
                ):
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate arm rollout beta mismatch"
                    )
            recomputed = _recompute_opex_candidate_metrics(
                raw, expected_protocol=rollout_contract
            )
            for field, expected in recomputed.items():
                try:
                    observed = float(record.get(field))
                except (TypeError, ValueError) as exc:
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate lacks {field}"
                    ) from exc
                if not _close(observed, expected):
                    raise AggregationError(
                        f"freeze OPEX {anchor} candidate {field} is not raw-derived"
                    )
            source["verified_implementation"] = dict(raw["implementation"])
            source["anchor"] = anchor
            source["step_size"] = float(record["step_size"])
            source["episode_count_per_arm"] = 10
            candidate_sources.append(source)
        chosen = max(
            records,
            key=lambda item: (
                float(item["adapted_normalized_score_mean"]),
                -float(item["step_size"]),
            ),
        )
        selected = selected_map[anchor]
        if not isinstance(selected, Mapping) or dict(selected) != dict(chosen):
            raise AggregationError(f"freeze OPEX selected {anchor} is not rule-optimal")
    if len(candidate_sources) != int(selection_spec["candidate_count"]):
        raise AggregationError("freeze OPEX candidate_count differs from inventory")
    selected = selected_map[selection_spec["anchor"]]
    if (
        not _close(
            float(selected["step_size"]),
            float(selection_spec["selected_step_size"]),
        )
        or selected["raw_sha256"]
        != selection_spec["selected_candidate_raw_sha256"]
    ):
        raise AggregationError("freeze OPEX selected candidate differs from manifest")
    return {
        "schema_version": schema,
        "candidate_count": len(candidate_sources),
        "provenance_sources": provenance_sources,
        "candidate_sources": candidate_sources,
        "chronology": expected_chronology,
    }


def _index_vector_sha256(indices: Any) -> str:
    import numpy as np

    values = np.asarray(indices, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _recompute_observation_split_fingerprint(
    dataset_path: Path,
    *,
    max_observations: Optional[int],
    audit_fraction: float,
    split_seed: int,
) -> Dict[str, Any]:
    """Recompute the trainer's deterministic split without trusting a summary."""

    import h5py
    import numpy as np

    with h5py.File(dataset_path, "r") as handle:
        if "observations" in handle:
            shape = handle["observations"].shape
            if len(shape) != 2 or int(shape[0]) < 2:
                raise AggregationError("freeze dataset observations have invalid shape")
            total = int(shape[0])
            stop = total if max_observations is None else min(total, max_observations)
            terminal_key = next(
                (key for key in ("terminals", "terminations") if key in handle),
                None,
            )
            timeout_key = next(
                (key for key in ("timeouts", "truncations") if key in handle),
                None,
            )
            episode_units = None
            if terminal_key is not None or timeout_key is not None:
                for key in (terminal_key, timeout_key):
                    if key is not None and int(handle[key].shape[0]) != total:
                        raise AggregationError(
                            "freeze dataset boundary array is not observation-aligned"
                        )
                terminals = (
                    np.asarray(handle[terminal_key][:stop], dtype=np.bool_).reshape(-1)
                    if terminal_key is not None
                    else np.zeros(stop, dtype=np.bool_)
                )
                timeouts = (
                    np.asarray(handle[timeout_key][:stop], dtype=np.bool_).reshape(-1)
                    if timeout_key is not None
                    else np.zeros(stop, dtype=np.bool_)
                )
                stops = (np.flatnonzero(terminals | timeouts) + 1).astype(np.int64)
                if stops.size == 0 or int(stops[-1]) != stop:
                    stops = np.concatenate((stops, np.asarray([stop], dtype=np.int64)))
                lengths = np.diff(
                    np.concatenate((np.asarray([0], dtype=np.int64), stops))
                )
                episode_units = np.repeat(
                    np.arange(stops.size, dtype=np.int64), lengths
                )
            count = stop
        else:
            keys = sorted(
                (key for key in handle if key.startswith("episode_")),
                key=lambda key: int(key.rsplit("_", 1)[1]),
            )
            lengths = []
            remaining = max_observations
            for key in keys:
                if "observations" not in handle[key]:
                    raise AggregationError(f"freeze dataset {key} lacks observations")
                take = int(handle[key]["observations"].shape[0])
                if remaining is not None:
                    take = min(take, remaining)
                    remaining -= take
                if take:
                    lengths.append(take)
                if remaining == 0:
                    break
            if not lengths:
                raise AggregationError("freeze dataset contains no observations")
            count = int(sum(lengths))
            episode_units = np.repeat(
                np.arange(len(lengths), dtype=np.int64),
                np.asarray(lengths, dtype=np.int64),
            )
    if count < 2:
        raise AggregationError("freeze dataset is too small for a split")
    target_audit_count = max(
        1, min(count - 1, int(np.ceil(float(audit_fraction) * count)))
    )
    rng = np.random.default_rng(int(split_seed))
    unique_units = (
        np.unique(episode_units)
        if episode_units is not None
        else np.asarray([], dtype=np.int64)
    )
    if episode_units is not None and unique_units.size >= 2:
        unit_values, unit_counts = np.unique(episode_units, return_counts=True)
        count_by_unit = {
            int(unit): int(unit_count)
            for unit, unit_count in zip(unit_values, unit_counts)
        }
        audit_units = []
        selected_count = 0
        for unit in rng.permutation(unique_units)[:-1]:
            audit_units.append(int(unit))
            selected_count += count_by_unit[int(unit)]
            if selected_count >= target_audit_count:
                break
        audit_mask = np.isin(
            episode_units, np.asarray(audit_units, dtype=np.int64)
        )
        audit_unit_array = np.sort(np.asarray(audit_units, dtype=np.int64))
        split_method = "whole_episode_permutation_to_state_count"
    else:
        audit_positions = rng.permutation(count)[:target_audit_count]
        audit_mask = np.zeros(count, dtype=np.bool_)
        audit_mask[audit_positions] = True
        audit_unit_array = np.asarray([], dtype=np.int64)
        split_method = "fixed_index_permutation_no_episode_boundaries"
    audit_indices = np.flatnonzero(audit_mask).astype(np.int64)
    train_indices = np.flatnonzero(~audit_mask).astype(np.int64)
    return {
        "considered_observation_count": count,
        "train_observation_count": int(train_indices.size),
        "audit_observation_count": int(audit_indices.size),
        "requested_audit_fraction": float(audit_fraction),
        "realized_audit_fraction": float(audit_indices.size / count),
        "split_method": split_method,
        "audit_episode_unit_count": int(audit_unit_array.size),
        "train_indices_sha256": _index_vector_sha256(train_indices),
        "audit_indices_sha256": _index_vector_sha256(audit_indices),
        "audit_episode_units_sha256": _index_vector_sha256(audit_unit_array),
    }


def freeze_strict_manifest_template(
    template_path: Path,
    output_path: Path,
    *,
    implementation_root: Path,
    artifact_paths: Sequence[Path],
    results_root_override: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve a strict template only before runs exist and verify all artifacts.

    Artifact paths are matched by content hash against every non-null dataset,
    base-checkpoint, and calibration SHA predeclared in the template. This keeps
    file roles in the immutable template while avoiding hand-entered freeze data.
    """

    template_path = template_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_path}")
    template, template_source = _read_json(template_path)
    _validate_manifest_has_no_results(template)
    validate_strict_manifest_definition(template, allow_placeholders=True)
    unresolved = _placeholder_locations(template)
    required_root_placeholder_locations = {
        f"manifest.frozen_implementation.{field}"
        for field in FROZEN_IMPLEMENTATION_FIELDS
    }
    if not required_root_placeholder_locations.issubset(set(unresolved)):
        raise AggregationError(
            "freeze template lacks one or more required adapter implementation placeholders"
        )
    unexpected_placeholder_locations = [
        location
        for location in unresolved
        if location not in required_root_placeholder_locations
        and ".expected_implementation." not in location
    ]
    if unexpected_placeholder_locations:
        raise AggregationError(
            "freeze template has placeholders outside declared implementation fields: "
            + ", ".join(unexpected_placeholder_locations[:3])
        )

    root_spec = template.get("results_root")
    if results_root_override is None:
        if not isinstance(root_spec, str) or not root_spec:
            raise AggregationError("freeze template requires results_root")
        results_root = (template_path.parent / root_spec).resolve()
    else:
        results_root = results_root_override.resolve()
    existing_run_dirs = []
    for spec in template["runs"]:
        run_dir = _resolve_results_path(
            results_root, spec["training_dir"], "training_dir"
        )
        if run_dir.exists():
            existing_run_dirs.append(str(run_dir))
    if existing_run_dirs:
        raise AggregationError(
            "refusing to freeze after confirmation output directories exist: "
            + ", ".join(existing_run_dirs[:3])
        )

    implementation_root = implementation_root.resolve()
    implementation_hashes: Dict[str, str] = {}
    implementation_sources: Dict[str, Dict[str, Any]] = {}
    for field, filename in IMPLEMENTATION_SOURCE_FILES.items():
        path = implementation_root / filename
        if not path.is_file():
            raise AggregationError(f"freeze implementation file missing: {path}")
        digest = sha256_file(path)
        implementation_hashes[field] = digest
        implementation_sources[field] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    placeholder_replacements: Dict[str, str] = {}
    resolved_implementation_sources_by_filename: Dict[str, Dict[str, Any]] = {}
    for placeholder, filename in FREEZE_PLACEHOLDER_SOURCE_FILES.items():
        path = implementation_root / filename
        if not path.is_file():
            raise AggregationError(f"freeze implementation file missing: {path}")
        digest = sha256_file(path)
        placeholder_replacements[placeholder] = digest
        resolved_implementation_sources_by_filename[filename] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }

    expected_artifact_hashes = {
        value
        for spec in template["runs"]
        for field, value in spec["expected_artifacts"].items()
        if field.endswith("_sha256") and value is not None
    }
    supplied_artifacts: List[Dict[str, Any]] = []
    supplied_hashes = set()
    for raw_path in artifact_paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise AggregationError(f"freeze artifact missing: {path}")
        digest = sha256_file(path)
        supplied_hashes.add(digest)
        supplied_artifacts.append(
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        )
    missing = sorted(expected_artifact_hashes - supplied_hashes)
    unexpected = sorted(supplied_hashes - expected_artifact_hashes)
    if missing or unexpected:
        raise AggregationError(
            "freeze artifact SHA set differs from template; "
            f"missing={missing}, unexpected={unexpected}"
        )

    frozen = _replace_freeze_placeholders(
        json.loads(canonical_json(template)), placeholder_replacements
    )
    frozen["frozen_implementation"] = implementation_hashes
    frozen["freeze_provenance"] = {
        "template": template_source,
        "results_root_verified_absent": str(results_root),
        "implementation_sources": implementation_sources,
        "resolved_implementation_sources_by_filename": (
            resolved_implementation_sources_by_filename
        ),
        "artifact_sources": supplied_artifacts,
        "selection_rule": (
            "frozen before any predeclared training_dir existed; all artifact "
            "hashes matched the predeclared set"
        ),
    }
    validate_strict_manifest_definition(frozen, allow_placeholders=False)
    artifact_path_by_sha = {
        source["sha256"]: Path(source["path"]) for source in supplied_artifacts
    }
    verified_split_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for run_spec in frozen["runs"]:
        expected_config = run_spec["expected_config"]
        dataset_sha = run_spec["expected_artifacts"]["dataset_sha256"]
        dataset_path = artifact_path_by_sha[dataset_sha]
        split_key = (
            dataset_sha,
            expected_config["max_observations"],
            float(expected_config["audit_fraction"]),
            int(expected_config["split_seed"]),
        )
        split = verified_split_by_key.get(split_key)
        if split is None:
            split = _recompute_observation_split_fingerprint(
                dataset_path,
                max_observations=expected_config["max_observations"],
                audit_fraction=float(expected_config["audit_fraction"]),
                split_seed=int(expected_config["split_seed"]),
            )
            split = {
                **split,
                "dataset_path": str(dataset_path),
                "dataset_sha256": dataset_sha,
                "max_observations": expected_config["max_observations"],
                "audit_fraction": float(expected_config["audit_fraction"]),
                "split_seed": int(expected_config["split_seed"]),
            }
            verified_split_by_key[split_key] = split
        for audit_spec in run_spec.get("audits", []):
            expected_protocol = audit_spec["expected_protocol"]
            for field in (
                "train_indices_sha256",
                "audit_indices_sha256",
                "audit_episode_units_sha256",
            ):
                if expected_protocol[field] != split[field]:
                    raise AggregationError(
                        f"freeze run {run_spec['run_id']} {field} differs from "
                        "the recomputed dataset split"
                    )
            if int(expected_protocol["selected_observations"]) != int(
                split["audit_observation_count"]
            ):
                raise AggregationError(
                    f"freeze run {run_spec['run_id']} held-out observation count mismatch"
                )
    frozen["freeze_provenance"]["verified_observation_splits"] = sorted(
        verified_split_by_key.values(),
        key=lambda item: (
            item["dataset_sha256"],
            item["split_seed"],
            item["audit_fraction"],
        ),
    )
    verified_external_sources = []
    verified_candidate_sources: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for external_spec in frozen.get("external_evaluations", []):
        if external_spec["raw_schema"] == "evaluate_td3bc_v1":
            _, _, external_sources = _parse_external_evaluation(
                results_root, external_spec, frozen
            )
            verified_external_sources.extend(external_sources)
        else:
            selection_spec = external_spec.get("development_selection")
            if selection_spec is not None:
                verification = verify_opex_development_selection(
                    implementation_root=implementation_root,
                    results_root=results_root,
                    selection_spec=selection_spec,
                    expected_implementation=external_spec[
                        "expected_implementation"
                    ],
                )
                for source in verification["provenance_sources"]:
                    tagged = dict(source)
                    tagged["selection_consumer_run_id"] = external_spec["run_id"]
                    verified_external_sources.append(tagged)
                for source in verification["candidate_sources"]:
                    key = (source["path"], source["sha256"])
                    prior = verified_candidate_sources.setdefault(key, dict(source))
                    if prior != source:
                        raise AggregationError(
                            "inconsistent duplicate OPEX development candidate source"
                        )
            future_path = _resolve_results_path(
                results_root,
                external_spec["path"],
                "future OPEX evaluation path",
            )
            if future_path.exists():
                raise AggregationError(
                    "refusing to freeze after a predeclared OPEX raw path exists: "
                    f"{future_path}"
                )
            verified_external_sources.append(
                {
                    "role": f"future_external_evaluation:{external_spec['run_id']}",
                    "path": str(future_path),
                    "status": "verified_absent_before_confirmation",
                }
            )
    frozen["freeze_provenance"]["verified_external_sources"] = (
        verified_external_sources
    )
    frozen["freeze_provenance"]["verified_opex_development_candidates"] = sorted(
        verified_candidate_sources.values(),
        key=lambda item: (item["role"], item["path"]),
    )
    # Re-run strict recursive validation after all derived provenance has been
    # attached. This catches placeholder tokens in both mapping keys and values
    # before an immutable manifest can be published.
    validate_strict_manifest_definition(frozen, allow_placeholders=False)
    atomic_write_new(
        output_path,
        json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    return {
        "status": "frozen",
        "manifest": str(output_path),
        "manifest_sha256": sha256_file(output_path),
        "runs": len(frozen["runs"]),
        "implementation": implementation_hashes,
        "verified_artifact_hashes": sorted(supplied_hashes),
    }


def write_report(report: Mapping[str, Any], output_dir: Path, stem: str) -> Dict[str, str]:
    output_dir = output_dir.resolve()
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    atomic_write(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    atomic_write(csv_path, render_csv(report))
    atomic_write(markdown_path, render_markdown(report))
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--freeze-template", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem", default="inverse_residual_results")
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument(
        "--implementation-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--freeze-artifact",
        type=Path,
        action="append",
        default=[],
        help=(
            "Dataset, base checkpoint, or calibration file to verify by SHA; "
            "repeat until the template's exact artifact-hash set is covered."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.freeze_template is not None:
        if args.frozen_manifest is None:
            parser.error("--freeze-template requires --frozen-manifest")
        result = freeze_strict_manifest_template(
            args.freeze_template,
            args.frozen_manifest,
            implementation_root=args.implementation_root,
            artifact_paths=args.freeze_artifact,
            results_root_override=args.results_root,
        )
        print(canonical_json(result))
        return 0
    if args.output_dir is None:
        parser.error("--manifest aggregation requires --output-dir")
    report = build_report(
        args.manifest,
        results_root_override=args.results_root,
        bootstrap_seed_override=args.bootstrap_seed,
        bootstrap_samples_override=args.bootstrap_samples,
    )
    paths = write_report(report, args.output_dir, args.stem)
    print(
        canonical_json(
            {
                "status": report["aggregation_status"],
                "runs": len(report["runs"]),
                "paired_comparisons": len(report["paired_comparisons"]),
                **paths,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
