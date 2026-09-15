#!/usr/bin/env python3
"""Deterministically aggregate TD3+BC experiment records.

The aggregator deliberately has no torch/numpy dependency.  It keeps episode-level
returns in their original JSON files and emits only aggregate statistics plus an
auditable source path, JSON pointer, and SHA-256 digest.  Evaluation episodes are
never counted as independent training seeds.
"""

from __future__ import annotations

import argparse
import csv
import glob
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


SCHEMA_VERSION = "td3bc-aggregate-v1"
CONDITIONS = ("clean", "persistent_action_noise")


class PairingError(ValueError):
    """Raised when two evaluations do not have an identical paired protocol."""


def canonical_json(value: Any) -> str:
    """Return a stable compact JSON representation used for fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def percentile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile, matching a common quantile definition."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    seed: int,
    samples: int,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap a paired mean, resampling *episode pairs* with replacement."""

    values = [float(value) for value in differences]
    if not values:
        raise ValueError("paired bootstrap requires at least one episode pair")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(int(seed))
    count = len(values)
    bootstrap_means = []
    for _ in range(samples):
        bootstrap_means.append(
            sum(values[rng.randrange(count)] for _ in range(count)) / count
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": statistics.fmean(values),
        "ci_low": percentile(bootstrap_means, tail),
        "ci_high": percentile(bootstrap_means, 1.0 - tail),
    }


def _require_numeric_list(record: Mapping[str, Any], key: str) -> List[float]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise PairingError(f"missing or empty {key!r}")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise PairingError(f"{key!r} must contain only numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise PairingError(f"{key!r} contains a non-finite value")
    return result


def _require_exact_seed_list(record: Mapping[str, Any], key: str) -> List[int]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise PairingError(f"missing or empty {key!r}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PairingError(f"{key!r} must contain integer seeds")
    return list(value)


def paired_episode_differences(
    target: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    condition: str,
    target_environment: Optional[str],
    reference_environment: Optional[str],
) -> Tuple[List[float], Dict[str, Any]]:
    """Validate an exact episode protocol, then return target-reference returns.

    This is intentionally strict: missing seeds, reordered seeds, different noise
    seeds, environments, beta values, or distributions prevent paired statistics.
    """

    if condition not in CONDITIONS:
        raise PairingError(f"unsupported condition {condition!r}")
    if not target_environment or not reference_environment:
        raise PairingError("both evaluations must record the environment")
    if target_environment != reference_environment:
        raise PairingError(
            f"environment mismatch: {target_environment!r} != {reference_environment!r}"
        )

    if condition == "clean":
        target_environment_seeds = _require_exact_seed_list(target, "episode_seeds")
        reference_environment_seeds = _require_exact_seed_list(reference, "episode_seeds")
        target_noise_seeds: Optional[List[int]] = None
        reference_noise_seeds: Optional[List[int]] = None
    else:
        target_environment_seeds = _require_exact_seed_list(target, "environment_seeds")
        reference_environment_seeds = _require_exact_seed_list(reference, "environment_seeds")
        target_noise_seeds = _require_exact_seed_list(target, "action_noise_seeds")
        reference_noise_seeds = _require_exact_seed_list(reference, "action_noise_seeds")
        if target_noise_seeds != reference_noise_seeds:
            raise PairingError("action-noise seed arrays differ")
        target_beta = target.get("action_noise_beta")
        reference_beta = reference.get("action_noise_beta")
        if target_beta is None or reference_beta is None:
            raise PairingError("persistent evaluations must record action_noise_beta")
        if float(target_beta) != float(reference_beta):
            raise PairingError(
                f"action-noise beta mismatch: {target_beta!r} != {reference_beta!r}"
            )
        if target.get("action_noise_distribution") != reference.get(
            "action_noise_distribution"
        ):
            raise PairingError("action-noise distributions differ")

    if target_environment_seeds != reference_environment_seeds:
        raise PairingError("environment seed arrays differ")
    target_returns = _require_numeric_list(target, "returns")
    reference_returns = _require_numeric_list(reference, "returns")
    if len(target_returns) != len(reference_returns):
        raise PairingError("return arrays have different lengths")
    if len(target_returns) != len(target_environment_seeds):
        raise PairingError("return count does not match environment-seed count")
    if target_noise_seeds is not None and len(target_returns) != len(target_noise_seeds):
        raise PairingError("return count does not match action-noise-seed count")

    differences = [target_value - reference_value for target_value, reference_value in zip(
        target_returns, reference_returns
    )]
    validation = {
        "environment_equal": True,
        "environment_seed_arrays_equal": True,
        "action_noise_seed_arrays_equal": True if condition != "clean" else "not_applicable",
        "episode_pairs": len(differences),
    }
    return differences, validation


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _sample_std(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate_across_training_seeds(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate run-level means with training seed as the statistical unit."""

    identity_fields = (
        "environment",
        "variant",
        "action_pairing",
        "hubl_alpha",
        "horizon_c",
        "td3bc_alpha",
        "command_scale",
        "command_transform",
        "command_transform_beta",
        "checkpoint_step",
        "condition",
        "action_noise_beta",
        "protocol_fingerprint",
    )
    grouped: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in identity_fields)].append(row)

    aggregates: List[Dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: canonical_json(item)):
        group_rows = grouped[key]
        identity = dict(zip(identity_fields, key))
        by_seed: MutableMapping[Any, List[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_seed[row.get("train_seed")].append(row)
        duplicate_seeds = sorted(
            (seed for seed, items in by_seed.items() if seed is not None and len(items) > 1),
            key=str,
        )
        missing_seed = None in by_seed
        per_seed = []
        for seed in sorted((seed for seed in by_seed if seed is not None), key=str):
            items = by_seed[seed]
            per_seed.append(
                {
                    "train_seed": seed,
                    "n_run_records": len(items),
                    "run_ids": [item.get("run_id") for item in items],
                    "return_means": [item.get("return_mean") for item in items],
                    "normalized_score_means": [
                        item.get("normalized_score_mean") for item in items
                    ],
                    "evaluation_episodes": [item.get("eval_episodes") for item in items],
                }
            )

        valid_unique = not duplicate_seeds and not missing_seed
        run_statuses = [row.get("run_status") for row in group_rows]
        complete_inputs = all(status == "complete" for status in run_statuses)
        if missing_seed:
            status = "invalid_missing_train_seed"
        elif duplicate_seeds:
            status = "invalid_duplicate_train_seed_runs"
        elif not complete_inputs:
            status = "incomplete_input_provenance"
        else:
            status = "complete"

        return_means = [float(row["return_mean"]) for row in group_rows if row.get("return_mean") is not None]
        normalized_means = [
            float(row["normalized_score_mean"])
            for row in group_rows
            if row.get("normalized_score_mean") is not None
        ]
        aggregate = {
            **identity,
            "status": status,
            "statistical_unit": "independent_training_seed",
            "n_train_seeds": len([seed for seed in by_seed if seed is not None]),
            "n_run_records": len(group_rows),
            "n_evaluation_episodes_total_descriptive_only": sum(
                int(row.get("eval_episodes") or 0) for row in group_rows
            ),
            "per_seed": per_seed,
            "duplicate_train_seeds": duplicate_seeds,
            "return_mean_across_train_seeds": (
                _safe_mean(return_means)
                if valid_unique and len(return_means) == len(group_rows)
                else None
            ),
            "return_sample_std_across_train_seeds": (
                _sample_std(return_means)
                if valid_unique and len(return_means) == len(group_rows)
                else None
            ),
            "normalized_score_mean_across_train_seeds": (
                _safe_mean(normalized_means)
                if valid_unique and len(normalized_means) == len(group_rows)
                else None
            ),
            "normalized_score_sample_std_across_train_seeds": (
                _sample_std(normalized_means)
                if valid_unique and len(normalized_means) == len(group_rows)
                else None
            ),
        }
        aggregates.append(aggregate)
    return aggregates


def _read_json_hashed(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    source: Dict[str, Any] = {"path": str(path.resolve()), "role": path.name}
    try:
        payload = path.read_bytes()
    except OSError as exc:
        source.update({"status": "unreadable", "error": str(exc)})
        return None, source
    source.update(
        {
            "status": "read",
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        source.update({"status": "invalid_json", "error": str(exc)})
        return None, source
    if not isinstance(value, dict):
        source.update({"status": "invalid_json_root", "error": "JSON root is not an object"})
        return None, source
    return value, source


def _first(mapping_values: Iterable[Tuple[Mapping[str, Any], str]]) -> Any:
    for mapping, key in mapping_values:
        if isinstance(mapping, Mapping) and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _condition_seed_protocol(condition_name: str, condition: Mapping[str, Any]) -> Dict[str, Any]:
    if condition_name == "clean":
        return {"environment_seeds": condition.get("episode_seeds"), "action_noise_seeds": None}
    return {
        "environment_seeds": condition.get("environment_seeds"),
        "action_noise_seeds": condition.get("action_noise_seeds"),
        "action_noise_beta": condition.get("action_noise_beta"),
        "action_noise_distribution": condition.get("action_noise_distribution"),
    }


def _check_condition(condition_name: str, condition: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    try:
        returns = _require_numeric_list(condition, "returns")
    except PairingError as exc:
        return [str(exc)]
    seed_key = "episode_seeds" if condition_name == "clean" else "environment_seeds"
    seeds = condition.get(seed_key)
    if not isinstance(seeds, list) or len(seeds) != len(returns):
        issues.append(f"{seed_key} count does not match returns")
    if condition_name == "persistent_action_noise":
        noise_seeds = condition.get("action_noise_seeds")
        if not isinstance(noise_seeds, list) or len(noise_seeds) != len(returns):
            issues.append("action_noise_seeds count does not match returns")
        if condition.get("action_noise_beta") is None:
            issues.append("missing action_noise_beta")
    reported_mean = condition.get("return_mean")
    if reported_mean is None:
        issues.append("missing return_mean")
    elif not math.isclose(float(reported_mean), statistics.fmean(returns), rel_tol=1e-9, abs_tol=1e-7):
        issues.append("reported return_mean does not match raw returns")
    reported_std = condition.get("return_std")
    computed_std = statistics.pstdev(returns)
    if reported_std is None:
        issues.append("missing return_std")
    elif not math.isclose(float(reported_std), computed_std, rel_tol=1e-9, abs_tol=1e-7):
        issues.append("reported return_std does not match raw returns")
    if condition.get("normalized_score_mean") is None:
        issues.append("missing normalized_score_mean")
    if condition.get("normalized_score_std") is None:
        issues.append("missing normalized_score_std")
    return issues


def parse_run_directory(run_dir: Path, *, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Read one run directory and return rows plus private pairing documents."""

    path = run_dir.resolve()
    record: Dict[str, Any] = {
        "run_id": run_id or path.name,
        "run_dir": str(path),
        "sources": [],
        "rows": [],
        "issues": [],
        "_evaluation_documents": [],
    }
    if not path.is_dir():
        record.update(
            {
                "training_status": "missing",
                "run_status": "missing_run_directory",
            }
        )
        record["issues"].append("run directory does not exist")
        return record

    config_path = path / "config.json"
    summary_path = path / "summary.json"
    config: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    for source_path, destination in ((config_path, "config"), (summary_path, "summary")):
        if source_path.exists():
            parsed, source = _read_json_hashed(source_path)
            record["sources"].append(source)
            if parsed is None:
                record["issues"].append(f"{source_path.name} is not valid readable JSON")
            elif destination == "config":
                config = parsed
            else:
                summary = parsed
        else:
            record["sources"].append(
                {"path": str(source_path), "role": source_path.name, "status": "missing"}
            )
            record["issues"].append(f"missing {source_path.name}")

    args = config.get("arguments") if isinstance(config.get("arguments"), dict) else {}
    variant = _first(((summary, "variant"), (args, "variant")))
    action_pairing = _first(((summary, "action_pairing"), (args, "action_pairing")))
    train_seed = _first(((summary, "train_seed"), (args, "train_seed")))
    updates = _first(((summary, "updates"), (args, "updates")))
    environment = _first(((args, "env_name"), (config, "environment")))
    hubl_alpha = args.get("heuristic_discount")
    horizon_c = args.get("horizon_noise_scale") if str(variant).startswith("hubl_horizon") else None
    td3bc_alpha = args.get("alpha")
    training_status = summary.get("status") if summary else None
    record.update(
        {
            "variant": variant,
            "action_pairing": action_pairing,
            "train_seed": train_seed,
            "updates": updates,
            "environment": environment,
            "hubl_alpha": hubl_alpha,
            "horizon_c": horizon_c,
            "td3bc_alpha": td3bc_alpha,
            "train_wall_time_seconds": summary.get("wall_time_seconds") if summary else None,
            "training_status": training_status or "missing_or_unfinished",
            "dataset_sha256": (
                config.get("dataset", {}).get("sha256")
                if isinstance(config.get("dataset"), dict)
                else None
            ),
            "_config": config,
        }
    )

    eval_paths = sorted(path.glob("independent_eval*.json"), key=lambda item: item.name)
    if not eval_paths:
        record["issues"].append("no independent_eval*.json file")
    any_invalid_evaluation = False
    for eval_path in eval_paths:
        evaluation, source = _read_json_hashed(eval_path)
        record["sources"].append(source)
        if evaluation is None:
            any_invalid_evaluation = True
            record["issues"].append(f"invalid evaluation file {eval_path.name}")
            continue
        eval_status = evaluation.get("status")
        if eval_status != "complete":
            any_invalid_evaluation = True
            record["issues"].append(
                f"{eval_path.name} status is {eval_status!r}, not 'complete'"
            )
        eval_environment = evaluation.get("environment") or environment
        if evaluation.get("checkpoint_step") is None:
            any_invalid_evaluation = True
            record["issues"].append(f"{eval_path.name} is missing checkpoint_step")
        if not eval_environment:
            any_invalid_evaluation = True
            record["issues"].append(f"{eval_path.name} is missing environment")
        if not any(evaluation.get(name) is not None for name in CONDITIONS):
            any_invalid_evaluation = True
            record["issues"].append(f"{eval_path.name} has no recognized evaluation condition")
        document = {
            "name": eval_path.name,
            "path": str(eval_path.resolve()),
            "sha256": source.get("sha256"),
            "record": evaluation,
            "environment": eval_environment,
        }
        record["_evaluation_documents"].append(document)
        for condition_name in CONDITIONS:
            condition = evaluation.get(condition_name)
            if condition is None:
                continue
            if not isinstance(condition, dict):
                any_invalid_evaluation = True
                record["issues"].append(
                    f"{eval_path.name}:{condition_name} is not an object"
                )
                continue
            condition_issues = _check_condition(condition_name, condition)
            if condition_issues:
                any_invalid_evaluation = True
                record["issues"].extend(
                    f"{eval_path.name}:{condition_name}: {issue}" for issue in condition_issues
                )
            returns = condition.get("returns") if isinstance(condition.get("returns"), list) else []
            protocol = _condition_seed_protocol(condition_name, condition)
            pointer = f"/{condition_name}/returns"
            command_scale = _first(
                ((condition, "command_scale"), (evaluation, "command_scale"))
            )
            command_transform = _first(
                ((condition, "command_transform"), (evaluation, "command_transform"))
            )
            row = {
                "run_id": record["run_id"],
                "run_dir": str(path),
                "run_status": None,
                "training_status": record["training_status"],
                "environment": eval_environment,
                "dataset_sha256": record["dataset_sha256"],
                "checkpoint": evaluation.get("checkpoint"),
                "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
                "checkpoint_step": evaluation.get("checkpoint_step"),
                "train_seed": train_seed,
                "variant": variant,
                "action_pairing": action_pairing,
                "hubl_alpha": hubl_alpha,
                "horizon_c": horizon_c,
                "td3bc_alpha": td3bc_alpha,
                "command_scale": command_scale if command_scale is not None else 1.0,
                "command_transform": command_transform or "identity",
                "command_transform_beta": _first(
                    (
                        (condition, "command_transform_beta"),
                        (evaluation, "command_transform_beta"),
                    )
                ),
                "evaluation_control_provenance": (
                    "explicit"
                    if evaluation.get("command_scale") is not None
                    or condition.get("command_scale") is not None
                    or evaluation.get("command_transform") is not None
                    or condition.get("command_transform") is not None
                    else "legacy_implicit_scale_1_identity"
                ),
                "command_saturation_fraction": condition.get(
                    "command_saturation_fraction"
                ),
                "command_transform_saturation_fraction": condition.get(
                    "command_transform_saturation_fraction"
                ),
                "transformed_command_at_bound_fraction": condition.get(
                    "transformed_command_at_bound_fraction"
                ),
                "condition": condition_name,
                "action_noise_beta": (
                    condition.get("action_noise_beta")
                    if condition_name == "persistent_action_noise"
                    else 0.0
                ),
                "eval_episodes": len(returns),
                "return_mean": condition.get("return_mean"),
                "return_std": condition.get("return_std"),
                "normalized_score_mean": condition.get("normalized_score_mean"),
                "normalized_score_std": condition.get("normalized_score_std"),
                "train_wall_time_seconds": record["train_wall_time_seconds"],
                "eval_wall_time_seconds": evaluation.get("wall_time_seconds"),
                "evaluation_status": eval_status or "missing",
                "evaluation_record_valid": not condition_issues,
                "protocol_fingerprint": stable_fingerprint(protocol),
                "source_file": str(eval_path.resolve()),
                "source_sha256": source.get("sha256"),
                "raw_returns_json_pointer": pointer,
                "environment_seeds_json_pointer": (
                    f"/{condition_name}/episode_seeds"
                    if condition_name == "clean"
                    else f"/{condition_name}/environment_seeds"
                ),
                "action_noise_seeds_json_pointer": (
                    None
                    if condition_name == "clean"
                    else f"/{condition_name}/action_noise_seeds"
                ),
            }
            record["rows"].append(row)

    config_valid = bool(config)
    summary_valid = bool(summary)
    if not config_valid or not summary_valid:
        run_status = "incomplete_provenance"
    elif training_status != "complete":
        run_status = "training_not_complete"
    elif not eval_paths:
        run_status = "independent_evaluation_missing"
    elif any_invalid_evaluation:
        run_status = "evaluation_incomplete_or_invalid"
    else:
        run_status = "complete"
    record["run_status"] = run_status
    for row in record["rows"]:
        row["run_status"] = run_status
    return record


def _match_reference_document(
    target_document: Mapping[str, Any], reference_documents: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    same_name = [doc for doc in reference_documents if doc.get("name") == target_document.get("name")]
    if len(same_name) == 1:
        return same_name[0]
    target_step = target_document.get("record", {}).get("checkpoint_step")
    same_step = [
        doc
        for doc in reference_documents
        if doc.get("record", {}).get("checkpoint_step") == target_step
    ]
    if len(same_step) == 1:
        return same_step[0]
    if len(reference_documents) == 1:
        return reference_documents[0]
    raise PairingError(
        f"cannot uniquely match reference evaluation for {target_document.get('name')!r}"
    )


def _normalization_scale(run: Mapping[str, Any]) -> Optional[float]:
    # Private config anchors are inserted only while constructing pair comparisons.
    minimum = run.get("_reference_min_score")
    maximum = run.get("_reference_max_score")
    if minimum is None or maximum is None or float(maximum) == float(minimum):
        return None
    return 100.0 / (float(maximum) - float(minimum))


def build_paired_comparisons(
    target_runs: Sequence[Mapping[str, Any]],
    reference_run: Mapping[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    reference_documents = reference_run.get("_evaluation_documents", [])
    for target_run in target_runs:
        if target_run.get("run_dir") == reference_run.get("run_dir"):
            continue
        for target_document in target_run.get("_evaluation_documents", []):
            try:
                reference_document = _match_reference_document(
                    target_document, reference_documents
                )
            except PairingError as exc:
                comparisons.append(
                    {
                        "status": "invalid_pairing",
                        "target_run_id": target_run.get("run_id"),
                        "reference_run_id": reference_run.get("run_id"),
                        "target_source_file": target_document.get("path"),
                        "error": str(exc),
                    }
                )
                continue
            for condition_name in CONDITIONS:
                target_condition = target_document.get("record", {}).get(condition_name)
                reference_condition = reference_document.get("record", {}).get(condition_name)
                if target_condition is None and reference_condition is None:
                    continue
                base = {
                    "target_run_id": target_run.get("run_id"),
                    "reference_run_id": reference_run.get("run_id"),
                    "condition": condition_name,
                    "target_checkpoint_step": target_document.get("record", {}).get(
                        "checkpoint_step"
                    ),
                    "reference_checkpoint_step": reference_document.get("record", {}).get(
                        "checkpoint_step"
                    ),
                    "target_source_file": target_document.get("path"),
                    "target_source_sha256": target_document.get("sha256"),
                    "reference_source_file": reference_document.get("path"),
                    "reference_source_sha256": reference_document.get("sha256"),
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_unit": "paired_evaluation_episode",
                }
                if target_condition is None or reference_condition is None:
                    comparisons.append(
                        {
                            **base,
                            "status": "invalid_pairing",
                            "error": "condition is missing from target or reference",
                        }
                    )
                    continue
                try:
                    differences, validation = paired_episode_differences(
                        target_condition,
                        reference_condition,
                        condition=condition_name,
                        target_environment=target_document.get("environment"),
                        reference_environment=reference_document.get("environment"),
                    )
                    bootstrap = paired_bootstrap_mean_ci(
                        differences,
                        seed=bootstrap_seed,
                        samples=bootstrap_samples,
                    )
                except (PairingError, ValueError) as exc:
                    comparisons.append(
                        {**base, "status": "invalid_pairing", "error": str(exc)}
                    )
                    continue
                comparison = {
                    **base,
                    "status": (
                        "complete"
                        if target_run.get("run_status") == "complete"
                        and reference_run.get("run_status") == "complete"
                        else "paired_values_valid_but_provenance_incomplete"
                    ),
                    "validation": validation,
                    "episode_pairs": len(differences),
                    "positive_pairs": sum(value > 0.0 for value in differences),
                    "negative_pairs": sum(value < 0.0 for value in differences),
                    "tied_pairs": sum(value == 0.0 for value in differences),
                    "return_mean_difference_target_minus_reference": bootstrap["mean"],
                    "return_paired_bootstrap_95ci_low": bootstrap["ci_low"],
                    "return_paired_bootstrap_95ci_high": bootstrap["ci_high"],
                }
                target_scale = _normalization_scale(target_run)
                reference_scale = _normalization_scale(reference_run)
                if (
                    target_scale is not None
                    and reference_scale is not None
                    and math.isclose(target_scale, reference_scale, rel_tol=0.0, abs_tol=0.0)
                    and target_run.get("_reference_min_score")
                    == reference_run.get("_reference_min_score")
                ):
                    comparison.update(
                        {
                            "normalized_score_mean_difference_target_minus_reference": bootstrap[
                                "mean"
                            ]
                            * target_scale,
                            "normalized_score_paired_bootstrap_95ci_low": bootstrap["ci_low"]
                            * target_scale,
                            "normalized_score_paired_bootstrap_95ci_high": bootstrap["ci_high"]
                            * target_scale,
                        }
                    )
                else:
                    comparison["normalized_score_difference_status"] = (
                        "unavailable_or_mismatched_normalization_anchors"
                    )
                comparisons.append(comparison)
    return comparisons


def _add_private_normalization_anchors(run: Dict[str, Any]) -> None:
    config = run.get("_config")
    if not isinstance(config, dict):
        return
    args = config.get("arguments", {})
    if isinstance(args, dict):
        run["_reference_min_score"] = args.get("reference_min_score")
        run["_reference_max_score"] = args.get("reference_max_score")


def _public_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_") and key != "rows"}


CSV_FIELDS = (
    "run_id",
    "run_dir",
    "run_status",
    "training_status",
    "environment",
    "dataset_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_step",
    "train_seed",
    "variant",
    "action_pairing",
    "hubl_alpha",
    "horizon_c",
    "td3bc_alpha",
    "command_scale",
    "command_transform",
    "command_transform_beta",
    "evaluation_control_provenance",
    "command_saturation_fraction",
    "command_transform_saturation_fraction",
    "transformed_command_at_bound_fraction",
    "condition",
    "action_noise_beta",
    "eval_episodes",
    "return_mean",
    "return_std",
    "normalized_score_mean",
    "normalized_score_std",
    "train_wall_time_seconds",
    "eval_wall_time_seconds",
    "evaluation_status",
    "evaluation_record_valid",
    "protocol_fingerprint",
    "source_file",
    "source_sha256",
    "raw_returns_json_pointer",
    "environment_seeds_json_pointer",
    "action_noise_seeds_json_pointer",
)


def render_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _md(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TD3+BC experiment aggregate",
        "",
        (
            "Statistical unit for cross-run uncertainty: **independent training seed**. "
            "Evaluation episodes estimate a fixed trained policy and are never reported "
            "as training repetitions."
        ),
        "",
        "## Run provenance",
        "",
        "| Run | Variant | Train seed | Updates | Status | Train wall (s) |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for run in report.get("runs", []):
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    run.get("run_id"),
                    run.get("variant"),
                    run.get("train_seed"),
                    run.get("updates"),
                    run.get("run_status"),
                    run.get("train_wall_time_seconds"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Independent evaluations",
            "",
            "| Run | Step | Seed | Variant | Pairing | HUBL α | c | Cmd scale | Transform | Condition | β | Episodes | Return | Normalized score | Status |",
            "|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report.get("evaluations", []):
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    row.get("run_id"),
                    row.get("checkpoint_step"),
                    row.get("train_seed"),
                    row.get("variant"),
                    row.get("action_pairing"),
                    row.get("hubl_alpha"),
                    row.get("horizon_c"),
                    row.get("command_scale"),
                    row.get("command_transform"),
                    row.get("condition"),
                    row.get("action_noise_beta"),
                    row.get("eval_episodes"),
                    row.get("return_mean"),
                    row.get("normalized_score_mean"),
                    row.get("run_status"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Episode-level returns are not duplicated here. Each row records its source JSON, SHA-256, and JSON pointer in the CSV/JSON outputs.",
            "",
            "## Across-training-seed summaries",
            "",
            "| Variant | Pairing | Cmd scale | Transform | Step | Condition | β | n train seeds | Mean normalized score | Sample SD across seeds | Status |",
            "|---|---|---:|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for aggregate in report.get("training_seed_aggregates", []):
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    aggregate.get("variant"),
                    aggregate.get("action_pairing"),
                    aggregate.get("command_scale"),
                    aggregate.get("command_transform"),
                    aggregate.get("checkpoint_step"),
                    aggregate.get("condition"),
                    aggregate.get("action_noise_beta"),
                    aggregate.get("n_train_seeds"),
                    aggregate.get("normalized_score_mean_across_train_seeds"),
                    aggregate.get("normalized_score_sample_std_across_train_seeds"),
                    aggregate.get("status"),
                )
            )
            + " |"
        )

    comparisons = report.get("paired_comparisons", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired episode comparisons",
                "",
                "Differences are target minus the user-specified reference. The bootstrap resamples paired evaluation episodes, not training seeds.",
                "",
                "| Target | Reference | Condition | Pairs | Positive | Δ normalized score | 95% paired bootstrap CI | Status |",
                "|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for comparison in comparisons:
            low = comparison.get("normalized_score_paired_bootstrap_95ci_low")
            high = comparison.get("normalized_score_paired_bootstrap_95ci_high")
            interval = "—" if low is None or high is None else f"[{low:.4f}, {high:.4f}]"
            lines.append(
                "| "
                + " | ".join(
                    _md(value)
                    for value in (
                        comparison.get("target_run_id"),
                        comparison.get("reference_run_id"),
                        comparison.get("condition"),
                        comparison.get("episode_pairs"),
                        comparison.get("positive_pairs"),
                        comparison.get(
                            "normalized_score_mean_difference_target_minus_reference"
                        ),
                        interval,
                        comparison.get("status"),
                    )
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Input hashes",
            "",
            "| Role | Path | Status | SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for source in report.get("input_files", []):
        lines.append(
            "| "
            + " | ".join(
                _md(value)
                for value in (
                    source.get("role"),
                    source.get("path"),
                    source.get("status"),
                    source.get("sha256"),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def atomic_write_text(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def _expand_run_specs(specs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for spec in specs:
        matches = glob.glob(spec, recursive=True)
        came_from_glob = glob.has_magic(spec)
        if not matches:
            matches = [spec]
            came_from_glob = False
        for match in matches:
            path = Path(match)
            if path.is_file():
                recognized_marker = path.name in {"config.json", "summary.json"} or (
                    path.name.startswith("independent_eval") and path.suffix == ".json"
                )
                if not recognized_marker:
                    continue
                path = path.parent
            elif came_from_glob and path.is_dir():
                has_marker = (path / "config.json").exists() or (path / "summary.json").exists()
                has_marker = has_marker or any(path.glob("independent_eval*.json"))
                if not has_marker:
                    continue
            resolved = path.resolve()
            key = os.path.normcase(str(resolved))
            if key not in seen:
                seen.add(key)
                paths.append(resolved)
    return sorted(paths, key=lambda item: os.path.normcase(str(item)))


def _assign_run_ids(paths: Sequence[Path]) -> Dict[Path, str]:
    counts: MutableMapping[str, int] = defaultdict(int)
    for path in paths:
        counts[path.name] += 1
    return {
        path: (
            path.name
            if counts[path.name] == 1
            else f"{path.name}-{sha256_bytes(str(path).encode('utf-8'))[:8]}"
        )
        for path in paths
    }


def build_report(
    run_paths: Sequence[Path],
    *,
    reference_path: Optional[Path],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    all_paths = list(run_paths)
    if reference_path is not None and reference_path.resolve() not in all_paths:
        all_paths.append(reference_path.resolve())
    run_ids = _assign_run_ids(all_paths)
    parsed_by_path = {
        path: parse_run_directory(path, run_id=run_ids[path]) for path in all_paths
    }
    for run in parsed_by_path.values():
        _add_private_normalization_anchors(run)
    selected_runs = [parsed_by_path[path] for path in run_paths]
    rows = [row for run in selected_runs for row in run.get("rows", [])]
    reference_run = parsed_by_path.get(reference_path.resolve()) if reference_path else None
    paired = (
        build_paired_comparisons(
            selected_runs,
            reference_run,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        )
        if reference_run is not None
        else []
    )
    all_sources = []
    for path in all_paths:
        role_prefix = "reference:" if reference_path and path == reference_path.resolve() else "run:"
        for source in parsed_by_path[path].get("sources", []):
            all_sources.append({**source, "role": role_prefix + str(source.get("role"))})
    has_incomplete_runs = any(run.get("run_status") != "complete" for run in selected_runs)
    has_incomplete_reference = bool(
        reference_run is not None and reference_run.get("run_status") != "complete"
    )
    has_invalid_pairs = any(item.get("status") != "complete" for item in paired)
    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation_status": (
            "incomplete_or_invalid"
            if has_incomplete_runs or has_incomplete_reference or has_invalid_pairs
            else "complete"
        ),
        "statistical_unit": "independent_training_seed",
        "evaluation_episode_note": (
            "Evaluation episodes estimate a fixed trained policy and are not "
            "independent training repeats."
        ),
        "bootstrap": {
            "seed": bootstrap_seed,
            "samples": bootstrap_samples,
            "confidence": 0.95,
            "unit": "paired_evaluation_episode",
        },
        "reference_run": _public_run(reference_run) if reference_run else None,
        "runs": [_public_run(run) for run in selected_runs],
        "evaluations": rows,
        "training_seed_aggregates": aggregate_across_training_seeds(rows),
        "paired_comparisons": paired,
        "input_files": all_sources,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="One or more run directories or glob patterns (expanded internally).",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        help="Optional reference run for strictly seed-paired episode comparisons.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="td3bc_results")
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    run_paths = _expand_run_specs(args.runs)
    report = build_report(
        run_paths,
        reference_path=args.reference_run,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir = args.output_dir.resolve()
    json_path = output_dir / f"{args.stem}.json"
    csv_path = output_dir / f"{args.stem}.csv"
    markdown_path = output_dir / f"{args.stem}.md"
    json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    atomic_write_text(json_path, json_text)
    atomic_write_text(csv_path, render_csv(report["evaluations"]))
    atomic_write_text(markdown_path, render_markdown(report))
    print(
        canonical_json(
            {
                "status": report["aggregation_status"],
                "runs": len(report["runs"]),
                "evaluation_rows": len(report["evaluations"]),
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
