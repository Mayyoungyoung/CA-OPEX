"""Deterministic episode-level audit for action-noise Q/V predictions.

This script is intentionally separate from the offline-RL trainers.  It joins
an action-noise dataset, a strict episode-OOF prediction archive produced by
``qv_predictor.py``, and (optionally) the logged/clean-action open-loop replay
audit.  It writes every episode-level value used in the analysis and derives
all reported correlations and quantile tables from that CSV.

The clean-action replay is only a finite-horizon open-loop mechanism label.  It
must not be interpreted as the expected return of the behavior policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr


SIGNALS: Tuple[str, ...] = (
    "raw_hubl_score",
    "oof_v_next_mean",
    "oof_v_current_mean",
    "oof_q_current_mean",
    "oof_advantage_mean",
    "oof_v_next_uncertainty",
    "oof_advantage_uncertainty",
)

FULL_TARGETS: Tuple[str, ...] = (
    "raw_return",
    "return_per_step",
    "episode_length",
    "action_delta_l2_mean",
    "action_delta_abs_mean",
    "action_clip_fraction",
    "sampled_noise_l2_mean",
    "raw_return_length_residual",
)

COUNTERFACTUAL_TARGETS: Tuple[str, ...] = (
    "clean_open_loop_return",
    "raw_minus_clean_open_loop_return",
    "clean_open_loop_steps",
    "clean_open_loop_ended_early",
)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def episode_ranges(terminals: np.ndarray, timeouts: np.ndarray) -> List[Tuple[int, int]]:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=bool).reshape(-1)
    if terminals.shape != timeouts.shape or terminals.size == 0:
        raise ValueError("terminal/timeout arrays must be non-empty with matching shape")
    stops = (np.flatnonzero(terminals | timeouts) + 1).tolist()
    if not stops or stops[-1] != terminals.size:
        stops.append(int(terminals.size))
    starts = [0, *stops[:-1]]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def finite_number(value: float) -> Optional[float]:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"n": int(x.size), "pearson": None, "spearman": None}
    return {
        "n": int(x.size),
        "pearson": finite_number(pearsonr(x, y).statistic),
        "spearman": finite_number(spearmanr(x, y).statistic),
    }


def numeric_summary(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def quantile_selection(
    score: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    fraction: float,
) -> Dict[str, object]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if not 0.0 < fraction < 0.5:
        raise ValueError("selection fraction must be in (0, 0.5)")
    if not np.all(np.isfinite(score)):
        raise ValueError("selection scores must be finite")
    count = max(1, int(math.ceil(fraction * score.size)))
    order = np.argsort(score, kind="stable")
    bottom = order[:count]
    top = order[-count:]
    result: Dict[str, object] = {
        "population_count": int(score.size),
        "selected_count": count,
        "fraction": fraction,
        "top_indices": top.astype(int).tolist(),
        "bottom_indices": bottom.astype(int).tolist(),
        "targets": {},
    }
    for name, raw in targets.items():
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.shape != score.shape:
            raise ValueError(f"target {name} shape does not match score")
        overall = float(values.mean())
        top_mean = float(values[top].mean())
        bottom_mean = float(values[bottom].mean())
        result["targets"][name] = {
            "overall_mean": overall,
            "top_mean": top_mean,
            "bottom_mean": bottom_mean,
            "top_minus_overall": top_mean - overall,
            "top_minus_bottom": top_mean - bottom_mean,
        }
    return result


def decile_table(
    score: np.ndarray, targets: Mapping[str, np.ndarray], bins: int
) -> List[Dict[str, object]]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    order = np.argsort(score, kind="stable")
    chunks = np.array_split(order, bins)
    table: List[Dict[str, object]] = []
    for bin_index, indices in enumerate(chunks):
        row: Dict[str, object] = {
            "bin": bin_index,
            "count": int(indices.size),
            "score_min": float(score[indices].min()),
            "score_max": float(score[indices].max()),
            "score_mean": float(score[indices].mean()),
        }
        for name, raw in targets.items():
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            row[name] = float(values[indices].mean())
        table.append(row)
    return table


def _linear_residual(target: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    covariate = np.asarray(covariate, dtype=np.float64).reshape(-1)
    design = np.stack((np.ones_like(covariate), covariate), axis=1)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return target - design @ coefficients


def _percentile_interval(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"replicates": 0, "mean": None, "lower_95": None, "upper_95": None}
    return {
        "replicates": int(values.size),
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "fraction_above_zero": float(np.mean(values > 0.0)),
    }


def paired_bootstrap_vs_reference(
    signals: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    reference: str,
    candidates: Sequence[str],
    replicates: int,
    seed: int,
    top_fraction: float,
) -> Dict[str, object]:
    """Paired episode bootstrap for candidate-minus-reference comparisons.

    The intervals quantify finite-episode sampling uncertainty only.  They do
    not cover model-training seeds, fold choices, or counterfactual replay
    stochasticity.
    """

    if replicates < 1:
        return {}
    reference_values = np.asarray(signals[reference], dtype=np.float64)
    population = int(reference_values.size)
    for name, values in {**signals, **targets}.items():
        if np.asarray(values).shape != (population,):
            raise ValueError(f"bootstrap field {name} has inconsistent shape")
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, population, size=(replicates, population))
    top_count = max(1, int(math.ceil(top_fraction * population)))
    result: Dict[str, object] = {}
    for candidate in candidates:
        candidate_values = np.asarray(signals[candidate], dtype=np.float64)
        by_target: Dict[str, object] = {}
        for target_name, raw_target in targets.items():
            target = np.asarray(raw_target, dtype=np.float64)
            correlation_differences = np.empty(replicates, dtype=np.float64)
            top_mean_differences = np.empty(replicates, dtype=np.float64)
            for replicate, indices in enumerate(sampled_indices):
                sampled_reference = reference_values[indices]
                sampled_candidate = candidate_values[indices]
                sampled_target = target[indices]
                reference_correlation = spearmanr(
                    sampled_reference, sampled_target
                ).statistic
                candidate_correlation = spearmanr(
                    sampled_candidate, sampled_target
                ).statistic
                correlation_differences[replicate] = (
                    candidate_correlation - reference_correlation
                )
                reference_top = np.argsort(sampled_reference, kind="stable")[-top_count:]
                candidate_top = np.argsort(sampled_candidate, kind="stable")[-top_count:]
                top_mean_differences[replicate] = (
                    sampled_target[candidate_top].mean()
                    - sampled_target[reference_top].mean()
                )
            by_target[target_name] = {
                "spearman_difference_candidate_minus_reference": _percentile_interval(
                    correlation_differences
                ),
                "top_fraction_mean_difference_candidate_minus_reference": _percentile_interval(
                    top_mean_differences
                ),
            }
        result[candidate] = by_target
    return {
        "reference": reference,
        "replicates": replicates,
        "seed": seed,
        "interval": "episode bootstrap percentile 95%",
        "scope": (
            "finite-episode sampling only; excludes fit seed, fold, and replay uncertainty"
        ),
        "comparisons": result,
    }


def _episode_means(values: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    values = np.asarray(values)
    return np.asarray([values[start:stop].mean() for start, stop in ranges])


def _episode_sums(values: np.ndarray, ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    values = np.asarray(values)
    return np.asarray([values[start:stop].sum(dtype=np.float64) for start, stop in ranges])


def _load_counterfactual(
    path: Path,
    *,
    dataset_sha256: str,
    raw_return: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != dataset_sha256:
        raise ValueError("counterfactual audit dataset SHA-256 does not match")
    records = payload.get("episodes", [])
    if not records:
        raise ValueError("counterfactual audit contains no episode records")
    ids = np.asarray([record["episode_id"] for record in records], dtype=np.int64)
    if np.unique(ids).size != ids.size or np.any(ids < 0) or np.any(ids >= raw_return.size):
        raise ValueError("counterfactual episode ids are duplicated or out of range")
    raw_recorded = np.asarray([record["raw_noisy_return"] for record in records])
    max_raw_error = float(np.max(np.abs(raw_recorded - raw_return[ids])))
    if max_raw_error > 1e-4:
        raise ValueError(f"counterfactual raw returns disagree by up to {max_raw_error}")
    clean = np.asarray(
        [record["clean_action_replay_return"] for record in records], dtype=np.float64
    )
    clean_steps = np.asarray(
        [record["clean_action_replay"]["steps"] for record in records], dtype=np.float64
    )
    clean_early = np.asarray(
        [record["clean_action_replay"]["ended_early"] for record in records],
        dtype=np.float64,
    )
    logged_pass = np.asarray(
        [record["logged_replay"]["reproduction_pass"] for record in records],
        dtype=bool,
    )
    targets = {
        "clean_open_loop_return": clean,
        "raw_minus_clean_open_loop_return": raw_return[ids] - clean,
        "clean_open_loop_steps": clean_steps,
        "clean_open_loop_ended_early": clean_early,
    }
    integrity = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "episode_count": int(ids.size),
        "episode_ids": ids.tolist(),
        "logged_reproduction_pass_count": int(logged_pass.sum()),
        "logged_reproduction_failure_count": int((~logged_pass).sum()),
        "clean_open_loop_early_end_count": int(clean_early.sum()),
        "raw_return_max_abs_join_error": max_raw_error,
        "interpretation": payload.get("interpretation"),
    }
    return ids, targets, integrity


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    dataset = Path(args.dataset)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha = sha256_file(dataset)

    with h5py.File(dataset, "r") as handle:
        required = (
            "actions",
            "clean_policy_actions",
            "sampled_action_noise",
            "rewards",
            "terminals",
            "timeouts",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"dataset fields missing: {missing}")
        actions = np.asarray(handle["actions"], dtype=np.float64)
        clean_actions = np.asarray(handle["clean_policy_actions"], dtype=np.float64)
        sampled_noise = np.asarray(handle["sampled_action_noise"], dtype=np.float64)
        rewards = np.asarray(handle["rewards"], dtype=np.float64).reshape(-1)
        terminals = np.asarray(handle["terminals"], dtype=bool).reshape(-1)
        timeouts = np.asarray(handle["timeouts"], dtype=bool).reshape(-1)
        saved_delta = (
            np.asarray(handle["applied_action_delta"], dtype=np.float64)
            if "applied_action_delta" in handle
            else None
        )

    ranges = episode_ranges(terminals, timeouts)
    episode_count = len(ranges)
    episode_lengths = np.asarray([stop - start for start, stop in ranges], dtype=np.int64)
    raw_return = _episode_sums(rewards, ranges)
    return_per_step = raw_return / episode_lengths
    action_delta = actions - clean_actions
    saved_delta_error = None
    if saved_delta is not None:
        saved_delta_error = float(np.max(np.abs(saved_delta - action_delta)))
        if saved_delta_error > 1e-6:
            raise ValueError("saved applied_action_delta disagrees with actions-clean_actions")
    delta_l2 = np.linalg.norm(action_delta, axis=1)
    sampled_noise_l2 = np.linalg.norm(sampled_noise, axis=1)
    clip = np.any(np.abs(actions) >= 1.0 - 1e-6, axis=1).astype(np.float64)

    with np.load(predictions_path, allow_pickle=False) as predictions:
        prediction_sha = str(np.asarray(predictions["dataset_sha256"]).item())
        if prediction_sha != dataset_sha:
            raise ValueError("prediction archive dataset SHA-256 does not match")
        if not np.array_equal(predictions["episode_lengths"], episode_lengths):
            raise ValueError("prediction episode lengths do not match dataset")
        predicted_episode_ids = np.asarray(predictions["episode_ids"], dtype=np.int64)
        predicted_sources = np.asarray(predictions["kept_source_indices"], dtype=np.int64)
        expected_episode_ids: List[np.ndarray] = []
        expected_sources: List[np.ndarray] = []
        for episode_id, (start, stop) in enumerate(ranges):
            sources = np.arange(start, stop, dtype=np.int64)
            if timeouts[stop - 1] or not (terminals[stop - 1] or timeouts[stop - 1]):
                sources = sources[:-1]
            expected_sources.append(sources)
            expected_episode_ids.append(np.full(sources.size, episode_id, dtype=np.int64))
        if not np.array_equal(predicted_sources, np.concatenate(expected_sources)):
            raise ValueError("prediction kept source order does not match dataset")
        if not np.array_equal(predicted_episode_ids, np.concatenate(expected_episode_ids)):
            raise ValueError("prediction kept episode order does not match dataset")
        signals = {
            "raw_hubl_score": np.asarray(predictions["noisy_episode_scores"], dtype=np.float64),
            "oof_v_next_mean": np.asarray(predictions["crossfit_mean"], dtype=np.float64),
            "oof_v_current_mean": np.asarray(
                predictions["episode_v_current_mean"], dtype=np.float64
            ),
            "oof_q_current_mean": np.asarray(
                predictions["episode_q_current_mean"], dtype=np.float64
            ),
            "oof_advantage_mean": np.asarray(
                predictions["controllable_score"], dtype=np.float64
            ),
            "oof_v_next_uncertainty": np.asarray(
                predictions["crossfit_std"], dtype=np.float64
            ),
            "oof_advantage_uncertainty": np.asarray(
                predictions["controllable_std"], dtype=np.float64
            ),
        }
        prediction_config = json.loads(str(np.asarray(predictions["config_json"]).item()))

    for name, values in signals.items():
        if values.shape != (episode_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"signal {name} has invalid shape or non-finite values")

    targets: Dict[str, np.ndarray] = {
        "raw_return": raw_return,
        "return_per_step": return_per_step,
        "episode_length": episode_lengths.astype(np.float64),
        "action_delta_l2_mean": _episode_means(delta_l2, ranges),
        "action_delta_abs_mean": _episode_means(np.abs(action_delta).mean(axis=1), ranges),
        "action_clip_fraction": _episode_means(clip, ranges),
        "sampled_noise_l2_mean": _episode_means(sampled_noise_l2, ranges),
        "raw_return_length_residual": _linear_residual(raw_return, episode_lengths),
    }

    cf_ids: Optional[np.ndarray] = None
    cf_targets: Dict[str, np.ndarray] = {}
    cf_integrity: Optional[Dict[str, object]] = None
    if args.counterfactual is not None:
        cf_ids, cf_targets, cf_integrity = _load_counterfactual(
            Path(args.counterfactual), dataset_sha256=dataset_sha, raw_return=raw_return
        )

    correlations = {
        signal: {target: correlation(values, targets[target]) for target in FULL_TARGETS}
        for signal, values in signals.items()
    }
    signal_correlations = {
        left: {right: correlation(signals[left], signals[right]) for right in SIGNALS}
        for left in SIGNALS
    }
    top_fraction = float(args.top_fraction)
    selections = {
        signal: quantile_selection(values, targets, fraction=top_fraction)
        for signal, values in signals.items()
    }
    deciles = {
        signal: decile_table(values, targets, args.quantile_bins)
        for signal, values in signals.items()
    }
    bootstrap_candidates = (
        "oof_v_next_mean",
        "oof_q_current_mean",
        "oof_advantage_mean",
    )
    bootstrap_targets = {
        name: targets[name]
        for name in (
            "raw_return",
            "raw_return_length_residual",
            "action_delta_l2_mean",
        )
    }
    bootstrap_full = paired_bootstrap_vs_reference(
        signals,
        bootstrap_targets,
        reference="raw_hubl_score",
        candidates=bootstrap_candidates,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        top_fraction=top_fraction,
    )

    top_count = max(1, int(math.ceil(top_fraction * episode_count)))
    top_sets = {
        name: set(np.argsort(values, kind="stable")[-top_count:].astype(int).tolist())
        for name, values in signals.items()
    }
    top_overlap: Dict[str, Dict[str, float]] = {}
    for left in SIGNALS:
        top_overlap[left] = {}
        for right in SIGNALS:
            intersection = len(top_sets[left] & top_sets[right])
            union = len(top_sets[left] | top_sets[right])
            top_overlap[left][right] = float(intersection / union)

    cf_analysis: Optional[Dict[str, object]] = None
    if cf_ids is not None:
        cf_signals = {name: values[cf_ids] for name, values in signals.items()}
        cf_correlations = {
            signal: {
                target: correlation(values, cf_targets[target])
                for target in COUNTERFACTUAL_TARGETS
            }
            for signal, values in cf_signals.items()
        }
        cf_selections = {
            signal: quantile_selection(values, cf_targets, fraction=top_fraction)
            for signal, values in cf_signals.items()
        }
        cf_analysis = {
            "integrity": cf_integrity,
            "target_summaries": {
                name: numeric_summary(values) for name, values in cf_targets.items()
            },
            "correlations": cf_correlations,
            "within_subset_quantile_selections": cf_selections,
            "paired_bootstrap_vs_raw_hubl": paired_bootstrap_vs_reference(
                cf_signals,
                {
                    name: cf_targets[name]
                    for name in (
                        "clean_open_loop_return",
                        "raw_minus_clean_open_loop_return",
                    )
                },
                reference="raw_hubl_score",
                candidates=bootstrap_candidates,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 1,
                top_fraction=top_fraction,
            ),
        }

    fields: Dict[str, np.ndarray] = {
        "episode_id": np.arange(episode_count, dtype=np.int64),
        **targets,
        **signals,
    }
    if cf_ids is not None:
        for name in COUNTERFACTUAL_TARGETS:
            joined = np.full(episode_count, np.nan, dtype=np.float64)
            joined[cf_ids] = cf_targets[name]
            fields[name] = joined
    csv_path = output_dir / "episode_metrics.csv"
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for index in range(episode_count):
            writer.writerow({name: values[index] for name, values in fields.items()})
    os.replace(csv_tmp, csv_path)

    result: Dict[str, object] = {
        "schema_version": 1,
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": dataset_sha,
            "bytes": dataset.stat().st_size,
            "transitions": int(rewards.size),
            "episodes": episode_count,
            "terminal_episodes": int(terminals.sum()),
            "timeout_episodes": int(timeouts.sum()),
            "tail_has_boundary": bool(terminals[-1] or timeouts[-1]),
            "saved_action_delta_max_abs_error": saved_delta_error,
        },
        "predictions": {
            "path": str(predictions_path.resolve()),
            "sha256": sha256_file(predictions_path),
            "archive_dataset_sha256": prediction_sha,
            "config": prediction_config,
        },
        "episode_csv": str(csv_path.resolve()),
        "episode_csv_sha256": sha256_file(csv_path),
        "signal_summaries": {
            name: numeric_summary(values) for name, values in signals.items()
        },
        "target_summaries": {
            name: numeric_summary(values) for name, values in targets.items()
        },
        "all_signal_target_correlations": correlations,
        "all_signal_pair_correlations": signal_correlations,
        "top_set_jaccard": top_overlap,
        "quantile_selections": selections,
        "quantile_tables": deciles,
        "paired_bootstrap_vs_raw_hubl": bootstrap_full,
        "counterfactual_subset": cf_analysis,
        "analysis_settings": {
            "top_fraction": top_fraction,
            "quantile_bins": int(args.quantile_bins),
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "ranking_tie_break": "stable episode order",
            "action_clip_definition": "any executed action dimension abs(a)>=1-1e-6",
            "raw_hubl_score_definition": (
                "mean discounted future return over kept transitions, from prediction archive"
            ),
            "oof_definition": "held-out episode predictions only",
        },
        "source_code_sha256": sha256_file(Path(__file__)),
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    json_path = output_dir / "audit_metrics.json"
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    json_tmp.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(json_tmp, json_path)
    result["metrics_json"] = str(json_path.resolve())
    result["metrics_json_sha256"] = sha256_file(json_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quantile_bins < 2:
        raise ValueError("quantile-bins must be at least 2")
    if args.bootstrap_replicates < 0:
        raise ValueError("bootstrap-replicates must be non-negative")
    result = run(args)
    summary = {
        "metrics_json": result["metrics_json"],
        "metrics_json_sha256": result["metrics_json_sha256"],
        "episode_csv": result["episode_csv"],
        "episode_csv_sha256": result["episode_csv_sha256"],
        "episodes": result["dataset"]["episodes"],
        "wall_time_seconds": result["wall_time_seconds"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
