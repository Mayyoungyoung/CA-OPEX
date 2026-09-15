import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_td3bc_results import (
    PairingError,
    _expand_run_specs,
    aggregate_across_training_seeds,
    build_report,
    paired_bootstrap_mean_ci,
    paired_episode_differences,
    parse_run_directory,
)


def condition(returns, env_seeds=(10, 11), noise_seeds=None):
    values = list(returns)
    result = {
        "returns": values,
        "return_mean": sum(values) / len(values),
        "return_std": 0.5 if values == [1.0, 2.0] else 0.0,
        "normalized_score_mean": 1.0,
        "normalized_score_std": 0.0,
    }
    if noise_seeds is None:
        result["episode_seeds"] = list(env_seeds)
    else:
        result.update(
            {
                "environment_seeds": list(env_seeds),
                "action_noise_seeds": list(noise_seeds),
                "action_noise_beta": 1.0,
                "action_noise_distribution": "iid_uniform_minus_beta_plus_beta_per_step",
            }
        )
    return result


def write_run(path: Path, *, seed=0, eval_env_seeds=(10, 11), noise_seeds=(20, 21)):
    path.mkdir()
    config = {
        "arguments": {
            "variant": "hubl_horizon",
            "action_pairing": "executed_executed",
            "train_seed": seed,
            "updates": 50,
            "env_name": "Walker2d-v4",
            "alpha": 2.5,
            "heuristic_discount": 1.0,
            "horizon_noise_scale": 0.02,
            "reference_min_score": 0.0,
            "reference_max_score": 100.0,
        },
        "dataset": {"sha256": "dataset-hash"},
    }
    summary = {
        "status": "complete",
        "variant": "hubl_horizon",
        "action_pairing": "executed_executed",
        "train_seed": seed,
        "updates": 50,
        "wall_time_seconds": 12.5,
    }
    evaluation = {
        "status": "complete",
        "environment": "Walker2d-v4",
        "checkpoint": "latest.pt",
        "checkpoint_sha256": "checkpoint-hash",
        "checkpoint_step": 50,
        "wall_time_seconds": 1.5,
        "command_scale": 1.0,
        "command_transform": "identity",
        "command_transform_beta": 1.0,
        "clean": condition([1.0, 2.0], eval_env_seeds),
        "persistent_action_noise": condition(
            [1.0, 2.0], eval_env_seeds, noise_seeds
        ),
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "independent_eval_2.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )


def test_paired_bootstrap_is_fixed_seed_and_uses_pairs():
    first = paired_bootstrap_mean_ci([2.0, 2.0, 2.0], seed=7, samples=100)
    second = paired_bootstrap_mean_ci([2.0, 2.0, 2.0], seed=7, samples=100)
    assert first == second == {"mean": 2.0, "ci_low": 2.0, "ci_high": 2.0}


def test_pairing_strictly_rejects_environment_and_noise_seed_mismatch():
    target = condition([1.0, 2.0], noise_seeds=(20, 21))
    reference = condition([0.0, 1.0], noise_seeds=(20, 22))
    with pytest.raises(PairingError, match="action-noise seed arrays differ"):
        paired_episode_differences(
            target,
            reference,
            condition="persistent_action_noise",
            target_environment="Walker2d-v4",
            reference_environment="Walker2d-v4",
        )
    with pytest.raises(PairingError, match="environment mismatch"):
        paired_episode_differences(
            target,
            target,
            condition="persistent_action_noise",
            target_environment="Walker2d-v4",
            reference_environment="Hopper-v4",
        )
    reference_same_noise = condition(
        [0.0, 1.0], env_seeds=(10, 12), noise_seeds=(20, 21)
    )
    with pytest.raises(PairingError, match="environment seed arrays differ"):
        paired_episode_differences(
            target,
            reference_same_noise,
            condition="persistent_action_noise",
            target_environment="Walker2d-v4",
            reference_environment="Walker2d-v4",
        )


def test_parse_run_references_raw_returns_without_copying_them(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir)
    run = parse_run_directory(run_dir)
    assert run["run_status"] == "complete"
    assert len(run["rows"]) == 2
    row = run["rows"][0]
    assert "returns" not in row
    assert row["raw_returns_json_pointer"].endswith("/returns")
    assert len(row["source_sha256"]) == 64
    assert row["eval_episodes"] == 2


def test_missing_summary_is_never_complete(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir)
    (run_dir / "summary.json").unlink()
    run = parse_run_directory(run_dir)
    assert run["run_status"] == "incomplete_provenance"
    assert run["training_status"] == "missing_or_unfinished"
    assert run["rows"][0]["run_status"] == "incomplete_provenance"


def test_across_seed_count_does_not_count_evaluation_episodes():
    common = {
        "environment": "Walker2d-v4",
        "variant": "hubl_horizon",
        "action_pairing": "executed_executed",
        "hubl_alpha": 1.0,
        "horizon_c": 0.02,
        "td3bc_alpha": 2.5,
        "command_scale": 1.0,
        "command_transform": "identity",
        "command_transform_beta": 1.0,
        "checkpoint_step": 50,
        "condition": "clean",
        "action_noise_beta": 0.0,
        "protocol_fingerprint": "same-protocol",
        "run_status": "complete",
    }
    rows = [
        {
            **common,
            "run_id": "seed0",
            "train_seed": 0,
            "eval_episodes": 50,
            "return_mean": 10.0,
            "normalized_score_mean": 20.0,
        },
        {
            **common,
            "run_id": "seed1",
            "train_seed": 1,
            "eval_episodes": 50,
            "return_mean": 14.0,
            "normalized_score_mean": 24.0,
        },
    ]
    aggregate = aggregate_across_training_seeds(rows)[0]
    assert aggregate["n_train_seeds"] == 2
    assert aggregate["n_evaluation_episodes_total_descriptive_only"] == 100
    assert aggregate["normalized_score_mean_across_train_seeds"] == 22.0
    assert aggregate["normalized_score_sample_std_across_train_seeds"] == pytest.approx(
        2.8284271247461903
    )


def test_evaluation_controls_are_distinct_aggregate_keys_not_duplicate_seeds():
    common = {
        "environment": "Walker2d-v4",
        "variant": "td3bc",
        "action_pairing": "executed_executed",
        "hubl_alpha": 1.0,
        "horizon_c": None,
        "td3bc_alpha": 2.5,
        "command_scale": 1.0,
        "command_transform_beta": 1.0,
        "checkpoint_step": 50,
        "condition": "persistent_action_noise",
        "action_noise_beta": 1.0,
        "protocol_fingerprint": "same-seeds",
        "run_status": "complete",
        "run_id": "seed0",
        "train_seed": 0,
        "eval_episodes": 50,
        "return_mean": 10.0,
        "normalized_score_mean": 20.0,
    }
    aggregates = aggregate_across_training_seeds(
        [
            {**common, "command_transform": "identity"},
            {**common, "command_transform": "uniform_mean_inverse"},
        ]
    )
    assert len(aggregates) == 2
    assert {item["command_transform"] for item in aggregates} == {
        "identity",
        "uniform_mean_inverse",
    }
    assert all(item["status"] == "complete" for item in aggregates)


def test_build_report_computes_only_valid_seed_paired_differences(tmp_path):
    reference = tmp_path / "reference"
    target = tmp_path / "target"
    write_run(reference, seed=0)
    write_run(target, seed=1)
    target_eval_path = target / "independent_eval_2.json"
    target_eval = json.loads(target_eval_path.read_text(encoding="utf-8"))
    for name in ("clean", "persistent_action_noise"):
        target_eval[name]["returns"] = [3.0, 4.0]
        target_eval[name]["return_mean"] = 3.5
        target_eval[name]["return_std"] = 0.5
    target_eval_path.write_text(json.dumps(target_eval), encoding="utf-8")

    report = build_report(
        [target],
        reference_path=reference,
        bootstrap_seed=9,
        bootstrap_samples=100,
    )
    assert report["aggregation_status"] == "complete"
    assert len(report["paired_comparisons"]) == 2
    assert all(item["status"] == "complete" for item in report["paired_comparisons"])
    assert all(item["positive_pairs"] == 2 for item in report["paired_comparisons"])
    assert all(
        item["return_mean_difference_target_minus_reference"] == 2.0
        for item in report["paired_comparisons"]
    )
    encoded = json.dumps(report)
    assert '"returns"' not in encoded


def test_glob_discovery_ignores_unrelated_files_but_keeps_explicit_missing(tmp_path):
    run_dir = tmp_path / "run"
    write_run(run_dir)
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")
    discovered = _expand_run_specs([str(tmp_path / "*")])
    assert discovered == [run_dir.resolve()]
    missing = tmp_path / "planned-but-not-started"
    assert _expand_run_specs([str(missing)]) == [missing.resolve()]
