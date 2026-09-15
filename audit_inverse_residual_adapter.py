"""Audit residual-adapter value gains using independent states and MC noise."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import torch

from evaluate_inverse_residual_adapter import load_controller
from inverse_residual_core import (
    ResampledAntitheticChannel,
    baseline_commands_from_desired,
    marginalized_twin_from_physical_actions,
)
from train_inverse_residual_adapter import (
    OBSERVATION_SPLIT_SCHEMA,
    build_observation_split,
    require_new_output_file,
)
from train_iql import save_json, sha256_file


def _summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "positive_fraction": float((values > 0.0).mean()),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path = require_new_output_file(Path(args.output))
    device = torch.device(args.device)
    controller, provenance = load_controller(
        Path(args.adapter_checkpoint).resolve(),
        device,
        Path(args.base_checkpoint).resolve() if args.base_checkpoint else None,
    )
    training_config = controller.adapter.config
    if args.audit_noise_seed == training_config.execution_noise_seed:
        raise ValueError("audit_noise_seed must differ from the training channel seed")
    audit_config = replace(
        training_config,
        execution_noise_samples=args.audit_noise_samples,
        execution_noise_seed=args.audit_noise_seed,
        execution_noise_beta=(
            training_config.execution_noise_beta
            if args.audit_action_noise_beta is None
            else args.audit_action_noise_beta
        ),
    )
    channel = ResampledAntitheticChannel(audit_config, device)
    dataset_path = Path(args.dataset).resolve()
    training_dataset = provenance.get("training_dataset")
    recorded_split = provenance.get("training_observation_split")
    if not isinstance(training_dataset, dict) or not isinstance(
        recorded_split, dict
    ):
        raise ValueError(
            "adapter checkpoint predates the recorded disjoint observation split"
        )
    if recorded_split.get("schema") != OBSERVATION_SPLIT_SCHEMA:
        raise ValueError("adapter checkpoint has an unsupported observation split")
    actual_dataset_sha = sha256_file(dataset_path)
    if actual_dataset_sha != training_dataset.get("sha256"):
        raise ValueError("audit dataset SHA256 differs from the training dataset")
    try:
        max_observations_value = recorded_split["max_observations"]
        max_observations = (
            None
            if max_observations_value is None
            else int(max_observations_value)
        )
        requested_fraction = float(recorded_split["requested_audit_fraction"])
        split_seed = int(recorded_split["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("adapter checkpoint observation split is incomplete") from exc
    observations, train_indices, audit_indices, recomputed_split = (
        build_observation_split(
            dataset_path,
            max_observations,
            requested_fraction,
            split_seed,
        )
    )
    split_comparison_fields = (
        "schema",
        "method",
        "seed",
        "requested_audit_fraction",
        "max_observations",
        "considered_observation_count",
        "train_observation_count",
        "audit_observation_count",
        "train_indices_sha256",
        "audit_indices_sha256",
        "audit_episode_units_sha256",
        "audit_episode_unit_count",
        "holdout_scope",
        "base_critic_training_partition_controlled",
    )
    for field in split_comparison_fields:
        if recomputed_split.get(field) != recorded_split.get(field):
            raise ValueError(
                f"recomputed audit partition disagrees on split field {field}"
            )
    if observations.shape[1] != audit_config.observation_dim:
        raise ValueError("audit dataset observation dimension does not match adapter")
    count = int(audit_indices.size)
    selected = observations[audit_indices]

    q1_gains: List[np.ndarray] = []
    q2_gains: List[np.ndarray] = []
    min_twin_gains: List[np.ndarray] = []
    baseline_disagreements: List[np.ndarray] = []
    adapted_disagreements: List[np.ndarray] = []
    disagreement_changes: List[np.ndarray] = []
    proposed_residuals: List[np.ndarray] = []
    command_saturation_count = 0
    action_value_count = 0
    base = controller.base_agent
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        observation_batch = torch.as_tensor(
            selected[start:stop], dtype=torch.float32, device=device
        )
        normalized = base.normalize_observations(observation_batch)
        desired = base.actor(normalized)
        baseline, _ = baseline_commands_from_desired(desired, audit_config)
        command, proposed, _ = controller.adapter.compose_command(
            normalized, baseline
        )
        noise = channel.sample(
            command.shape[0], dtype=command.dtype, device=command.device
        )
        baseline_actions = channel.apply(baseline, noise)
        adapted_actions = channel.apply(command, noise)
        baseline_q1, baseline_q2 = marginalized_twin_from_physical_actions(
            base.critic, normalized, baseline_actions
        )
        adapted_q1, adapted_q2 = marginalized_twin_from_physical_actions(
            base.critic, normalized, adapted_actions
        )
        baseline_min = torch.minimum(baseline_q1, baseline_q2)
        adapted_min = torch.minimum(adapted_q1, adapted_q2)
        baseline_disagreement = (baseline_q1 - baseline_q2).abs()
        adapted_disagreement = (adapted_q1 - adapted_q2).abs()
        q1_gains.append((adapted_q1 - baseline_q1).cpu().numpy())
        q2_gains.append((adapted_q2 - baseline_q2).cpu().numpy())
        min_twin_gains.append((adapted_min - baseline_min).cpu().numpy())
        baseline_disagreements.append(baseline_disagreement.cpu().numpy())
        adapted_disagreements.append(adapted_disagreement.cpu().numpy())
        disagreement_changes.append(
            (adapted_disagreement - baseline_disagreement).cpu().numpy()
        )
        proposed_residuals.append(proposed.abs().cpu().numpy().reshape(-1))
        preclip = baseline + proposed
        command_saturation_count += int(
            (
                (preclip < -audit_config.max_action)
                | (preclip > audit_config.max_action)
            ).sum()
        )
        action_value_count += int(preclip.numel())

    q1_gain = np.concatenate(q1_gains).astype(np.float64)
    q2_gain = np.concatenate(q2_gains).astype(np.float64)
    min_twin_gain = np.concatenate(min_twin_gains).astype(np.float64)
    baseline_disagreement = np.concatenate(baseline_disagreements).astype(np.float64)
    adapted_disagreement = np.concatenate(adapted_disagreements).astype(np.float64)
    disagreement_change = np.concatenate(disagreement_changes).astype(np.float64)
    residual_abs = np.concatenate(proposed_residuals).astype(np.float64)
    result: Dict[str, object] = {
        "status": "complete",
        **provenance,
        "baseline_transform": training_config.baseline_transform,
        "dataset": {
            "path": str(dataset_path),
            "sha256": actual_dataset_sha,
            "total_observations": int(observations.shape[0]),
            "selected_observations": int(count),
            "selection": "exact_recorded_held_out_partition",
            "selected_indices_sha256": recomputed_split[
                "audit_indices_sha256"
            ],
            "train_indices_sha256": recomputed_split["train_indices_sha256"],
            "train_audit_disjoint": bool(
                np.intersect1d(train_indices, audit_indices).size == 0
            ),
            "recorded_split": recorded_split,
            "recomputed_split": recomputed_split,
        },
        "audit_channel": {
            **channel.metadata(),
            "model_or_calibration_beta": float(
                training_config.execution_noise_beta
            ),
            "audit_action_noise_beta": float(audit_config.execution_noise_beta),
            "beta_mismatch": bool(
                audit_config.execution_noise_beta
                != training_config.execution_noise_beta
            ),
            "noise_seed_distinct_from_training_channel": True,
            "training_noise_seed": int(training_config.execution_noise_seed),
        },
        "metrics": {
            "adapted_minus_baseline_q1": _summary(q1_gain),
            "adapted_minus_baseline_q2": _summary(q2_gain),
            "adapted_minus_baseline_min_twin": _summary(min_twin_gain),
            "baseline_absolute_twin_disagreement": _summary(
                baseline_disagreement
            ),
            "adapted_absolute_twin_disagreement": _summary(
                adapted_disagreement
            ),
            "adapted_minus_baseline_twin_disagreement": _summary(
                disagreement_change
            ),
            "proposed_residual_abs": _summary(residual_abs),
            "command_saturation_fraction": command_saturation_count
            / max(action_value_count, 1),
        },
        "cost": {
            "physical_action_rows": int(
                2 * count * audit_config.execution_noise_samples
            ),
            "individual_critic_network_rows": int(
                4 * count * audit_config.execution_noise_samples
            ),
        },
        "interpretation_guardrail": (
            "states are held out from adapter SGD and Q-scale calibration, not "
            "from the already-trained base critic; frozen-Q gains are diagnostics "
            "and closed-loop returns determine whether the adapter exploits critic error"
        ),
        "implementation": {
            "audit_sha256": sha256_file(Path(__file__).resolve()),
            "evaluate_sha256": sha256_file(
                Path(__file__).resolve().with_name(
                    "evaluate_inverse_residual_adapter.py"
                )
            ),
            "core_sha256": sha256_file(
                Path(__file__).resolve().with_name("inverse_residual_core.py")
            ),
            "train_sha256": sha256_file(
                Path(__file__).resolve().with_name(
                    "train_inverse_residual_adapter.py"
                )
            ),
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    save_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-noise-samples", type=int, default=64)
    parser.add_argument("--audit-noise-seed", type=int, default=987653)
    parser.add_argument(
        "--audit-action-noise-beta",
        type=float,
        default=None,
        help=(
            "Physical actuator-noise beta used by this audit; defaults to the "
            "checkpoint model/calibration beta. Set explicitly for frozen audits."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
