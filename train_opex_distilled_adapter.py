"""Distill calibrated channel-aware OPEX commands into a residual adapter.

This is a development-only competitor/extension for the inverse-residual study.
For every sampled source-training state, a frozen physical Q1 critic produces a
projected channel-aware OPEX teacher command from the exact calibrated channel-
mean inverse anchor.  A small residual MLP is trained only by normalized command
MSE to amortize that per-state optimizer:

    u0(s) = inverse_mean_channel(pi_base(s))
    u_teacher(s) = CA-OPEX(u0(s); beta, K, T, eta, delta_max)
    u_student(s) = clip(u0(s) + delta_max*tanh(f_psi(s)))
    L(psi) = mean(((u_student - stopgrad(u_teacher))/delta_max)^2).

The actor and critic are frozen.  Training uses only the recorded source-train
partition; the held-out audit partition is not sampled.  The default teacher is
the development setting K=8, T=2, eta=0.1, delta_max=0.25.  Its isolated channel
RNG, the minibatch-index RNG, optimizer, and process RNGs are checkpointed for
exact continuation.  Checkpoints intentionally use the existing
``inverse_residual_adapter_v2`` inference format, so the independent residual
evaluator can load the distilled adapter without a special deployment path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Dict, Mapping, Tuple

import numpy as np
import torch

from evaluate_channel_opex import (
    ChannelOPEXConfig,
    canonical_device,
    make_episode_gradient_generator,
    sample_antithetic_channel_noise,
)
from inverse_residual_core import (
    InverseResidualAdapter,
    InverseResidualConfig,
    module_state_sha256,
)
from train_inverse_residual_adapter import (
    ADAPTER_CHECKPOINT_FORMAT,
    _atomic_torch_save,
    _canonical_json_sha256,
    _capture_global_rng_state,
    _load_json_object,
    _restore_global_rng_state,
    _runtime_manifest,
    _validate_progress_for_resume,
    build_observation_split,
    load_frozen_physical_agent,
    precompute_baseline_commands,
    prepare_fresh_output_dir,
    prepare_resume_output_dir,
    resolve_channel_calibration,
)
from train_iql import save_json, seed_everything, sha256_file


METHOD_NAME = "calibrated_channel_opex_teacher_distillation"
RESUME_SIGNATURE_SCHEMA = "opex_distilled_adapter_resume_signature_v1"


def _implementation_manifest() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "train_opex_distilled_adapter_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "evaluate_channel_opex_sha256": sha256_file(
            root / "evaluate_channel_opex.py"
        ),
        "inverse_residual_core_sha256": sha256_file(
            root / "inverse_residual_core.py"
        ),
        "evaluation_controls_sha256": sha256_file(
            root / "evaluation_controls.py"
        ),
        "train_inverse_residual_adapter_sha256": sha256_file(
            root / "train_inverse_residual_adapter.py"
        ),
        "td3bc_core_sha256": sha256_file(root / "td3bc_core.py"),
        "train_iql_sha256": sha256_file(root / "train_iql.py"),
    }


def _validate_bounds(
    baseline_commands: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if baseline_commands.ndim != 2 or baseline_commands.shape[1] != action_dim:
        raise ValueError("baseline_commands must have shape [batch, action_dim]")
    low = torch.as_tensor(
        action_low,
        dtype=baseline_commands.dtype,
        device=baseline_commands.device,
    ).reshape(1, action_dim)
    high = torch.as_tensor(
        action_high,
        dtype=baseline_commands.dtype,
        device=baseline_commands.device,
    ).reshape(1, action_dim)
    if bool(torch.any(low > high)):
        raise ValueError("action-space lower bounds exceed upper bounds")
    if bool(torch.any(baseline_commands < low)) or bool(
        torch.any(baseline_commands > high)
    ):
        raise ValueError("pre-gradient baseline command lies outside action bounds")
    return low, high


def batched_channel_opex_teacher(
    critic: object,
    normalized_observations: torch.Tensor,
    baseline_commands: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    config: ChannelOPEXConfig,
    gradient_generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
    """Vectorized fixed-anchor CA-OPEX teacher with the evaluator's semantics.

    ``batch_size == 1`` consumes the same generator draws and computes the same
    update as :func:`evaluate_channel_opex.channel_opex_step`.  For larger
    batches, each row receives independent antithetic samples at every gradient
    step, and summing per-row expected values yields independent action
    gradients because the critic has no cross-row operations in evaluation mode.
    """

    if normalized_observations.ndim != 2:
        raise ValueError("normalized_observations must have rank two")
    batch_size = int(normalized_observations.shape[0])
    if batch_size <= 0 or baseline_commands.shape[0] != batch_size:
        raise ValueError("observation and baseline-command batches must align")
    if normalized_observations.device != baseline_commands.device:
        raise ValueError("observation and baseline-command devices differ")
    resolved = canonical_device(baseline_commands.device)
    if canonical_device(gradient_generator.device) != resolved:
        raise ValueError("teacher generator and command tensor devices differ")
    low, high = _validate_bounds(
        baseline_commands, action_low, action_high, config.action_dim
    )

    anchor = baseline_commands.detach().clone()
    trust_low = torch.maximum(low, anchor - config.delta_max)
    trust_high = torch.minimum(high, anchor + config.delta_max)
    command = anchor
    expected_q_by_step = []
    gradient_l2_by_step = []
    gradient_abs_max_by_step = []
    raw_step_abs_max_by_step = []
    trust_region_clipped_value_count = 0

    for _ in range(config.gradient_steps):
        current = command.detach().clone().requires_grad_(True)
        noise = sample_antithetic_channel_noise(
            batch_size,
            config,
            generator=gradient_generator,
            dtype=current.dtype,
            device=current.device,
        )
        if config.channel_beta == 0.0:
            physical_actions = current[:, None, :].expand(
                -1, config.gradient_noise_samples, -1
            )
        else:
            physical_actions = torch.maximum(
                torch.minimum(current[:, None, :] + noise, high[:, None, :]),
                low[:, None, :],
            )
        observation_rows = normalized_observations[:, None, :].expand(
            -1, config.gradient_noise_samples, -1
        )
        q1_rows = critic.q1_only(
            observation_rows.reshape(
                batch_size * config.gradient_noise_samples, -1
            ),
            physical_actions.reshape(
                batch_size * config.gradient_noise_samples, config.action_dim
            ),
        )
        expected_shape = (batch_size * config.gradient_noise_samples,)
        if tuple(q1_rows.shape) != expected_shape:
            raise ValueError("critic.q1_only must return one scalar per Q row")
        expected_q_per_state = q1_rows.reshape(
            batch_size, config.gradient_noise_samples
        ).mean(dim=1)
        gradient = torch.autograd.grad(
            expected_q_per_state.sum(),
            current,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        if not bool(torch.isfinite(q1_rows).all()) or not bool(
            torch.isfinite(gradient).all()
        ):
            raise FloatingPointError("non-finite teacher Q value or command gradient")
        raw_step = config.step_size * gradient
        unprojected = current + raw_step
        trust_region_clipped_value_count += int(
            ((unprojected.detach() < trust_low) | (unprojected.detach() > trust_high))
            .sum()
            .cpu()
        )
        command = torch.maximum(torch.minimum(unprojected, trust_high), trust_low)
        expected_q_by_step.append(expected_q_per_state.detach())
        gradient_l2_by_step.append(gradient.detach().norm(p=2, dim=1))
        gradient_abs_max_by_step.append(gradient.detach().abs().amax(dim=1))
        raw_step_abs_max_by_step.append(raw_step.detach().abs().amax(dim=1))

    proposed_residual = command.detach() - anchor
    return command.detach(), {
        "q1_rows": int(
            batch_size
            * config.gradient_noise_samples
            * config.gradient_steps
        ),
        "q1_gradient_calls": int(config.gradient_steps),
        "action_gradient_vectors": int(batch_size * config.gradient_steps),
        "noise_sample_calls": int(config.gradient_steps),
        "half_noise_vectors": int(
            batch_size
            * (config.gradient_noise_samples // 2)
            * config.gradient_steps
            if config.channel_beta > 0.0
            else 0
        ),
        "expected_q1_mean": torch.stack(expected_q_by_step).mean(),
        "gradient_l2_mean": torch.stack(gradient_l2_by_step).mean(),
        "gradient_abs_max": torch.stack(gradient_abs_max_by_step).amax(),
        "raw_step_abs_max": torch.stack(raw_step_abs_max_by_step).amax(),
        "teacher_residual_abs_mean": proposed_residual.abs().mean(),
        "teacher_residual_abs_max": proposed_residual.abs().amax(),
        "teacher_command_at_bound_fraction": (
            (command.detach() <= low) | (command.detach() >= high)
        ).float().mean(),
        "trust_region_clipped_value_count": int(
            trust_region_clipped_value_count
        ),
    }


def distillation_objective(
    adapter: InverseResidualAdapter,
    normalized_observations: torch.Tensor,
    baseline_commands: torch.Tensor,
    teacher_commands: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fit the bounded deployed command; teacher tensors are always detached."""

    if adapter.config.delta_max <= 0.0:
        raise ValueError("distillation requires a positive delta_max")
    if teacher_commands.shape != baseline_commands.shape:
        raise ValueError("teacher and baseline command shapes differ")
    student, proposed, applied = adapter.compose_command(
        normalized_observations, baseline_commands
    )
    target = teacher_commands.detach()
    error = (student - target) / adapter.config.delta_max
    loss = error.square().mean()
    target_residual = target - baseline_commands
    return loss, {
        "normalized_command_mse": loss,
        "command_rmse": (student - target).square().mean().sqrt(),
        "residual_rmse": (applied - target_residual).square().mean().sqrt(),
        "student_residual_abs_mean": applied.abs().mean(),
        "student_proposed_residual_abs_max": proposed.abs().amax(),
        "teacher_residual_abs_mean": target_residual.abs().mean(),
        "teacher_residual_abs_max": target_residual.abs().amax(),
    }


def _float_metrics(values: Mapping[str, torch.Tensor | int]) -> Dict[str, float | int]:
    result: Dict[str, float | int] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu())
        else:
            result[key] = int(value)
    return result


def _teacher_state(
    generator: torch.Generator,
    *,
    updates_completed: int,
    batch_size: int,
    config: ChannelOPEXConfig,
) -> Dict[str, object]:
    return {
        "generator_state": generator.get_state().cpu(),
        "updates_completed": int(updates_completed),
        "noise_sample_calls": int(updates_completed * config.gradient_steps),
        "half_noise_vectors": int(
            updates_completed
            * batch_size
            * (config.gradient_noise_samples // 2)
            * config.gradient_steps
            if config.channel_beta > 0.0
            else 0
        ),
        "q1_input_rows": int(
            updates_completed
            * batch_size
            * config.gradient_noise_samples
            * config.gradient_steps
        ),
        "q1_gradient_calls": int(
            updates_completed * config.gradient_steps
        ),
        "action_gradient_vectors": int(
            updates_completed * batch_size * config.gradient_steps
        ),
    }


def _checkpoint(
    *,
    step: int,
    adapter: InverseResidualAdapter,
    optimizer: torch.optim.Optimizer,
    teacher_generator: torch.Generator,
    teacher_config: ChannelOPEXConfig,
    index_generator: torch.Generator,
    batch_size: int,
    base_checkpoint: Path,
    base_checkpoint_sha256: str,
    base_step: int,
    base_parameter_sha256: str,
    config_payload: Dict[str, object],
    resume_signature: Dict[str, object],
    elapsed_seconds_completed: float,
    optimizer_seconds_completed: float,
    target_total_updates: int,
    resume_count: int,
) -> Dict[str, object]:
    # The common fields intentionally satisfy evaluate_inverse_residual_adapter.
    return {
        "format": ADAPTER_CHECKPOINT_FORMAT,
        "training_method": METHOD_NAME,
        "step": int(step),
        "adapter_config": asdict(adapter.config),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "channel": _teacher_state(
            teacher_generator,
            updates_completed=step,
            batch_size=batch_size,
            config=teacher_config,
        ),
        "teacher": {
            "config": asdict(teacher_config),
            **_teacher_state(
                teacher_generator,
                updates_completed=step,
                batch_size=batch_size,
                config=teacher_config,
            ),
        },
        "index_generator_state": index_generator.get_state().cpu(),
        "global_rng_state": _capture_global_rng_state(),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_checkpoint_step": int(base_step),
        "base_parameter_sha256": base_parameter_sha256,
        "config": config_payload,
        "resume_signature": resume_signature,
        "resume_signature_sha256": _canonical_json_sha256(resume_signature),
        "elapsed_seconds_completed": float(elapsed_seconds_completed),
        "optimizer_seconds_completed": float(optimizer_seconds_completed),
        "target_total_updates": int(target_total_updates),
        "resume_count": int(resume_count),
        "exact_training_resume_implemented": True,
    }


def _append_progress(path: Path, event: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch_size must be positive")
    if args.log_period <= 0 or args.checkpoint_period <= 0:
        raise ValueError("log_period and checkpoint_period must be positive")
    if args.delta_max <= 0.0 or not np.isfinite(args.delta_max):
        raise ValueError("delta_max must be finite and positive")
    output_dir = (
        prepare_resume_output_dir(Path(args.output_dir))
        if args.resume
        else prepare_fresh_output_dir(Path(args.output_dir))
    )
    torch.set_num_threads(args.torch_threads)
    device = canonical_device(args.device)
    base_checkpoint = Path(args.base_checkpoint).resolve()
    base_checkpoint_sha = sha256_file(base_checkpoint)
    base_agent, _, base_step = load_frozen_physical_agent(base_checkpoint, device)
    base_modules = {
        "actor": base_agent.actor,
        "critic": base_agent.critic,
        "actor_target": base_agent.actor_target,
        "critic_target": base_agent.critic_target,
    }
    base_hash_before = module_state_sha256(base_modules)

    channel_beta, channel_calibration = resolve_channel_calibration(
        Path(args.channel_calibration) if args.channel_calibration else None,
        args.execution_noise_beta,
    )
    if channel_calibration.get("action_dim") is not None and int(
        channel_calibration["action_dim"]
    ) != base_agent.config.action_dim:
        raise ValueError("channel calibration action_dim does not match base checkpoint")
    for field, expected in (
        ("action_low", -base_agent.config.max_action),
        ("action_high", base_agent.config.max_action),
    ):
        observed = channel_calibration.get(field)
        if observed is not None and float(observed) != float(expected):
            raise ValueError(f"channel calibration {field} does not match base bounds")

    seed_everything(args.train_seed)
    adapter_config = InverseResidualConfig(
        observation_dim=base_agent.config.observation_dim,
        action_dim=base_agent.config.action_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        max_action=base_agent.config.max_action,
        execution_noise_beta=channel_beta,
        execution_noise_samples=args.teacher_k,
        execution_noise_seed=args.teacher_noise_seed,
        delta_max=args.delta_max,
        alpha=0.0,
        residual_penalty=0.0,
        baseline_transform="inverse",
        value_estimator="sampled_expected_q1",
    )
    teacher_config = ChannelOPEXConfig(
        action_dim=base_agent.config.action_dim,
        step_size=args.teacher_step_size,
        channel_beta=channel_beta,
        gradient_noise_samples=args.teacher_k,
        gradient_steps=args.teacher_gradient_steps,
        delta_max=args.delta_max,
        baseline_transform="inverse",
    )
    adapter = InverseResidualAdapter(adapter_config).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    base_parameter_ids = {
        id(parameter)
        for module in base_modules.values()
        for parameter in module.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if base_parameter_ids & optimizer_parameter_ids:
        raise RuntimeError("optimizer unexpectedly contains frozen base parameters")

    dataset_path = Path(args.dataset).resolve()
    observations, train_indices, audit_indices, split = build_observation_split(
        dataset_path,
        args.max_observations,
        args.audit_fraction,
        args.split_seed,
    )
    if observations.shape[1] != adapter_config.observation_dim:
        raise ValueError("dataset observation dimension differs from base checkpoint")
    dataset_sha = sha256_file(dataset_path)
    preprocessing_started = time.perf_counter()
    normalized_cpu, inverse_commands_cpu, inverse_audit = precompute_baseline_commands(
        base_agent,
        observations[train_indices],
        adapter_config,
        args.precompute_batch_size,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    with torch.no_grad():
        check_count = min(1024, normalized_cpu.shape[0])
        initial_commands, proposed, _ = adapter.compose_command(
            normalized_cpu[:check_count].to(device),
            inverse_commands_cpu[:check_count].to(device),
        )
        initial_difference = float(
            (initial_commands - inverse_commands_cpu[:check_count].to(device))
            .abs()
            .amax()
        )
        if initial_difference != 0.0 or int(torch.count_nonzero(proposed)) != 0:
            raise RuntimeError("zero initialization failed baseline identity")

    teacher_generator = make_episode_gradient_generator(
        device, args.teacher_noise_seed
    )
    index_generator = torch.Generator(device="cpu")
    implementation = _implementation_manifest()
    runtime = _runtime_manifest(device)
    initial_adapter_hash = module_state_sha256({"adapter": adapter})
    dataset_metadata = {
        "path": str(dataset_path),
        "sha256": dataset_sha,
        "considered_observation_count": int(observations.shape[0]),
        "training_observation_count": int(train_indices.size),
        "held_out_audit_observation_count": int(audit_indices.size),
    }
    teacher_metadata = {
        "algorithm": "fixed_anchor_projected_channel_aware_opex",
        "anchor": "calibrated_exact_clipped_uniform_mean_inverse",
        "critic": "frozen_q1_on_executed_physical_actions",
        "channel_estimator": "fresh_antithetic_monte_carlo_expected_q1",
        "config": asdict(teacher_config),
        "q1_rows_per_state_per_update": int(
            teacher_config.gradient_noise_samples
            * teacher_config.gradient_steps
        ),
        "q1_gradient_calls_per_minibatch_update": int(
            teacher_config.gradient_steps
        ),
        "action_gradient_vectors_per_state_per_update": int(
            teacher_config.gradient_steps
        ),
        "teacher_rng": {
            "seed": int(args.teacher_noise_seed),
            "device": str(device),
            "isolated_from_minibatch_and_global_rngs": True,
            "fresh_noise_each_gradient_step_and_update": True,
        },
    }
    resume_signature: Dict[str, object] = {
        "schema": RESUME_SIGNATURE_SCHEMA,
        "method": METHOD_NAME,
        "adapter": asdict(adapter_config),
        "teacher": teacher_metadata,
        "optimization": {
            "optimizer": "torch.optim.Adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "train_seed": int(args.train_seed),
            "loss": "mean(((student_command-teacher_command)/delta_max)^2)",
            "precompute_batch_size": int(args.precompute_batch_size),
            "log_period": int(args.log_period),
            "checkpoint_period": int(args.checkpoint_period),
            "torch_threads": int(args.torch_threads),
        },
        "base": {
            "path": str(base_checkpoint),
            "checkpoint_sha256": base_checkpoint_sha,
            "checkpoint_step": int(base_step),
            "parameter_sha256": base_hash_before,
            "action_pairing": base_agent.config.action_pairing,
        },
        "channel_calibration": channel_calibration,
        "dataset": dataset_metadata,
        "observation_split": split,
        "baseline_precomputation": inverse_audit,
        "initial_adapter_parameter_sha256": initial_adapter_hash,
        "implementation": implementation,
        "runtime": runtime,
    }
    resume_signature_sha = _canonical_json_sha256(resume_signature)
    candidate_config: Dict[str, object] = {
        "method": METHOD_NAME,
        "development_only": True,
        "arguments": dict(vars(args)),
        "adapter": asdict(adapter_config),
        "teacher": teacher_metadata,
        "objective": {
            "formula": "mean(((student_command-stopgrad(teacher_command))/delta_max)^2)",
            "supervision": "online_teacher_command_for_each_sampled_source_train_state",
            "student_deployment_q1_rows_per_action": 0,
        },
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha,
        "base_checkpoint_step": int(base_step),
        "base_action_pairing": base_agent.config.action_pairing,
        "base_parameter_sha256_before": base_hash_before,
        "baseline_transform": "inverse",
        "channel_calibration": channel_calibration,
        "dataset": dataset_metadata,
        "observation_split": split,
        "baseline_precomputation": {
            **inverse_audit,
            "seconds": preprocessing_seconds,
            "initial_adapter_max_abs_command_difference": initial_difference,
        },
        "q1_rows_per_state_per_update": int(
            teacher_config.gradient_noise_samples
            * teacher_config.gradient_steps
        ),
        "checkpoint_policy": (
            "atomic latest.pt before each logged progress event, requested "
            "checkpoint period, and final update"
        ),
        "implementation": implementation,
        "runtime": runtime,
        "resume_signature": resume_signature,
        "resume_signature_sha256": resume_signature_sha,
        "selection_rule": "all requested updates retained; no run exclusion",
        "exact_training_resume_implemented": True,
    }

    progress_path = output_dir / "progress.jsonl"
    start_step = 0
    elapsed_offset = 0.0
    optimizer_offset = 0.0
    resume_count = 0
    if args.resume:
        checkpoint_path = output_dir / "latest.pt"
        checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != (
            ADAPTER_CHECKPOINT_FORMAT
        ):
            raise ValueError("resume requires an inverse_residual_adapter_v2 checkpoint")
        if checkpoint.get("training_method") != METHOD_NAME:
            raise ValueError("resume checkpoint was not produced by OPEX distillation")
        stored_signature = checkpoint.get("resume_signature")
        if not isinstance(stored_signature, dict) or checkpoint.get(
            "resume_signature_sha256"
        ) != _canonical_json_sha256(stored_signature):
            raise ValueError("resume checkpoint signature is absent or invalid")
        if stored_signature != resume_signature:
            raise ValueError(
                "resume signature mismatch: method, data, base, code, or runtime changed"
            )
        disk_config = _load_json_object(output_dir / "config.json")
        if checkpoint.get("config") != disk_config:
            raise ValueError("on-disk config differs from checkpoint config")
        start_step = int(checkpoint.get("step", -1))
        if start_step <= 0 or args.updates <= start_step:
            raise ValueError("resume total --updates must exceed the positive saved step")
        _validate_progress_for_resume(progress_path, start_step)
        if checkpoint.get("adapter_config") != asdict(adapter_config):
            raise ValueError("resume adapter config mismatch")
        if checkpoint.get("base_checkpoint_sha256") != base_checkpoint_sha or (
            checkpoint.get("base_parameter_sha256") != base_hash_before
        ):
            raise ValueError("resume base provenance mismatch")
        teacher_state = checkpoint.get("teacher")
        if not isinstance(teacher_state, dict) or teacher_state.get("config") != (
            asdict(teacher_config)
        ):
            raise ValueError("resume teacher state/config mismatch")
        expected_teacher = _teacher_state(
            teacher_generator,
            updates_completed=start_step,
            batch_size=args.batch_size,
            config=teacher_config,
        )
        for field in (
            "updates_completed",
            "noise_sample_calls",
            "half_noise_vectors",
            "q1_input_rows",
            "q1_gradient_calls",
            "action_gradient_vectors",
        ):
            if int(teacher_state.get(field, -1)) != int(expected_teacher[field]):
                raise ValueError(f"resume teacher accounting mismatch for {field}")
        adapter_state = checkpoint.get("adapter")
        optimizer_state = checkpoint.get("optimizer")
        index_state = checkpoint.get("index_generator_state")
        generator_state = teacher_state.get("generator_state")
        if not isinstance(adapter_state, dict) or not isinstance(
            optimizer_state, dict
        ):
            raise ValueError("resume checkpoint lacks adapter or optimizer")
        if not torch.is_tensor(index_state) or not torch.is_tensor(generator_state):
            raise ValueError("resume checkpoint lacks an RNG state")
        if optimizer_state.get("param_groups") != optimizer.state_dict()[
            "param_groups"
        ]:
            raise ValueError("resume optimizer hyperparameters mismatch")
        elapsed_offset = float(checkpoint.get("elapsed_seconds_completed", -1.0))
        optimizer_offset = float(
            checkpoint.get("optimizer_seconds_completed", -1.0)
        )
        if min(elapsed_offset, optimizer_offset) < 0.0 or not np.isfinite(
            elapsed_offset + optimizer_offset
        ):
            raise ValueError("resume time accounting is invalid")
        resume_count = int(checkpoint.get("resume_count", -1)) + 1
        if resume_count <= 0:
            raise ValueError("resume counter is invalid")
        adapter.load_state_dict(adapter_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        index_generator.set_state(index_state.detach().cpu())
        teacher_generator.set_state(generator_state.detach().cpu())
        _restore_global_rng_state(checkpoint.get("global_rng_state"))
        config_payload = disk_config
        _append_progress(
            progress_path,
            {
                "event": "resume",
                "from_step": start_step,
                "target_total_updates": int(args.updates),
                "resume_count": resume_count,
                "checkpoint_sha256": checkpoint_sha,
                "elapsed_seconds_offset": elapsed_offset,
                "optimizer_seconds_offset": optimizer_offset,
            },
        )
    else:
        index_generator.manual_seed(args.train_seed + 1729)
        config_payload = candidate_config
        save_json(output_dir / "config.json", config_payload)
        progress_path.write_text("", encoding="utf-8")

    low = torch.full(
        (adapter_config.action_dim,),
        -adapter_config.max_action,
        dtype=torch.float32,
        device=device,
    )
    high = -low
    last_metrics: Dict[str, float | int] = {}
    optimizer_seconds_this_invocation = 0.0
    for step in range(start_step + 1, args.updates + 1):
        update_started = time.perf_counter()
        indices = torch.randint(
            0,
            normalized_cpu.shape[0],
            (args.batch_size,),
            generator=index_generator,
        )
        normalized = normalized_cpu[indices].to(device)
        anchors = inverse_commands_cpu[indices].to(device)
        teacher_commands, teacher_diagnostics = batched_channel_opex_teacher(
            base_agent.critic,
            normalized,
            anchors,
            low,
            high,
            teacher_config,
            teacher_generator,
        )
        loss, student_diagnostics = distillation_objective(
            adapter, normalized, anchors, teacher_commands
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_seconds_this_invocation += time.perf_counter() - update_started
        optimizer_seconds = optimizer_offset + optimizer_seconds_this_invocation
        elapsed_seconds = elapsed_offset + time.perf_counter() - started
        last_metrics = {
            **_float_metrics(student_diagnostics),
            **{f"teacher_{key}": value for key, value in _float_metrics(
                teacher_diagnostics
            ).items()},
        }
        should_log = step == 1 or step % args.log_period == 0 or (
            step == args.updates
        )
        should_checkpoint = should_log or step % args.checkpoint_period == 0
        if should_checkpoint:
            _atomic_torch_save(
                output_dir / "latest.pt",
                _checkpoint(
                    step=step,
                    adapter=adapter,
                    optimizer=optimizer,
                    teacher_generator=teacher_generator,
                    teacher_config=teacher_config,
                    index_generator=index_generator,
                    batch_size=args.batch_size,
                    base_checkpoint=base_checkpoint,
                    base_checkpoint_sha256=base_checkpoint_sha,
                    base_step=base_step,
                    base_parameter_sha256=base_hash_before,
                    config_payload=config_payload,
                    resume_signature=resume_signature,
                    elapsed_seconds_completed=elapsed_seconds,
                    optimizer_seconds_completed=optimizer_seconds,
                    target_total_updates=args.updates,
                    resume_count=resume_count,
                ),
            )
        if should_log:
            _append_progress(
                progress_path,
                {
                    "event": "train",
                    "step": step,
                    "elapsed_seconds": elapsed_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "updates_per_optimizer_second": step
                    / max(optimizer_seconds, 1e-12),
                    "teacher_q1_input_rows_cumulative": int(
                        step
                        * args.batch_size
                        * teacher_config.gradient_noise_samples
                        * teacher_config.gradient_steps
                    ),
                    **last_metrics,
                },
            )

    base_hash_after = module_state_sha256(base_modules)
    if base_hash_after != base_hash_before:
        raise RuntimeError("frozen base actor or critic changed during distillation")
    invocation_wall_time = time.perf_counter() - started
    wall_time = elapsed_offset + invocation_wall_time
    optimizer_seconds = optimizer_offset + optimizer_seconds_this_invocation
    teacher_accounting = _teacher_state(
        teacher_generator,
        updates_completed=args.updates,
        batch_size=args.batch_size,
        config=teacher_config,
    )
    teacher_accounting.pop("generator_state")
    summary: Dict[str, object] = {
        "status": "complete",
        "method": METHOD_NAME,
        "development_only": True,
        "updates": int(args.updates),
        "target_total_updates": int(args.updates),
        "started_from_step": int(start_step),
        "resumed": bool(args.resume),
        "resume_count": int(resume_count),
        "train_seed": int(args.train_seed),
        "wall_time_seconds": wall_time,
        "invocation_wall_time_seconds": invocation_wall_time,
        "preprocessing_seconds_this_invocation": preprocessing_seconds,
        "optimizer_seconds": optimizer_seconds,
        "updates_per_optimizer_second": args.updates
        / max(optimizer_seconds, 1e-12),
        "teacher": teacher_metadata,
        "teacher_accounting": teacher_accounting,
        "last_train_metrics": last_metrics,
        "channel_calibration": channel_calibration,
        "observation_split": split,
        "baseline_precomputation": inverse_audit,
        "base_parameter_sha256_before": base_hash_before,
        "base_parameter_sha256_after": base_hash_after,
        "base_parameters_unchanged": True,
        "deployment_q1_rows_per_action": 0,
        "checkpoint": str(output_dir / "latest.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "latest.pt"),
        "resume_signature_sha256": resume_signature_sha,
        "exact_training_resume_implemented": True,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue exactly; --updates is the new total target",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--delta-max", type=float, default=0.25)
    parser.add_argument("--channel-calibration")
    parser.add_argument("--execution-noise-beta", type=float)
    parser.add_argument("--teacher-step-size", type=float, default=0.1)
    parser.add_argument("--teacher-k", type=int, default=8)
    parser.add_argument("--teacher-gradient-steps", type=int, default=2)
    parser.add_argument("--teacher-noise-seed", type=int, default=271828)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--audit-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=424242)
    parser.add_argument("--precompute-batch-size", type=int, default=8192)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--checkpoint-period", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
