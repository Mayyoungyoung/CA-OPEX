"""Train the fixed CC-FQE -> residual policy-improvement fallback.

The default research protocol is intentionally a single configuration:

* 20,000 fitted-Q updates on logged executed actions;
* no HUBL Monte-Carlo mixing and no TD3 target-policy smoothing;
* 5,000 adapter-only updates after freezing the fitted critics;
* K=4 resampled antithetic channel samples during training; and
* an independent K=64 audit saved from raw per-state records.

``--continuation-mode source`` is the equal-compute control.  It changes only
the execution beta in the fitted-Q continuation; the inverse baseline and
residual improvement still target ``--target-beta``.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cc_fqe_rpi_core import (
    CCFQERPIAgent,
    CCFQERPIConfig,
    CONTINUATION_MODES,
    audit_policy_k64,
)
from td3bc_core import TD3BCConfig
from train_iql import prepare_dataset
from train_td3bc import load_action_channels


CHECKPOINT_FORMAT = "cc_fqe_rpi_training_v1"


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(np.asarray(arrays[name])).encode("ascii"))
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_variant(run_config: object) -> Optional[str]:
    if not isinstance(run_config, dict):
        return None
    arguments = run_config.get("arguments")
    if isinstance(arguments, dict) and isinstance(arguments.get("variant"), str):
        return str(arguments["variant"])
    variant = run_config.get("variant")
    return str(variant) if isinstance(variant, str) else None


def load_hubl_executed_checkpoint(
    path: Path, device: torch.device
) -> Tuple[Dict[str, object], TD3BCConfig, Dict[str, object]]:
    """Load only a causally grounded executed/executed HUBL initialisation."""

    # Keep serialized RNG/optimizer tensors on CPU; model state dictionaries
    # are copied to ``device`` by load_state_dict in CCFQERPIAgent.
    wrapper = torch.load(path, map_location="cpu")
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("agent"), dict):
        raise KeyError("base checkpoint must contain wrapper['agent']")
    payload = wrapper["agent"]
    required = (
        "config",
        "observation_mean",
        "observation_std",
        "actor",
        "critic",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"base checkpoint lacks fields: {missing}")
    base_config = TD3BCConfig(**payload["config"])
    if base_config.action_pairing != "executed_executed":
        raise ValueError(
            "CC-FQE/RPI requires an executed/executed base checkpoint so that "
            "the critic input is the physical action and the actor denotes a "
            "desired physical action"
        )
    if base_config.marginalize_execution_noise:
        raise ValueError("base checkpoint must precede command-channel marginalisation")
    run_config = wrapper.get("config", {})
    variant = _checkpoint_variant(run_config)
    builder_variant = (
        run_config.get("hubl_variant_used_by_shared_builder")
        if isinstance(run_config, dict)
        else None
    )
    is_hubl = bool(variant and variant.startswith("hubl_")) or bool(
        isinstance(builder_variant, str) and builder_variant.startswith("hubl_")
    )
    if not is_hubl:
        raise ValueError("base checkpoint provenance does not identify a HUBL variant")
    mean = payload["observation_mean"]
    std = payload["observation_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    initialisation: Dict[str, object] = {
        "observation_mean": np.asarray(mean, dtype=np.float32),
        "observation_std": np.asarray(std, dtype=np.float32),
        "actor": payload["actor"],
        "critic": payload["critic"],
    }
    audit = {
        "step": int(wrapper.get("step", payload.get("total_updates", -1))),
        "variant": variant,
        "hubl_variant_used_by_shared_builder": builder_variant,
        "action_pairing": base_config.action_pairing,
        "marginalize_execution_noise": bool(
            base_config.marginalize_execution_noise
        ),
    }
    return initialisation, base_config, audit


def load_channel_calibration(
    path: Optional[Path], target_beta: float
) -> Dict[str, object]:
    if path is None:
        return {
            "source": "fixed_numeric_estimate",
            "target_beta": float(target_beta),
            "calibration_file": None,
            "calibration_sha256": None,
        }
    resolved = path.resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        estimate = float(payload["estimate"]["beta_mle"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("calibration JSON lacks estimate.beta_mle") from error
    if not np.isclose(estimate, target_beta, rtol=0.0, atol=1e-7):
        raise ValueError(
            f"target beta {target_beta} differs from calibration estimate {estimate}"
        )
    return {
        "source": "calibrate_uniform_channel.py JSON",
        "target_beta": float(target_beta),
        "calibration_beta_mle": estimate,
        "calibration_file": str(resolved),
        "calibration_sha256": sha256_file(resolved),
        "pair_count": payload.get("pair_count"),
        "estimator": payload.get("estimator"),
        "uncertainty_claim": (
            "point_estimate_only; endpoint bootstrap is not treated as calibrated"
        ),
    }


@torch.no_grad()
def precompute_inverse_commands(
    agent: CCFQERPIAgent,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("precompute batch_size must be positive")
    values = torch.empty(
        (observations.shape[0], agent.config.action_dim), dtype=torch.float32
    )
    value_count = 0
    saturation_count = 0
    at_bound_count = 0
    for start in range(0, observations.shape[0], batch_size):
        stop = min(start + batch_size, observations.shape[0])
        batch = torch.as_tensor(
            observations[start:stop], dtype=torch.float32, device=agent.device
        )
        commands, audit = agent.inverse_baseline_commands(batch)
        values[start:stop] = commands.cpu()
        value_count += int(audit["command_transform_value_count"])
        saturation_count += int(audit["command_transform_saturation_count"])
        at_bound_count += int(audit["transformed_command_at_bound_count"])
    return values, {
        "observation_count": int(observations.shape[0]),
        "action_value_count": value_count,
        "inverse_saturation_count": saturation_count,
        "inverse_saturation_fraction": saturation_count / max(value_count, 1),
        "inverse_command_at_bound_count": at_bound_count,
        "inverse_command_at_bound_fraction": at_bound_count / max(value_count, 1),
        "target_beta": agent.config.target_beta,
        "semantics": (
            "generalized clipped-uniform mean inverse; exact only in attainable "
            "mean range and clamped outside"
        ),
    }


def make_cpu_dataset_tensors(
    dataset: object,
    commanded_actions: np.ndarray,
    inverse_commands: torch.Tensor,
    next_inverse_commands: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        "observations": torch.as_tensor(dataset.observations, dtype=torch.float32),
        "executed_actions": torch.as_tensor(dataset.actions, dtype=torch.float32),
        "logged_commands": torch.as_tensor(commanded_actions, dtype=torch.float32),
        "next_observations": torch.as_tensor(
            dataset.next_observations, dtype=torch.float32
        ),
        # Use the environment rewards stored in the fixed dataset.  HUBL MC
        # suffixes and synthetic reward-noise arrays are intentionally absent.
        "rewards": torch.as_tensor(dataset.clean_rewards, dtype=torch.float32),
        "terminals": torch.as_tensor(dataset.terminals, dtype=torch.float32),
        "inverse_commands": inverse_commands,
        "next_inverse_commands": next_inverse_commands,
    }


def sample_batch(
    tensors: Mapping[str, torch.Tensor],
    keys: Sequence[str],
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    count = tensors["observations"].shape[0]
    indices = torch.randint(0, count, (batch_size,), generator=generator)
    return {key: tensors[key][indices].to(device) for key in keys}


def fixed_audit_indices(count: int, requested: int, seed: int) -> np.ndarray:
    if count <= 0 or requested <= 0:
        raise ValueError("audit counts must be positive")
    rng = np.random.default_rng(seed)
    size = min(count, requested)
    return rng.permutation(count)[:size].astype(np.int64, copy=False)


def capture_global_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_global_rng_state(payload: Mapping[str, object]) -> None:
    required = ("python", "numpy", "torch_cpu", "torch_cuda")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"global RNG checkpoint lacks fields: {missing}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    cuda_states = payload["torch_cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG states but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


def training_checkpoint(
    agent: CCFQERPIAgent,
    *,
    fqe_batch_generator: torch.Generator,
    adapter_batch_generator: torch.Generator,
    provenance_fingerprint: str,
    provenance: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        # Make this an immutable point-in-time snapshot even before torch.save;
        # state_dict/optimizer dictionaries otherwise retain tensor aliases.
        "agent": copy.deepcopy(agent.checkpoint()),
        "fqe_batch_generator_state": fqe_batch_generator.get_state().clone(),
        "adapter_batch_generator_state": adapter_batch_generator.get_state().clone(),
        "global_rng_state": capture_global_rng_state(),
        "provenance_fingerprint": provenance_fingerprint,
        "provenance": dict(provenance),
        "resume_scope": (
            "exact RNG/model/optimizer continuation on the same device/backend; "
            "cross-backend floating-point identity is not claimed"
        ),
    }


def restore_training_checkpoint(
    payload: Mapping[str, object],
    agent: CCFQERPIAgent,
    *,
    fqe_batch_generator: torch.Generator,
    adapter_batch_generator: torch.Generator,
    provenance_fingerprint: str,
) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported CC-FQE/RPI training checkpoint")
    if payload.get("provenance_fingerprint") != provenance_fingerprint:
        raise ValueError("resume provenance fingerprint mismatch")
    agent.restore(payload["agent"])
    fqe_state = payload.get("fqe_batch_generator_state")
    adapter_state = payload.get("adapter_batch_generator_state")
    if not isinstance(fqe_state, torch.Tensor) or not isinstance(
        adapter_state, torch.Tensor
    ):
        raise ValueError("checkpoint lacks both minibatch RNG states")
    fqe_batch_generator.set_state(fqe_state)
    adapter_batch_generator.set_state(adapter_state)
    global_state = payload.get("global_rng_state")
    if not isinstance(global_state, dict):
        raise ValueError("checkpoint lacks global RNG state")
    restore_global_rng_state(global_state)


def audit_summary(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for key, value in arrays.items():
        if key == "indices":
            continue
        values = np.asarray(value, dtype=np.float64)
        metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        **dict(metadata),
        "raw_arrays_sha256": arrays_sha256(arrays),
        "indices_sha256": array_sha256(np.asarray(arrays["indices"])),
        "metrics": metrics,
    }


def _append_progress(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.batch_size <= 0 or args.precompute_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.audit_observations <= 0 or args.audit_batch_size <= 0:
        raise ValueError("audit sizes must be positive")
    if args.checkpoint_period <= 0 or args.log_period <= 0:
        raise ValueError("checkpoint/log periods must be positive")
    output_dir = Path(args.output_dir).resolve()
    resume_path = Path(args.resume).resolve() if args.resume else None
    if output_dir.exists() and any(output_dir.iterdir()) and resume_path is None:
        if not args.dry_run:
            raise FileExistsError(
                f"refusing to overwrite nonempty output directory: {output_dir}"
            )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    dataset_path = Path(args.dataset).resolve()
    base_checkpoint_path = Path(args.base_checkpoint).resolve()
    dataset_hash = sha256_file(dataset_path)
    base_checkpoint_hash = sha256_file(base_checkpoint_path)
    initialisation, base_config, base_audit = load_hubl_executed_checkpoint(
        base_checkpoint_path, device
    )
    base_run_payload = torch.load(base_checkpoint_path, map_location="cpu")
    base_run_config = base_run_payload.get("config", {})
    recorded_dataset_hash = None
    if isinstance(base_run_config, dict) and isinstance(
        base_run_config.get("dataset"), dict
    ):
        recorded_dataset_hash = base_run_config["dataset"].get("sha256")
    if recorded_dataset_hash is not None and recorded_dataset_hash != dataset_hash:
        raise ValueError("dataset SHA256 differs from the base HUBL checkpoint")

    config = CCFQERPIConfig(
        observation_dim=base_config.observation_dim,
        action_dim=base_config.action_dim,
        base_hidden_dim=base_config.hidden_dim,
        base_depth=base_config.depth,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_depth=args.adapter_depth,
        max_action=base_config.max_action,
        discount=base_config.discount,
        tau=args.tau,
        source_beta=args.source_beta,
        target_beta=args.target_beta,
        continuation_mode=args.continuation_mode,
        train_channel_samples=args.train_channel_samples,
        audit_channel_samples=args.audit_channel_samples,
        continuation_noise_seed=args.continuation_noise_seed,
        actor_noise_seed=args.actor_noise_seed,
        audit_noise_seed=args.audit_noise_seed,
        fqe_updates=args.fqe_updates,
        adapter_updates=args.adapter_updates,
        critic_learning_rate=args.critic_learning_rate,
        adapter_learning_rate=args.adapter_learning_rate,
        delta_max=args.delta_max,
        value_alpha=args.value_alpha,
        residual_penalty=args.residual_penalty,
        support_penalty=args.support_penalty,
        q_scale_epsilon=args.q_scale_epsilon,
    )
    calibration_audit = load_channel_calibration(
        Path(args.channel_calibration) if args.channel_calibration else None,
        config.target_beta,
    )
    dataset = prepare_dataset(
        dataset_path,
        discount=config.discount,
        noise_seed=0,
        iid_noise_scale=0.0,
        episode_noise_scale=0.0,
        max_episodes=args.max_episodes,
    )
    commanded_actions, action_channel_audit = load_action_channels(
        dataset_path,
        dataset.actions,
        action_pairing="executed_commanded",
        max_episodes=args.max_episodes,
    )
    if dataset.observations.shape[1] != config.observation_dim:
        raise ValueError("dataset/base observation dimensions differ")
    if dataset.actions.shape[1] != config.action_dim:
        raise ValueError("dataset/base action dimensions differ")
    agent = CCFQERPIAgent(
        config,
        device,
        initialisation["observation_mean"],
        initialisation["observation_std"],
        initialisation["actor"],
        initialisation["critic"],
    )
    preprocessing_started = time.perf_counter()
    inverse_commands, inverse_audit = precompute_inverse_commands(
        agent, dataset.observations, batch_size=args.precompute_batch_size
    )
    next_inverse_commands, next_inverse_audit = precompute_inverse_commands(
        agent, dataset.next_observations, batch_size=args.precompute_batch_size
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    tensors = make_cpu_dataset_tensors(
        dataset,
        commanded_actions,
        inverse_commands,
        next_inverse_commands,
    )
    audit_indices = fixed_audit_indices(
        dataset.observations.shape[0], args.audit_observations, args.audit_index_seed
    )
    code_hashes = {
        "train_cc_fqe_rpi_sha256": sha256_file(Path(__file__).resolve()),
        "cc_fqe_rpi_core_sha256": sha256_file(
            Path(__file__).resolve().with_name("cc_fqe_rpi_core.py")
        ),
        "td3bc_core_sha256": sha256_file(
            Path(__file__).resolve().with_name("td3bc_core.py")
        ),
    }
    provenance: Dict[str, object] = {
        "method": "CC-FQE-to-one-step-RPI",
        "method_config": asdict(config),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "schema": dataset.source_schema,
            "transition_count": int(dataset.observations.shape[0]),
            "reward_source": "stored_environment_reward_scaled_by_shared_loader",
            "reward_scale": float(dataset.reward_scale),
            "raw_mc_next_used": False,
        },
        "base_checkpoint": {
            "path": str(base_checkpoint_path),
            "sha256": base_checkpoint_hash,
            **base_audit,
        },
        "channel_calibration": calibration_audit,
        "action_channels": action_channel_audit,
        "inverse_current": inverse_audit,
        "inverse_next": next_inverse_audit,
        "audit_indices": {
            "seed": int(args.audit_index_seed),
            "count": int(audit_indices.size),
            "sha256": array_sha256(audit_indices),
            "selection": "fixed_seed_global_permutation_without_replacement",
        },
        "operator": {
            "critic_current_action": "logged_executed_action",
            "continuation_mode": config.continuation_mode,
            "continuation_beta": config.continuation_beta,
            "target_beta": config.target_beta,
            "twin_reduction": "min_of_channel_expectations",
            "hubl_mc_mixing": False,
            "td3_target_policy_smoothing": False,
            "timeouts": "shared loader drops timeout final transition; true terminals stop bootstrap",
        },
        "support_warning": (
            "known conditional channel-support expansion penalty is heuristic; "
            "beta_target>beta_source can violate positivity and state occupancy "
            "shift remains unidentified from the fixed data"
        ),
        "equal_compute_control": (
            "set continuation_mode=source; only FQE continuation beta changes, "
            "while inverse, RPI target beta, K, stages, and Q-row shapes remain fixed"
        ),
        "code": code_hashes,
        "train_seed": int(args.train_seed),
        "minibatch_seeds": {
            "fqe": int(args.train_seed + 1729),
            "adapter": int(args.train_seed + 1730),
        },
        "nonstandard_schedule": bool(
            config.fqe_updates != 20_000
            or config.adapter_updates != 5_000
            or config.train_channel_samples != 4
            or config.audit_channel_samples != 64
        ),
    }
    # Paths, logging cadence, and resume location are deliberately excluded;
    # everything that changes samples, objectives, or weights is included.
    provenance_fingerprint = json_sha256(provenance)
    provenance["fingerprint_sha256"] = provenance_fingerprint

    dry_result = {
        "status": "dry_run",
        "provenance": provenance,
        "preprocessing_seconds": preprocessing_seconds,
        "agent": agent.metadata(),
    }
    if args.dry_run:
        return dry_result

    config_path = output_dir / "config.json"
    progress_path = output_dir / "progress.jsonl"
    if resume_path is None:
        atomic_json(config_path, provenance)
        progress_path.write_text("", encoding="utf-8")
    else:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                saved_config = json.load(handle)
            if saved_config.get("fingerprint_sha256") != provenance_fingerprint:
                raise ValueError("output directory config fingerprint mismatch")
        else:
            atomic_json(config_path, provenance)
        if not progress_path.exists():
            progress_path.write_text("", encoding="utf-8")
        # Generator states are CPU ByteTensors even for CUDA generators, and
        # the minibatch generators are explicitly CPU generators.
        payload = torch.load(resume_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("resume checkpoint is not a mapping")

    fqe_batch_generator = torch.Generator(device="cpu")
    fqe_batch_generator.manual_seed(args.train_seed + 1729)
    adapter_batch_generator = torch.Generator(device="cpu")
    adapter_batch_generator.manual_seed(args.train_seed + 1730)
    if resume_path is not None:
        restore_training_checkpoint(
            payload,
            agent,
            fqe_batch_generator=fqe_batch_generator,
            adapter_batch_generator=adapter_batch_generator,
            provenance_fingerprint=provenance_fingerprint,
        )
        _append_progress(
            progress_path,
            {
                "event": "resume",
                "stage": agent.stage,
                "fqe_steps": agent.fqe_steps,
                "adapter_steps": agent.adapter_steps,
                "checkpoint": str(resume_path),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    latest_path = output_dir / "latest.pt"
    last_metrics: Dict[str, float] = {}
    optimizer_seconds = 0.0
    fqe_keys = (
        "observations",
        "executed_actions",
        "next_observations",
        "next_inverse_commands",
        "rewards",
        "terminals",
    )
    while agent.stage == "fqe":
        update_started = time.perf_counter()
        batch = sample_batch(
            tensors,
            fqe_keys,
            batch_size=args.batch_size,
            generator=fqe_batch_generator,
            device=device,
        )
        last_metrics = agent.fqe_update(batch)
        optimizer_seconds += time.perf_counter() - update_started
        step = agent.fqe_steps
        if step == 1 or step % args.log_period == 0 or step == config.fqe_updates:
            _append_progress(
                progress_path,
                {
                    "event": "fqe",
                    "step": step,
                    "stage_total": config.fqe_updates,
                    "elapsed_seconds": time.perf_counter() - started,
                    **last_metrics,
                },
            )
        if step % args.checkpoint_period == 0 or step == config.fqe_updates:
            atomic_torch_save(
                latest_path,
                training_checkpoint(
                    agent,
                    fqe_batch_generator=fqe_batch_generator,
                    adapter_batch_generator=adapter_batch_generator,
                    provenance_fingerprint=provenance_fingerprint,
                    provenance=provenance,
                ),
            )

    if agent.stage == "value_scale_calibration":
        calibration_arrays, calibration_metadata = audit_policy_k64(
            agent,
            tensors["observations"],
            tensors["inverse_commands"],
            tensors["logged_commands"],
            audit_indices,
            batch_size=args.audit_batch_size,
        )
        raw_std = float(
            np.asarray(calibration_arrays["baseline_value"], dtype=np.float64).std()
        )
        agent.enter_adapter_stage(max(raw_std, config.q_scale_epsilon))
        _append_progress(
            progress_path,
            {
                "event": "value_scale_calibration",
                "value_scale": agent.value_scale,
                "raw_std": raw_std,
                "epsilon_floor_used": raw_std < config.q_scale_epsilon,
                "audit": calibration_metadata,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        atomic_torch_save(
            latest_path,
            training_checkpoint(
                agent,
                fqe_batch_generator=fqe_batch_generator,
                adapter_batch_generator=adapter_batch_generator,
                provenance_fingerprint=provenance_fingerprint,
                provenance=provenance,
            ),
        )

    adapter_keys = ("observations", "inverse_commands", "logged_commands")
    while agent.stage == "adapter":
        update_started = time.perf_counter()
        batch = sample_batch(
            tensors,
            adapter_keys,
            batch_size=args.batch_size,
            generator=adapter_batch_generator,
            device=device,
        )
        last_metrics = agent.adapter_update(batch)
        optimizer_seconds += time.perf_counter() - update_started
        step = agent.adapter_steps
        if step == 1 or step % args.log_period == 0 or step == config.adapter_updates:
            _append_progress(
                progress_path,
                {
                    "event": "adapter",
                    "step": step,
                    "stage_total": config.adapter_updates,
                    "elapsed_seconds": time.perf_counter() - started,
                    **last_metrics,
                },
            )
        if step % args.checkpoint_period == 0 or step == config.adapter_updates:
            atomic_torch_save(
                latest_path,
                training_checkpoint(
                    agent,
                    fqe_batch_generator=fqe_batch_generator,
                    adapter_batch_generator=adapter_batch_generator,
                    provenance_fingerprint=provenance_fingerprint,
                    provenance=provenance,
                ),
            )

    if agent.stage != "complete":
        raise RuntimeError(f"training ended in unexpected stage {agent.stage}")
    audit_arrays, audit_metadata = audit_policy_k64(
        agent,
        tensors["observations"],
        tensors["inverse_commands"],
        tensors["logged_commands"],
        audit_indices,
        batch_size=args.audit_batch_size,
    )
    raw_audit_path = output_dir / "audit_k64_raw.npz"
    temporary_audit_path = output_dir / "audit_k64_raw.tmp.npz"
    np.savez_compressed(temporary_audit_path, **audit_arrays)
    os.replace(temporary_audit_path, raw_audit_path)
    audit_payload = audit_summary(audit_arrays, audit_metadata)
    audit_payload.update(
        {
            "raw_path": str(raw_audit_path),
            "raw_file_sha256": sha256_file(raw_audit_path),
            "primary_interpretation": (
                "diagnostic only; the base checkpoint has already seen the fixed "
                "dataset and closed-loop target return remains primary"
            ),
        }
    )
    atomic_json(output_dir / "audit_k64.json", audit_payload)
    wall_time = time.perf_counter() - started
    summary: Dict[str, object] = {
        "status": "complete",
        "method": "CC-FQE-to-one-step-RPI",
        "continuation_mode": config.continuation_mode,
        "continuation_beta": config.continuation_beta,
        "target_beta": config.target_beta,
        "source_beta": config.source_beta,
        "fqe_updates": agent.fqe_steps,
        "adapter_updates": agent.adapter_steps,
        "train_channel_samples": config.train_channel_samples,
        "audit_channel_samples": config.audit_channel_samples,
        "value_scale": agent.value_scale,
        "wall_time_seconds": wall_time,
        "preprocessing_seconds": preprocessing_seconds,
        "optimizer_seconds": optimizer_seconds,
        "last_train_metrics": last_metrics,
        "agent": agent.metadata(),
        "audit_k64": audit_payload,
        "checkpoint": str(latest_path),
        "checkpoint_sha256": sha256_file(latest_path),
        "provenance_fingerprint": provenance_fingerprint,
        "q_input_rows": {
            "fqe_current_rows_per_critic": int(agent.fqe_steps * args.batch_size),
            "fqe_target_rows_per_critic": int(
                agent.fqe_steps * args.batch_size * config.train_channel_samples
            ),
            "adapter_rows_per_critic": int(
                agent.adapter_steps
                * args.batch_size
                * 2
                * config.train_channel_samples
            ),
            "audit_rows_per_critic": int(
                audit_indices.size * 2 * config.audit_channel_samples
            ),
        },
        "selection_rule": "fixed final 20k-FQE/5k-adapter checkpoint; no run exclusion",
        "statistical_warning": provenance["support_warning"],
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--channel-calibration")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument(
        "--continuation-mode", choices=CONTINUATION_MODES, default="target"
    )
    parser.add_argument("--source-beta", type=float, default=1.0)
    parser.add_argument("--target-beta", type=float, default=1.2498949)
    parser.add_argument("--fqe-updates", type=int, default=20_000)
    parser.add_argument("--adapter-updates", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-channel-samples", type=int, default=4)
    parser.add_argument("--audit-channel-samples", type=int, default=64)
    parser.add_argument("--continuation-noise-seed", type=int, default=314159)
    parser.add_argument("--actor-noise-seed", type=int, default=314160)
    parser.add_argument("--audit-noise-seed", type=int, default=314161)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--adapter-depth", type=int, default=2)
    parser.add_argument("--delta-max", type=float, default=0.25)
    parser.add_argument("--value-alpha", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=1.0)
    parser.add_argument("--support-penalty", type=float, default=1.0)
    parser.add_argument("--q-scale-epsilon", type=float, default=1e-6)
    parser.add_argument("--audit-observations", type=int, default=4096)
    parser.add_argument("--audit-index-seed", type=int, default=271828)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--precompute-batch-size", type=int, default=8192)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--checkpoint-period", type=int, default=1000)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate checkpoint, data, inverse, and provenance without updates",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
