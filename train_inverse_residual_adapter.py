"""Train a value-aware residual command adapter on a frozen TD3+BC checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from inverse_residual_core import (
    BASELINE_TRANSFORMS,
    VALUE_ESTIMATORS,
    InverseResidualAdapter,
    InverseResidualConfig,
    ResampledAntitheticChannel,
    adapter_objective,
    baseline_commands_from_desired,
    marginalized_q1_from_physical_actions,
    module_state_sha256,
    value_estimator_action_rows,
    value_estimator_metadata,
)
from td3bc_core import TD3BCAgent, TD3BCConfig
from train_iql import save_json, seed_everything, sha256_file


CALIBRATION_ESTIMATOR = "censored_uniform_plus_clip_mle"
CALIBRATION_PAIR_SCHEMA = "paired_uniform_clip_channel_v1"
OBSERVATION_SPLIT_SCHEMA = "adapter_observation_split_v1"
RESUME_SIGNATURE_SCHEMA = "inverse_residual_resume_signature_v1"
ADAPTER_CHECKPOINT_FORMAT = "inverse_residual_adapter_v2"


def _array_index_sha256(indices: np.ndarray) -> str:
    """Hash an ordered index vector including its shape and canonical dtype."""

    values = np.asarray(indices, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def prepare_fresh_output_dir(path: Path) -> Path:
    """Create or accept an empty run directory, refusing any prior contents."""

    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"output directory is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"refusing to reuse nonempty output directory: {resolved}"
            )
    else:
        resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def require_new_output_file(path: Path) -> Path:
    """Refuse to replace either a completed output or a stale atomic temp file."""

    resolved = path.resolve()
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    if resolved.exists() or temporary.exists():
        existing = resolved if resolved.exists() else temporary
        raise FileExistsError(f"refusing to overwrite existing output: {existing}")
    return resolved


def prepare_resume_output_dir(path: Path) -> Path:
    """Validate the fixed files needed for an in-place exact continuation."""

    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {resolved}")
    required = ("config.json", "progress.jsonl", "latest.pt")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"resume directory lacks files: {missing}")
    stale_checkpoint = resolved / "latest.pt.tmp"
    if stale_checkpoint.exists():
        raise FileExistsError(
            f"refusing resume with stale checkpoint temp file: {stale_checkpoint}"
        )
    return resolved


def _implementation_manifest() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "train_sha256": sha256_file(Path(__file__).resolve()),
        "core_sha256": sha256_file(root / "inverse_residual_core.py"),
        "evaluation_controls_sha256": sha256_file(root / "evaluation_controls.py"),
        "td3bc_core_sha256": sha256_file(root / "td3bc_core.py"),
        "train_iql_sha256": sha256_file(root / "train_iql.py"),
    }


def _runtime_manifest(device: torch.device) -> Dict[str, object]:
    """Record runtime knobs that can change a supposedly exact continuation."""

    concrete_device = torch.device(device)
    if concrete_device.type == "cuda" and concrete_device.index is None:
        concrete_device = torch.device("cuda", torch.cuda.current_device())
    runtime: Dict[str, object] = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "device": str(concrete_device),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if concrete_device.type == "cuda":
        properties = torch.cuda.get_device_properties(concrete_device)
        runtime["cuda_device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(concrete_device)),
            "total_memory_bytes": int(properties.total_memory),
            "device_count": int(torch.cuda.device_count()),
        }
    else:
        runtime["cuda_device"] = None
    return runtime


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_global_rng_state() -> Dict[str, object]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_global_rng_state(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint lacks global RNG state")
    required = ("python_random", "numpy_random", "torch_cpu", "torch_cuda")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"resume checkpoint lacks RNG fields: {missing}")
    random.setstate(payload["python_random"])
    numpy_state = payload["numpy_random"]
    if not isinstance(numpy_state, tuple):
        raise ValueError("resume checkpoint NumPy RNG state is invalid")
    np.random.set_state(numpy_state)
    torch_cpu = payload["torch_cpu"]
    if not torch.is_tensor(torch_cpu):
        raise ValueError("resume checkpoint torch CPU RNG state is invalid")
    torch.set_rng_state(torch_cpu.detach().cpu())
    cuda_states = payload["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise ValueError("resume checkpoint torch CUDA RNG states are invalid")
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint requires CUDA RNG state")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("resume checkpoint CUDA device count differs")
        if not all(torch.is_tensor(state) for state in cuda_states):
            raise ValueError("resume checkpoint contains an invalid CUDA RNG tensor")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale checkpoint: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_json_object(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _validate_progress_for_resume(path: Path, checkpoint_step: int) -> None:
    last_train_step = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid progress JSON on line {line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"progress line {line_number} is not an object")
            if event.get("event") == "train":
                step = int(event.get("step", -1))
                if step <= last_train_step:
                    raise ValueError("progress train steps are not strictly increasing")
                last_train_step = step
    if checkpoint_step > 0 and last_train_step < 0:
        raise ValueError("resume checkpoint has updates but progress has no train event")
    if last_train_step > checkpoint_step:
        raise ValueError(
            "progress contains a train step newer than the recoverable checkpoint"
        )


def load_frozen_physical_agent(
    checkpoint_path: Path, device: torch.device
) -> Tuple[TD3BCAgent, Dict[str, object], int]:
    """Load actor and both critic pairs explicitly, then freeze every base tensor."""

    wrapper = torch.load(checkpoint_path, map_location=device)
    if not isinstance(wrapper, dict) or "agent" not in wrapper:
        raise KeyError("base checkpoint must contain wrapper['agent']")
    payload = wrapper["agent"]
    if not isinstance(payload, dict):
        raise ValueError("base checkpoint agent payload must be a mapping")
    required = (
        "config",
        "observation_mean",
        "observation_std",
        "actor",
        "critic",
        "actor_target",
        "critic_target",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"base checkpoint lacks fields: {missing}")
    config = TD3BCConfig(**payload["config"])
    if config.action_pairing != "executed_executed":
        raise ValueError(
            "inverse-residual adaptation requires an executed/executed base: "
            "its actor must denote desired physical action and its critic must "
            "be trained on executed physical actions"
        )
    if config.marginalize_execution_noise:
        raise ValueError("base checkpoint must not already marginalize command noise")
    mean = payload["observation_mean"]
    std = payload["observation_std"]
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    if torch.is_tensor(std):
        std = std.detach().cpu().numpy()
    agent = TD3BCAgent(
        config,
        device,
        np.asarray(mean, dtype=np.float32),
        np.asarray(std, dtype=np.float32),
    )
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.critic.load_state_dict(payload["critic"], strict=True)
    agent.actor_target.load_state_dict(payload["actor_target"], strict=True)
    agent.critic_target.load_state_dict(payload["critic_target"], strict=True)
    for module in (
        agent.actor,
        agent.critic,
        agent.actor_target,
        agent.critic_target,
    ):
        module.requires_grad_(False)
        module.eval()
    step = int(wrapper.get("step", payload.get("total_updates", -1)))
    return agent, wrapper, step


def _load_observations_and_episode_units(
    path: Path, max_observations: Optional[int]
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    """Read states and, when recoverable, an episode id for every state row."""

    if max_observations is not None and max_observations <= 0:
        raise ValueError("max_observations must be positive when provided")
    with h5py.File(path, "r") as handle:
        if "observations" in handle:
            total = int(handle["observations"].shape[0])
            stop = total if max_observations is None else min(total, max_observations)
            observations = np.asarray(handle["observations"][:stop], dtype=np.float32)
            terminal_key = next(
                (key for key in ("terminals", "terminations") if key in handle),
                None,
            )
            timeout_key = next(
                (key for key in ("timeouts", "truncations") if key in handle),
                None,
            )
            episode_units: Optional[np.ndarray] = None
            boundary_source = "unavailable"
            if terminal_key is not None or timeout_key is not None:
                if terminal_key is not None and int(handle[terminal_key].shape[0]) != total:
                    raise ValueError("flat terminal array is not aligned with observations")
                if timeout_key is not None and int(handle[timeout_key].shape[0]) != total:
                    raise ValueError("flat timeout array is not aligned with observations")
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
                boundary_source = "+".join(
                    key for key in (terminal_key, timeout_key) if key is not None
                )
            layout = {
                "storage_layout": "flat",
                "source_observation_count": total,
                "boundary_source": boundary_source,
                "episode_unit_count": (
                    int(np.unique(episode_units).size)
                    if episode_units is not None
                    else None
                ),
            }
        else:
            keys = sorted(
                (key for key in handle if key.startswith("episode_")),
                key=lambda key: int(key.rsplit("_", 1)[1]),
            )
            pieces: List[np.ndarray] = []
            unit_pieces: List[np.ndarray] = []
            remaining = max_observations
            for unit_id, key in enumerate(keys):
                group = handle[key]
                if "observations" not in group:
                    raise KeyError(f"{key} lacks observations")
                values = np.asarray(group["observations"], dtype=np.float32)
                if remaining is not None:
                    values = values[:remaining]
                    remaining -= values.shape[0]
                if values.shape[0] > 0:
                    pieces.append(values)
                    unit_pieces.append(
                        np.full(values.shape[0], unit_id, dtype=np.int64)
                    )
                if remaining == 0:
                    break
            if not pieces:
                raise ValueError("dataset contains no observations")
            observations = np.concatenate(pieces, axis=0)
            episode_units = np.concatenate(unit_pieces, axis=0)
            layout = {
                "storage_layout": "episodic_groups",
                "source_observation_count": int(
                    sum(int(handle[key]["observations"].shape[0]) for key in keys)
                ),
                "boundary_source": "episode_group_membership",
                "episode_unit_count": int(np.unique(episode_units).size),
            }
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError("observations must be a nonempty rank-two array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations contain non-finite values")
    if episode_units is not None and episode_units.shape != (observations.shape[0],):
        raise RuntimeError("episode-unit construction is not aligned with observations")
    return observations, episode_units, layout


def load_observations(path: Path, max_observations: Optional[int]) -> np.ndarray:
    """Read only state observations from flat or episodic HDF5 storage."""

    observations, _, _ = _load_observations_and_episode_units(
        path, max_observations
    )
    return observations


def build_observation_split(
    path: Path,
    max_observations: Optional[int],
    audit_fraction: float,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build a deterministic train/audit partition, preferring whole episodes."""

    fraction = float(audit_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("audit_fraction must be finite and lie strictly in (0, 1)")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    observations, episode_units, layout = _load_observations_and_episode_units(
        path, max_observations
    )
    count = int(observations.shape[0])
    if count < 2:
        raise ValueError("at least two observations are required for a disjoint split")
    rng = np.random.default_rng(int(split_seed))
    target_audit_count = max(1, min(count - 1, int(np.ceil(fraction * count))))

    if episode_units is not None:
        unique_units, unit_counts = np.unique(episode_units, return_counts=True)
        count_by_unit = {
            int(unit): int(unit_count)
            for unit, unit_count in zip(unique_units, unit_counts)
        }
    else:
        unique_units = np.asarray([])
        count_by_unit = {}
    if episode_units is not None and unique_units.size >= 2:
        permuted_units = rng.permutation(unique_units)
        audit_units: List[int] = []
        selected_count = 0
        # Always leave at least one complete episode for training.
        for unit in permuted_units[:-1]:
            audit_units.append(int(unit))
            selected_count += count_by_unit[int(unit)]
            if selected_count >= target_audit_count:
                break
        audit_mask = np.isin(
            episode_units, np.asarray(audit_units, dtype=np.int64)
        )
        split_method = "whole_episode_permutation_to_state_count"
        audit_unit_array = np.sort(np.asarray(audit_units, dtype=np.int64))
    else:
        audit_positions = rng.permutation(count)[:target_audit_count]
        audit_mask = np.zeros(count, dtype=np.bool_)
        audit_mask[audit_positions] = True
        split_method = "fixed_index_permutation_no_episode_boundaries"
        audit_unit_array = np.asarray([], dtype=np.int64)

    audit_indices = np.flatnonzero(audit_mask).astype(np.int64)
    train_indices = np.flatnonzero(~audit_mask).astype(np.int64)
    if train_indices.size == 0 or audit_indices.size == 0:
        raise RuntimeError("deterministic split produced an empty partition")
    metadata: Dict[str, object] = {
        "schema": OBSERVATION_SPLIT_SCHEMA,
        "method": split_method,
        "seed": int(split_seed),
        "requested_audit_fraction": fraction,
        "max_observations": (
            int(max_observations) if max_observations is not None else None
        ),
        "considered_observation_count": count,
        "train_observation_count": int(train_indices.size),
        "audit_observation_count": int(audit_indices.size),
        "realized_audit_fraction": float(audit_indices.size / count),
        "train_indices_sha256": _array_index_sha256(train_indices),
        "audit_indices_sha256": _array_index_sha256(audit_indices),
        "audit_episode_units_sha256": _array_index_sha256(audit_unit_array),
        "audit_episode_unit_count": int(audit_unit_array.size),
        "holdout_scope": "adapter_sgd_and_value_scale_calibration",
        "base_critic_training_partition_controlled": False,
        **layout,
    }
    return observations, train_indices, audit_indices, metadata


def resolve_channel_calibration(
    calibration_path: Optional[Path], requested_beta: Optional[float]
) -> Tuple[float, Dict[str, object]]:
    """Resolve the model channel beta and retain auditable calibration provenance."""

    if calibration_path is None:
        if requested_beta is None:
            raise ValueError(
                "provide --channel-calibration or --execution-noise-beta"
            )
        beta = float(requested_beta)
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("execution_noise_beta must be finite and non-negative")
        return beta, {
            "source": "cli_known_beta_without_pair_calibration",
            "beta": beta,
            "calibration_path": None,
            "calibration_sha256": None,
            "pair_count": None,
        }
    path = calibration_path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("channel calibration JSON must contain an object")
    if payload.get("status") != "complete":
        raise ValueError("channel calibration status must be 'complete'")
    if payload.get("estimator") != CALIBRATION_ESTIMATOR:
        raise ValueError(
            f"channel calibration estimator must be {CALIBRATION_ESTIMATOR!r}"
        )
    if payload.get("estimator_uses_provenance_beta") is not False:
        raise ValueError(
            "channel calibration must explicitly state that provenance beta was unused"
        )
    try:
        beta = float(payload["estimate"]["beta_mle"])
        pair_count = int(payload["pair_count"])
        action_dim = int(payload["action_dim"])
        action_low = float(payload["action_low"])
        action_high = float(payload["action_high"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "channel calibration lacks a valid estimate, count, dimension, or bounds"
        ) from exc
    if (
        not np.isfinite(beta)
        or beta < 0.0
        or pair_count <= 0
        or action_dim <= 0
        or not np.isfinite(action_low)
        or not np.isfinite(action_high)
        or action_low >= action_high
    ):
        raise ValueError("channel calibration numeric schema is invalid")
    for field in (
        "pair_archive_sha256",
        "commands_sha256",
        "executed_sha256",
        "calibration_script_sha256",
    ):
        if not _is_sha256(payload.get(field)):
            raise ValueError(f"channel calibration {field} is not a SHA256 digest")
    input_provenance = payload.get("input_provenance")
    if not isinstance(input_provenance, dict):
        raise ValueError("channel calibration lacks input_provenance")
    if input_provenance.get("schema") != CALIBRATION_PAIR_SCHEMA:
        raise ValueError("channel calibration pair schema is unsupported")
    if input_provenance.get("channel") != "iid_uniform_additive_then_clip":
        raise ValueError("channel calibration provenance has the wrong channel")
    if input_provenance.get("contains_state_or_transition_fields") is not False:
        raise ValueError("channel calibration pairs must be state/transition-free")
    consistency_fields = {
        "pair_count": pair_count,
        "action_dim": action_dim,
        "action_low": action_low,
        "action_high": action_high,
        "commands_sha256": payload["commands_sha256"],
        "executed_sha256": payload["executed_sha256"],
    }
    for field, expected in consistency_fields.items():
        observed = input_provenance.get(field)
        if observed != expected:
            raise ValueError(
                f"channel calibration {field} disagrees with pair provenance"
            )
    if requested_beta is not None and float(requested_beta) != beta:
        raise ValueError(
            "when calibration is provided its beta_mle is authoritative; omit "
            "--execution-noise-beta or pass the exact estimate"
        )
    return beta, {
        "source": "censored_uniform_plus_clip_pair_calibration",
        "beta": beta,
        "calibration_path": str(path),
        "calibration_sha256": sha256_file(path),
        "estimator": payload.get("estimator"),
        "pair_count": pair_count,
        "action_dim": action_dim,
        "action_low": action_low,
        "action_high": action_high,
        "pair_archive_path": payload.get("pair_archive_path"),
        "pair_archive_sha256": payload.get("pair_archive_sha256"),
        "commands_sha256": payload.get("commands_sha256"),
        "executed_sha256": payload.get("executed_sha256"),
        "uncertainty_beta_interval": (
            payload.get("uncertainty", {}).get("beta_interval")
            if isinstance(payload.get("uncertainty"), dict)
            else None
        ),
        "input_provenance": input_provenance,
    }


@torch.no_grad()
def precompute_baseline_commands(
    base_agent: TD3BCAgent,
    observations: np.ndarray,
    config: InverseResidualConfig,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Cache normalized states and the declared baseline commands on CPU."""

    if batch_size <= 0:
        raise ValueError("precompute batch_size must be positive")
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    mean = base_agent.observation_mean.detach().cpu()
    std = base_agent.observation_std.detach().cpu()
    normalized = (observation_tensor - mean) / std
    inverse_commands = torch.empty(
        (observations.shape[0], config.action_dim), dtype=torch.float32
    )
    transform_values = 0
    transform_saturations = 0
    command_at_bound = 0
    for start in range(0, observations.shape[0], batch_size):
        stop = min(start + batch_size, observations.shape[0])
        normalized_device = normalized[start:stop].to(base_agent.device)
        desired = base_agent.actor(normalized_device)
        commands, audit = baseline_commands_from_desired(desired, config)
        inverse_commands[start:stop] = commands.cpu()
        transform_values += int(audit["command_transform_value_count"])
        transform_saturations += int(audit["command_transform_saturation_count"])
        command_at_bound += int(audit["transformed_command_at_bound_count"])
    return normalized, inverse_commands, {
        "observation_count": int(observations.shape[0]),
        "action_value_count": int(transform_values),
        "inverse_saturation_count": int(transform_saturations),
        "inverse_saturation_fraction": transform_saturations
        / max(transform_values, 1),
        "inverse_command_at_bound_count": int(command_at_bound),
        "inverse_command_at_bound_fraction": command_at_bound
        / max(transform_values, 1),
    }


def _float_metrics(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach()) for key, value in values.items()}


@torch.no_grad()
def calibrate_value_scale(
    base_agent: TD3BCAgent,
    normalized_cpu: torch.Tensor,
    inverse_commands_cpu: torch.Tensor,
    config: InverseResidualConfig,
    observation_count: int,
    train_seed: int,
) -> Tuple[float, Dict[str, object]]:
    """Freeze a reward-shift-invariant Q0 standard-deviation scale."""

    count = min(int(observation_count), normalized_cpu.shape[0])
    if count <= 0:
        raise ValueError("scale calibration observation_count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(train_seed + 65537)
    indices = torch.randperm(
        normalized_cpu.shape[0], generator=generator
    )[:count]
    normalized = normalized_cpu[indices].to(base_agent.device)
    inverse_commands = inverse_commands_cpu[indices].to(base_agent.device)
    calibration_config = replace(
        config, execution_noise_seed=config.execution_noise_seed + 104729
    )
    channel = ResampledAntitheticChannel(calibration_config, base_agent.device)
    # Use the tensor's concrete device (e.g. ``cuda:0``), not the possibly
    # implicit CLI spelling ``cuda``.
    physical_actions, _ = value_estimator_action_rows(inverse_commands, channel)
    q0 = marginalized_q1_from_physical_actions(
        base_agent.critic, normalized, physical_actions
    )
    raw_std = float(q0.std(unbiased=False))
    scale = max(raw_std, config.q_scale_epsilon)
    return scale, {
        "method": "frozen_std_of_declared_baseline_value_estimator_q1",
        "baseline_transform": config.baseline_transform,
        "value_estimator": value_estimator_metadata(config),
        "observation_count": int(count),
        "channel_seed": (
            int(calibration_config.execution_noise_seed)
            if config.value_estimator == "sampled_expected_q1"
            else None
        ),
        "channel_sample_calls": int(channel.draw_calls),
        "channel_half_vectors_drawn": int(channel.half_vectors_drawn),
        "q1_input_rows": int(count * config.execution_noise_samples),
        "q0_mean": float(q0.mean()),
        "q0_std": raw_std,
        "q0_min": float(q0.min()),
        "q0_max": float(q0.max()),
        "scale": scale,
        "epsilon_floor_used": bool(raw_std < config.q_scale_epsilon),
    }


def _adapter_checkpoint(
    *,
    step: int,
    adapter: InverseResidualAdapter,
    optimizer: torch.optim.Optimizer,
    channel: ResampledAntitheticChannel,
    index_generator: torch.Generator,
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
    return {
        "format": ADAPTER_CHECKPOINT_FORMAT,
        "step": int(step),
        "adapter_config": asdict(adapter.config),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "channel": channel.checkpoint(),
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


def run(args: argparse.Namespace) -> Dict[str, object]:
    started = time.perf_counter()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.checkpoint_period <= 0 or args.log_period <= 0:
        raise ValueError("checkpoint_period and log_period must be positive")
    output_dir = (
        prepare_resume_output_dir(Path(args.output_dir))
        if args.resume
        else prepare_fresh_output_dir(Path(args.output_dir))
    )
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    base_checkpoint = Path(args.base_checkpoint).resolve()
    base_checkpoint_sha = sha256_file(base_checkpoint)
    base_agent, _, base_step = load_frozen_physical_agent(base_checkpoint, device)
    base_modules = {
        "actor": base_agent.actor,
        "critic": base_agent.critic,
        "actor_target": base_agent.actor_target,
        "critic_target": base_agent.critic_target,
    }
    base_parameter_hash_before = module_state_sha256(base_modules)

    channel_beta, channel_calibration = resolve_channel_calibration(
        Path(args.channel_calibration) if args.channel_calibration else None,
        args.execution_noise_beta,
    )

    seed_everything(args.train_seed)
    config = InverseResidualConfig(
        observation_dim=base_agent.config.observation_dim,
        action_dim=base_agent.config.action_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        max_action=base_agent.config.max_action,
        execution_noise_beta=channel_beta,
        execution_noise_samples=args.execution_noise_samples,
        execution_noise_seed=args.execution_noise_seed,
        delta_max=args.delta_max,
        alpha=args.alpha,
        residual_penalty=args.residual_penalty,
        q_scale_epsilon=args.q_scale_epsilon,
        baseline_transform=args.baseline_transform,
        value_estimator=args.value_estimator,
    )
    calibrated_action_dim = channel_calibration.get("action_dim")
    if calibrated_action_dim is not None and int(calibrated_action_dim) != (
        config.action_dim
    ):
        raise ValueError("channel calibration action_dim does not match base checkpoint")
    for field, expected in (
        ("action_low", -config.max_action),
        ("action_high", config.max_action),
    ):
        value = channel_calibration.get(field)
        if value is not None and float(value) != expected:
            raise ValueError(f"channel calibration {field} does not match base bounds")
    adapter = InverseResidualAdapter(config).to(device)
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
    channel = ResampledAntitheticChannel(config, device)

    dataset_path = Path(args.dataset).resolve()
    observations, train_indices, audit_indices, observation_split = (
        build_observation_split(
            dataset_path,
            args.max_observations,
            args.audit_fraction,
            args.split_seed,
        )
    )
    if observations.shape[1] != config.observation_dim:
        raise ValueError(
            f"dataset observation dimension {observations.shape[1]} does not "
            f"match base checkpoint {config.observation_dim}"
        )
    dataset_sha = sha256_file(dataset_path)
    train_observations = observations[train_indices]
    preprocessing_started = time.perf_counter()
    normalized_cpu, inverse_commands_cpu, inverse_audit = (
        precompute_baseline_commands(
            base_agent,
            train_observations,
            config,
            args.precompute_batch_size,
        )
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    value_scale, value_scale_audit = calibrate_value_scale(
        base_agent,
        normalized_cpu,
        inverse_commands_cpu,
        config,
        args.scale_calibration_observations,
        args.train_seed,
    )
    with torch.no_grad():
        check_count = min(1024, normalized_cpu.shape[0])
        initialized_commands, initial_residual, _ = adapter.compose_command(
            normalized_cpu[:check_count].to(device),
            inverse_commands_cpu[:check_count].to(device),
        )
        initial_difference = (
            initialized_commands
            - inverse_commands_cpu[:check_count].to(device)
        ).abs().max()
        if float(initial_difference) != 0.0 or int(torch.count_nonzero(initial_residual)):
            raise RuntimeError("zero initialization failed exact baseline-command identity")

    implementation_manifest = _implementation_manifest()
    runtime_manifest = _runtime_manifest(channel.device)
    initial_adapter_sha = module_state_sha256({"adapter": adapter})
    resume_signature: Dict[str, object] = {
        "schema": RESUME_SIGNATURE_SCHEMA,
        "adapter": asdict(config),
        "optimization": {
            "optimizer": "torch.optim.Adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "train_seed": int(args.train_seed),
            "precompute_batch_size": int(args.precompute_batch_size),
            "scale_calibration_observations": int(
                args.scale_calibration_observations
            ),
            "log_period": int(args.log_period),
            "checkpoint_period": int(args.checkpoint_period),
            "torch_threads": int(args.torch_threads),
            "value_estimator": config.value_estimator,
        },
        "base": {
            "path": str(base_checkpoint),
            "checkpoint_sha256": base_checkpoint_sha,
            "checkpoint_step": int(base_step),
            "parameter_sha256": base_parameter_hash_before,
            "action_pairing": base_agent.config.action_pairing,
        },
        "channel_calibration": channel_calibration,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha,
            "considered_observation_count": int(observations.shape[0]),
            "training_observation_count": int(train_indices.size),
            "held_out_audit_observation_count": int(audit_indices.size),
        },
        "observation_split": observation_split,
        "baseline_precomputation": inverse_audit,
        "value_scale_calibration": value_scale_audit,
        "value_estimator": value_estimator_metadata(config),
        "initial_adapter_parameter_sha256": initial_adapter_sha,
        "implementation": implementation_manifest,
        "runtime": runtime_manifest,
    }
    resume_signature_sha = _canonical_json_sha256(resume_signature)
    candidate_config_payload: Dict[str, object] = {
        "arguments": dict(vars(args)),
        "adapter": asdict(config),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha,
        "base_checkpoint_step": base_step,
        "base_action_pairing": base_agent.config.action_pairing,
        "base_parameter_sha256_before": base_parameter_hash_before,
        "baseline_transform": config.baseline_transform,
        "channel_calibration": channel_calibration,
        "dataset": resume_signature["dataset"],
        "observation_split": observation_split,
        "baseline_precomputation": {
            **inverse_audit,
            "seconds": preprocessing_seconds,
            "initial_adapter_max_abs_command_difference": float(initial_difference),
        },
        "value_scale_calibration": value_scale_audit,
        "channel": channel.metadata(),
        "value_estimator": value_estimator_metadata(config),
        "objective": {
            "formula": (
                "-alpha*mean(q_adapted-q_baseline)/"
                "frozen_std(q_baseline_calibration)"
                "+residual_penalty*mean((proposed_delta/delta_max)^2)"
            ),
            "q_value_estimator": config.value_estimator,
        },
        "q1_rows_per_state_per_update": int(2 * config.execution_noise_samples),
        "checkpoint_policy": (
            "atomic latest.pt at every log step, requested checkpoint period, "
            "and final step; checkpoint precedes append-only progress"
        ),
        "implementation": implementation_manifest,
        "runtime": runtime_manifest,
        "resume_signature": resume_signature,
        "resume_signature_sha256": resume_signature_sha,
        "selection_rule": "all requested updates retained; no run exclusion",
        "exact_training_resume_implemented": True,
    }
    progress_path = output_dir / "progress.jsonl"
    index_generator = torch.Generator(device="cpu")
    start_step = 0
    elapsed_seconds_offset = 0.0
    optimizer_seconds_offset = 0.0
    resume_count = 0
    prior_checkpoint_sha: Optional[str] = None

    if args.resume:
        checkpoint_path = output_dir / "latest.pt"
        prior_checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError("resume checkpoint is not a mapping")
        if checkpoint.get("format") != ADAPTER_CHECKPOINT_FORMAT:
            raise ValueError(
                "resume requires an exact-state inverse_residual_adapter_v2 checkpoint"
            )
        if checkpoint.get("exact_training_resume_implemented") is not True:
            raise ValueError("resume checkpoint does not declare exact-state support")
        stored_signature = checkpoint.get("resume_signature")
        if not isinstance(stored_signature, dict):
            raise ValueError("resume checkpoint lacks a structured resume signature")
        if checkpoint.get("resume_signature_sha256") != _canonical_json_sha256(
            stored_signature
        ):
            raise ValueError("resume checkpoint signature hash is invalid")
        if stored_signature != resume_signature:
            raise ValueError(
                "resume signature mismatch: parameters, data, base checkpoint, "
                "code, provenance, or runtime changed"
            )
        if checkpoint.get("adapter_config") != asdict(config):
            raise ValueError("resume checkpoint adapter config mismatch")
        if checkpoint.get("base_checkpoint") != str(base_checkpoint):
            raise ValueError("resume checkpoint base path mismatch")
        if checkpoint.get("base_checkpoint_sha256") != base_checkpoint_sha:
            raise ValueError("resume checkpoint base file hash mismatch")
        if checkpoint.get("base_parameter_sha256") != base_parameter_hash_before:
            raise ValueError("resume checkpoint base parameter hash mismatch")
        if int(checkpoint.get("base_checkpoint_step", -1)) != int(base_step):
            raise ValueError("resume checkpoint base step mismatch")
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError("resume checkpoint lacks its immutable run config")
        disk_config = _load_json_object(output_dir / "config.json")
        if disk_config != checkpoint_config:
            raise ValueError("on-disk config differs from the checkpoint config")
        if disk_config.get("resume_signature") != stored_signature or (
            disk_config.get("resume_signature_sha256")
            != checkpoint.get("resume_signature_sha256")
        ):
            raise ValueError("on-disk config resume provenance is inconsistent")
        start_step = int(checkpoint.get("step", -1))
        if start_step <= 0:
            raise ValueError("resume checkpoint step must be positive")
        prior_target = int(checkpoint.get("target_total_updates", -1))
        if prior_target < start_step:
            raise ValueError("resume checkpoint has an invalid prior update target")
        if args.updates <= start_step:
            raise ValueError(
                "--updates is the total target and must exceed the saved step "
                f"({start_step}) when --resume is used"
            )
        elapsed_seconds_offset = float(
            checkpoint.get("elapsed_seconds_completed", -1.0)
        )
        optimizer_seconds_offset = float(
            checkpoint.get("optimizer_seconds_completed", -1.0)
        )
        if (
            not np.isfinite(elapsed_seconds_offset)
            or elapsed_seconds_offset < 0.0
            or not np.isfinite(optimizer_seconds_offset)
            or optimizer_seconds_offset < 0.0
        ):
            raise ValueError("resume checkpoint time accounting is invalid")
        resume_count = int(checkpoint.get("resume_count", -1)) + 1
        if resume_count <= 0:
            raise ValueError("resume checkpoint resume counter is invalid")
        _validate_progress_for_resume(progress_path, start_step)
        adapter_state = checkpoint.get("adapter")
        optimizer_state = checkpoint.get("optimizer")
        channel_state = checkpoint.get("channel")
        index_state = checkpoint.get("index_generator_state")
        if not isinstance(adapter_state, dict) or not isinstance(
            optimizer_state, dict
        ):
            raise ValueError("resume checkpoint lacks adapter or optimizer state")
        if not isinstance(channel_state, dict):
            raise ValueError("resume checkpoint lacks channel state")
        if not torch.is_tensor(index_state):
            raise ValueError("resume checkpoint lacks index-generator state")
        expected_optimizer_groups = optimizer.state_dict()["param_groups"]
        if optimizer_state.get("param_groups") != expected_optimizer_groups:
            raise ValueError("resume checkpoint optimizer hyperparameters mismatch")
        expected_draw_calls = (
            start_step
            if config.value_estimator == "sampled_expected_q1"
            else 0
        )
        expected_half_vectors = (
            start_step
            * args.batch_size
            * (config.execution_noise_samples // 2)
            if config.value_estimator == "sampled_expected_q1"
            and config.execution_noise_beta > 0.0
            else 0
        )
        if int(channel_state.get("draw_calls", -1)) != expected_draw_calls or int(
            channel_state.get("half_vectors_drawn", -1)
        ) != expected_half_vectors:
            raise ValueError("resume checkpoint channel accounting mismatch")
        adapter.load_state_dict(adapter_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        channel.restore(channel_state)
        index_generator.set_state(index_state.detach().cpu())
        # Restore process-global RNGs last: setup, verification, and state loading
        # above are deliberately prevented from perturbing the resumed stream.
        _restore_global_rng_state(checkpoint.get("global_rng_state"))
        config_payload = checkpoint_config
        resume_event = {
            "event": "resume",
            "from_step": int(start_step),
            "target_total_updates": int(args.updates),
            "resume_count": int(resume_count),
            "checkpoint_sha256": prior_checkpoint_sha,
            "elapsed_seconds_offset": elapsed_seconds_offset,
            "optimizer_seconds_offset": optimizer_seconds_offset,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(resume_event, sort_keys=True) + "\n")
    else:
        index_generator.manual_seed(args.train_seed + 1729)
        config_payload = candidate_config_payload
        save_json(output_dir / "config.json", config_payload)
        progress_path.write_text("", encoding="utf-8")

    last_metrics: Dict[str, float] = {}
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
        inverse_commands = inverse_commands_cpu[indices].to(device)
        loss, diagnostic_tensors = adapter_objective(
            adapter,
            base_agent.critic,
            normalized,
            inverse_commands,
            channel,
            torch.as_tensor(value_scale, dtype=torch.float32, device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer_seconds_this_invocation += time.perf_counter() - update_started
        optimizer_seconds = (
            optimizer_seconds_offset + optimizer_seconds_this_invocation
        )
        last_metrics = {
            "loss": float(loss.detach()),
            **_float_metrics(diagnostic_tensors),
        }
        elapsed_seconds = elapsed_seconds_offset + time.perf_counter() - started
        should_log = (
            step == 1 or step % args.log_period == 0 or step == args.updates
        )
        should_checkpoint = (
            should_log
            or step % args.checkpoint_period == 0
            or step == args.updates
        )
        # Commit the recoverable state before its train record.  This keeps the
        # append-only progress file from ever advertising an unrecoverable step.
        if should_checkpoint:
            _atomic_torch_save(
                output_dir / "latest.pt",
                _adapter_checkpoint(
                    step=step,
                    adapter=adapter,
                    optimizer=optimizer,
                    channel=channel,
                    index_generator=index_generator,
                    base_checkpoint=base_checkpoint,
                    base_checkpoint_sha256=base_checkpoint_sha,
                    base_step=base_step,
                    base_parameter_sha256=base_parameter_hash_before,
                    config_payload=config_payload,
                    resume_signature=resume_signature,
                    elapsed_seconds_completed=elapsed_seconds,
                    optimizer_seconds_completed=optimizer_seconds,
                    target_total_updates=args.updates,
                    resume_count=resume_count,
                ),
            )
        if should_log:
            event = {
                "event": "train",
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                "optimizer_seconds": optimizer_seconds,
                "updates_per_optimizer_second": step
                / max(optimizer_seconds, 1e-12),
                "q1_input_rows_cumulative": int(
                    step
                    * args.batch_size
                    * 2
                    * config.execution_noise_samples
                ),
                **last_metrics,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    base_parameter_hash_after = module_state_sha256(base_modules)
    if base_parameter_hash_after != base_parameter_hash_before:
        raise RuntimeError("frozen base actor or critic changed during adapter training")
    invocation_wall_time = time.perf_counter() - started
    wall_time = elapsed_seconds_offset + invocation_wall_time
    optimizer_seconds = optimizer_seconds_offset + optimizer_seconds_this_invocation
    summary: Dict[str, object] = {
        "status": "complete",
        "updates": int(args.updates),
        "target_total_updates": int(args.updates),
        "started_from_step": int(start_step),
        "resumed": bool(args.resume),
        "resume_count": int(resume_count),
        "train_seed": int(args.train_seed),
        "wall_time_seconds": wall_time,
        "elapsed_seconds_before_invocation": elapsed_seconds_offset,
        "invocation_wall_time_seconds": invocation_wall_time,
        "preprocessing_seconds_this_invocation": preprocessing_seconds,
        "optimizer_seconds": optimizer_seconds,
        "optimizer_seconds_before_invocation": optimizer_seconds_offset,
        "optimizer_seconds_this_invocation": optimizer_seconds_this_invocation,
        "updates_per_optimizer_second": args.updates
        / max(optimizer_seconds, 1e-12),
        "q1_input_rows": int(
            args.updates
            * args.batch_size
            * 2
            * config.execution_noise_samples
        ),
        "q1_rows_per_state_per_update": int(2 * config.execution_noise_samples),
        "last_train_metrics": last_metrics,
        "channel": channel.metadata(),
        "baseline_transform": config.baseline_transform,
        "value_estimator": value_estimator_metadata(config),
        "channel_calibration": channel_calibration,
        "baseline_precomputation": inverse_audit,
        "value_scale_calibration": value_scale_audit,
        "base_parameter_sha256_before": base_parameter_hash_before,
        "base_parameter_sha256_after": base_parameter_hash_after,
        "base_parameters_unchanged": True,
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
        help=(
            "continue exact state from OUTPUT_DIR/latest.pt; --updates is the "
            "new total update target, not an additional-update count"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument(
        "--updates",
        type=int,
        default=5_000,
        help=(
            "total adapter-only update target (also under --resume); intended "
            "screening range is 2,000-10,000"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=1.0)
    parser.add_argument("--delta-max", type=float, default=0.25)
    parser.add_argument(
        "--baseline-transform",
        choices=BASELINE_TRANSFORMS,
        default="inverse",
        help=(
            "inverse learns around the known channel-mean inverse; identity is "
            "the equal-capacity direct-residual control around the base actor"
        ),
    )
    parser.add_argument(
        "--value-estimator",
        choices=VALUE_ESTIMATORS,
        default="sampled_expected_q1",
        help=(
            "sampled_expected_q1 estimates E[Q1] with paired antithetic "
            "execution samples; q1_at_channel_mean is the equal-Q-row "
            "mechanism ablation Q1(E[action]) using the exact clipped-uniform mean"
        ),
    )
    parser.add_argument("--q-scale-epsilon", type=float, default=1e-6)
    parser.add_argument("--execution-noise-beta", type=float)
    parser.add_argument(
        "--channel-calibration",
        help=(
            "calibration JSON from calibrate_uniform_channel.py; its beta_mle "
            "is authoritative and its pair provenance is recorded"
        ),
    )
    parser.add_argument("--execution-noise-samples", type=int, default=2)
    parser.add_argument("--execution-noise-seed", type=int, default=271828)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument(
        "--audit-fraction",
        type=float,
        default=0.1,
        help=(
            "held-out source-state fraction; whole episodes are used whenever "
            "terminal/timeout boundaries are available"
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=424242,
        help="fixed seed for the recorded disjoint train/audit state partition",
    )
    parser.add_argument("--precompute-batch-size", type=int, default=8192)
    parser.add_argument("--scale-calibration-observations", type=int, default=4096)
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
