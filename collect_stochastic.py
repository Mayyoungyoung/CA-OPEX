"""Collect a stochastic Walker/Hopper dataset from embedded D4RL SAC weights.

The saved ``actions`` are the actions actually executed by the environment.
Clean policy actions and sampled perturbations are retained in separate audit
fields so the observation/action convention cannot be silently changed later.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import gymnasium as gym
import h5py
import numpy as np
import torch

try:
    from .behavior_policy import (
        D4RLTanhGaussianPolicy,
        audit_embedded_log_probs,
        sha256_file,
    )
except ImportError:  # Direct ``python collect_stochastic.py`` execution.
    from behavior_policy import (  # type: ignore
        D4RLTanhGaussianPolicy,
        audit_embedded_log_probs,
        sha256_file,
    )


def _metadata_scalar(group: h5py.Group, name: str, value) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=np.bytes_(value))
    else:
        group.create_dataset(name, data=value)


def _write_hdf5_atomic(
    output: Path, arrays: Dict[str, np.ndarray], metadata: Dict[str, object]
) -> None:
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w") as handle:
        for name, value in arrays.items():
            parent, _, leaf = name.rpartition("/")
            group = handle.require_group(parent) if parent else handle
            group.create_dataset(leaf, data=value, compression="lzf", chunks=True)
        metadata_group = handle.require_group("metadata/collection")
        for name, value in metadata.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                _metadata_scalar(metadata_group, name, value)
        handle.flush()
    os.replace(temporary, output)


def collect(args: argparse.Namespace) -> Dict[str, object]:
    if args.transitions < 1:
        raise ValueError("--transitions must be positive")
    if args.beta < 0:
        raise ValueError("--beta must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {args.output}; pass --overwrite explicitly"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(args.source)
    policy = D4RLTanhGaussianPolicy.from_hdf5(args.source, device=args.device)
    audit_samples = min(args.audit_samples, _transition_count(args.source))
    audit = audit_embedded_log_probs(
        args.source, policy, sample_count=audit_samples, seed=args.audit_seed
    )
    if not args.allow_policy_audit_failure and (
        audit.mean_absolute_error > args.max_logprob_mae
        or audit.pearson_correlation < args.min_logprob_correlation
    ):
        raise RuntimeError(
            "embedded policy audit failed: "
            f"MAE={audit.mean_absolute_error:.6g}, "
            f"correlation={audit.pearson_correlation:.6g}"
        )

    env = gym.make(args.env)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("collector requires a continuous Box action space")
    if env.observation_space.shape != (policy.observation_dim,):
        raise ValueError(
            f"environment observation shape {env.observation_space.shape} does "
            f"not match policy dimension {policy.observation_dim}"
        )
    if env.action_space.shape != (policy.action_dim,):
        raise ValueError(
            f"environment action shape {env.action_space.shape} does not match "
            f"policy dimension {policy.action_dim}"
        )

    count = args.transitions
    obs = np.empty((count, policy.observation_dim), dtype=np.float32)
    next_obs = np.empty_like(obs)
    actions = np.empty((count, policy.action_dim), dtype=np.float32)
    clean_actions = np.empty_like(actions)
    sampled_noise = np.empty_like(actions)
    applied_delta = np.empty_like(actions)
    rewards = np.empty(count, dtype=np.float32)
    terminals = np.zeros(count, dtype=np.bool_)
    timeouts = np.zeros(count, dtype=np.bool_)
    collector_truncations = np.zeros(count, dtype=np.bool_)
    clean_log_probs = np.empty(count, dtype=np.float32)

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "data"):
        env.close()
        raise TypeError("environment does not expose MuJoCo qpos/qvel data")
    qpos_dim = int(np.asarray(unwrapped.data.qpos).size)
    qvel_dim = int(np.asarray(unwrapped.data.qvel).size)
    qpos = np.empty((count, qpos_dim), dtype=np.float64)
    qvel = np.empty((count, qvel_dim), dtype=np.float64)

    observation, _ = env.reset(seed=args.env_seed)
    noise_rng = np.random.default_rng(args.noise_seed)
    generator_device = torch.device(args.device).type
    policy_generator = torch.Generator(device=generator_device)
    policy_generator.manual_seed(args.policy_seed)
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    episode_returns = []
    episode_lengths = []
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    clipped_values = 0
    started = time.perf_counter()

    for index in range(count):
        obs[index] = np.asarray(observation, dtype=np.float32)
        qpos[index] = np.asarray(unwrapped.data.qpos, dtype=np.float64).reshape(-1)
        qvel[index] = np.asarray(unwrapped.data.qvel, dtype=np.float64).reshape(-1)
        observation_tensor = torch.as_tensor(
            obs[index], dtype=torch.float32, device=args.device
        )
        with torch.no_grad():
            if args.policy_mode == "stochastic":
                clean_tensor, log_prob_tensor = policy.sample_action(
                    observation_tensor, policy_generator
                )
            else:
                clean_tensor = policy.deterministic_action(observation_tensor)
                log_prob_tensor = policy.log_prob(observation_tensor, clean_tensor)
        clean = clean_tensor.cpu().numpy().astype(np.float32, copy=False)
        log_prob = float(log_prob_tensor.cpu().item())
        epsilon = noise_rng.uniform(-1.0, 1.0, size=clean.shape).astype(np.float32)
        perturbation = np.float32(args.beta) * epsilon
        requested = clean + perturbation
        executed = np.clip(requested, action_low, action_high).astype(
            np.float32, copy=False
        )
        clipped_values += int(((requested < action_low) | (requested > action_high)).sum())

        following, reward, terminated, truncated, _ = env.step(executed)
        next_obs[index] = np.asarray(following, dtype=np.float32)
        actions[index] = executed
        clean_actions[index] = clean
        sampled_noise[index] = perturbation
        applied_delta[index] = executed - clean
        rewards[index] = np.float32(reward)
        terminals[index] = bool(terminated)
        timeouts[index] = bool(truncated)
        clean_log_probs[index] = np.float32(log_prob)
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_index += 1
            observation, _ = env.reset(seed=args.env_seed + episode_index)
            episode_return = 0.0
            episode_length = 0
        else:
            observation = following

        if args.progress_every and (index + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"collected {index + 1}/{count} transitions "
                f"({(index + 1) / elapsed:.1f} steps/s)",
                file=sys.stderr,
            )

    # Preserve the final incomplete segment as an explicit dataset truncation.
    if not (terminals[-1] or timeouts[-1]):
        timeouts[-1] = True
        collector_truncations[-1] = True
    elapsed = time.perf_counter() - started
    env.close()

    try:
        import mujoco

        mujoco_version = mujoco.__version__
    except Exception:
        mujoco_version = "unavailable"
    metadata: Dict[str, object] = {
        "schema_version": 1,
        "source_dataset_path": str(args.source.resolve()),
        "source_dataset_sha256": source_hash,
        "environment": args.env,
        "environment_seed": args.env_seed,
        "noise_seed": args.noise_seed,
        "policy_seed": args.policy_seed,
        "policy_mode": args.policy_mode,
        "beta": args.beta,
        "noise_distribution": "iid_uniform_minus1_plus1",
        "logged_action_convention": "executed_noisy_action",
        "transitions": count,
        "episodes_completed": len(episode_returns),
        "partial_episode_length": episode_length,
        "partial_episode_return": episode_return,
        "action_clip_fraction": clipped_values / actions.size,
        "wall_seconds": elapsed,
        "steps_per_second": count / elapsed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "mujoco_version": mujoco_version,
        "policy_nonlinearity": policy.nonlinearity,
        "policy_output_distribution": policy.output_distribution,
    }
    arrays = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "clean_policy_actions": clean_actions,
        "sampled_action_noise": sampled_noise,
        "applied_action_delta": applied_delta,
        "collector_truncations": collector_truncations,
        "infos/qpos": qpos,
        "infos/qvel": qvel,
        "infos/clean_action_log_probs": clean_log_probs,
    }
    _write_hdf5_atomic(args.output, arrays, metadata)
    output_hash = sha256_file(args.output)
    manifest: Dict[str, object] = {
        **metadata,
        "output_dataset_path": str(args.output.resolve()),
        "output_dataset_sha256": output_hash,
        "policy_logprob_audit": audit.as_dict(),
        "episode_return_mean": (
            float(np.mean(episode_returns)) if episode_returns else None
        ),
        "episode_return_std": (
            float(np.std(episode_returns)) if episode_returns else None
        ),
        "episode_length_mean": (
            float(np.mean(episode_lengths)) if episode_lengths else None
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".part")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _transition_count(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle["actions"].shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", default="Walker2d-v4")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--transitions", type=int, default=1_000)
    parser.add_argument(
        "--policy-mode", choices=("stochastic", "deterministic"), default="stochastic"
    )
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=2)
    parser.add_argument("--audit-seed", type=int, default=3)
    parser.add_argument("--audit-samples", type=int, default=10_000)
    parser.add_argument("--max-logprob-mae", type=float, default=0.01)
    parser.add_argument("--min-logprob-correlation", type=float, default=0.999)
    parser.add_argument("--allow-policy-audit-failure", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = collect(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
