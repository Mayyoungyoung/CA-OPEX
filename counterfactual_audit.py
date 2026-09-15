"""Counterfactual replay audit for action-noise Walker datasets.

The collector stores both the action that was executed in MuJoCo and the clean
policy action before additive action noise.  This module first replays the
executed actions from the exact logged episode state.  Only if this logged
replay agrees with the stored rewards and next observations is the second
replay scientifically interpretable.  The second replay substitutes the
stored clean actions, open loop, and measures the resulting finite-horizon
return.

The clean-action replay is a trajectory-level mechanism label.  It is *not*
an estimate of the behavior policy's expected return: actions are not
recomputed at the counterfactual states and only one deterministic MuJoCo
rollout is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


REQUIRED_FIELDS: Tuple[str, ...] = (
    "observations",
    "next_observations",
    "actions",
    "clean_policy_actions",
    "rewards",
    "terminals",
    "timeouts",
    "infos/qpos",
    "infos/qvel",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def episode_slices(
    terminals: np.ndarray, timeouts: np.ndarray
) -> List[Tuple[int, int]]:
    """Split a flat replay buffer at ``terminal | timeout`` boundaries.

    A final segment without a boundary is retained.  ``collect_stochastic.py``
    marks such a segment as a collector truncation, but retaining it also makes
    this reader robust to other flat HDF5 writers.
    """

    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    timeouts = np.asarray(timeouts, dtype=bool).reshape(-1)
    if terminals.shape != timeouts.shape:
        raise ValueError("terminals and timeouts must have the same shape")
    if terminals.size == 0:
        return []
    ends = (np.flatnonzero(terminals | timeouts) + 1).tolist()
    if not ends or ends[-1] != terminals.size:
        ends.append(int(terminals.size))
    starts = [0, *ends[:-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _hdf5_scalar(handle: h5py.File, name: str) -> Optional[Any]:
    if name not in handle:
        return None
    value = handle[name][()]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collection_metadata(handle: h5py.File) -> Dict[str, Any]:
    """Read scalar collection metadata without copying arbitrary HDF5 arrays."""

    path = "metadata/collection"
    if path not in handle or not isinstance(handle[path], h5py.Group):
        return {}
    metadata: Dict[str, Any] = {}
    group = handle[path]
    for name, node in group.items():
        if not isinstance(node, h5py.Dataset) or node.shape not in ((), (1,)):
            continue
        value = node[()]
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(()).item()
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8")
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[name] = value
    return metadata


def _validate_dataset(handle: h5py.File) -> int:
    missing = [name for name in REQUIRED_FIELDS if name not in handle]
    if missing:
        raise KeyError(f"dataset is missing required fields: {', '.join(missing)}")
    count = int(handle["actions"].shape[0])
    if count < 1:
        raise ValueError("dataset contains no transitions")
    for name in REQUIRED_FIELDS:
        if handle[name].ndim < 1:
            raise ValueError(f"{name} must have a leading transition dimension")
        if int(handle[name].shape[0]) != count:
            raise ValueError(
                f"{name} has {handle[name].shape[0]} rows, expected {count}"
            )
    if handle["actions"].shape != handle["clean_policy_actions"].shape:
        raise ValueError("actions and clean_policy_actions must have identical shape")
    if handle["observations"].shape != handle["next_observations"].shape:
        raise ValueError("observations and next_observations must have identical shape")
    return count


def _set_episode_state(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Set a Gymnasium MuJoCo environment to a logged pre-action state."""

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "set_state"):
        raise TypeError("environment.unwrapped does not expose set_state(qpos, qvel)")
    unwrapped.set_state(
        np.asarray(qpos, dtype=np.float64).copy(),
        np.asarray(qvel, dtype=np.float64).copy(),
    )
    if not hasattr(unwrapped, "_get_obs"):
        raise TypeError("environment.unwrapped does not expose _get_obs()")
    return np.asarray(unwrapped._get_obs(), dtype=np.float64).reshape(-1)


def _error_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "max": float("nan")}
    return {"mean": float(array.mean()), "max": float(array.max())}


def _replay_logged_actions(
    env: Any,
    *,
    reset_seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    initial_observation: np.ndarray,
    actions: np.ndarray,
    expected_rewards: np.ndarray,
    expected_next_observations: np.ndarray,
    expected_terminals: np.ndarray,
    expected_timeouts: np.ndarray,
    collector_truncations: np.ndarray,
    reward_atol: float,
    observation_atol: float,
) -> Dict[str, Any]:
    env.reset(seed=reset_seed)
    replay_initial = _set_episode_state(env, qpos, qvel)
    expected_initial = np.asarray(initial_observation, dtype=np.float64).reshape(-1)
    if replay_initial.shape != expected_initial.shape:
        raise ValueError(
            "environment observation shape does not match stored observations: "
            f"{replay_initial.shape} versus {expected_initial.shape}"
        )
    initial_error = float(np.max(np.abs(replay_initial - expected_initial)))

    reward_errors: List[float] = []
    next_max_errors: List[float] = []
    next_squared_errors: List[float] = []
    replay_rewards: List[float] = []
    done_mismatches = 0
    unexpected_early_done = False
    steps = 0
    for offset, action in enumerate(actions):
        following, reward, terminated, truncated, _ = env.step(action)
        following = np.asarray(following, dtype=np.float64).reshape(-1)
        expected_following = np.asarray(
            expected_next_observations[offset], dtype=np.float64
        ).reshape(-1)
        if following.shape != expected_following.shape:
            raise ValueError(
                "environment next-observation shape does not match dataset: "
                f"{following.shape} versus {expected_following.shape}"
            )
        delta = following - expected_following
        next_max_errors.append(float(np.max(np.abs(delta))))
        next_squared_errors.extend(np.square(delta).tolist())
        reward_errors.append(abs(float(reward) - float(expected_rewards[offset])))
        replay_rewards.append(float(reward))
        steps += 1

        actual_done = bool(terminated or truncated)
        expected_done = bool(
            expected_terminals[offset] or expected_timeouts[offset]
        )
        artificial_boundary = bool(collector_truncations[offset])
        # A collector-created final boundary is not an environment event.
        if actual_done != expected_done and not artificial_boundary:
            done_mismatches += 1
        if actual_done and offset + 1 < len(actions):
            unexpected_early_done = True
            break

    reward_stats = _error_summary(reward_errors)
    next_stats = _error_summary(next_max_errors)
    next_rmse = (
        float(np.sqrt(np.mean(next_squared_errors)))
        if next_squared_errors
        else float("nan")
    )
    missing_steps = int(len(actions) - steps)
    passed = bool(
        missing_steps == 0
        and not unexpected_early_done
        and done_mismatches == 0
        and initial_error <= observation_atol
        and reward_stats["max"] <= reward_atol
        and next_stats["max"] <= observation_atol
    )
    return {
        "return": float(np.sum(replay_rewards, dtype=np.float64)),
        "steps": steps,
        "missing_steps": missing_steps,
        "unexpected_early_done": unexpected_early_done,
        "done_mismatch_count": done_mismatches,
        "initial_observation_max_abs_error": initial_error,
        "reward_mae": reward_stats["mean"],
        "reward_max_abs_error": reward_stats["max"],
        "next_observation_rmse": next_rmse,
        "next_observation_mean_max_abs_error": next_stats["mean"],
        "next_observation_max_abs_error": next_stats["max"],
        "reproduction_pass": passed,
    }


def _replay_clean_actions(
    env: Any,
    *,
    reset_seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actions: np.ndarray,
) -> Dict[str, Any]:
    env.reset(seed=reset_seed)
    _set_episode_state(env, qpos, qvel)
    rewards: List[float] = []
    terminated = False
    truncated = False
    for action in actions:
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return {
        "return": float(np.sum(rewards, dtype=np.float64)),
        "steps": len(rewards),
        "ended_early": bool(len(rewards) < len(actions)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _default_env_factory(environment: str) -> Any:
    import gymnasium as gym

    return gym.make(environment)


def audit_dataset(
    dataset: Path,
    output: Path,
    *,
    environment: Optional[str] = None,
    max_episodes: Optional[int] = None,
    reset_seed: int = 0,
    reward_atol: float = 1e-5,
    observation_atol: float = 1e-5,
    overwrite: bool = False,
    env_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Audit logged and clean-action replays and atomically write JSON results."""

    dataset = Path(dataset)
    output = Path(output)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --overwrite explicitly"
        )
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive when provided")
    if reward_atol < 0 or observation_atol < 0:
        raise ValueError("replay tolerances must be non-negative")

    dataset_hash = sha256_file(dataset)
    started = time.perf_counter()
    factory = env_factory or _default_env_factory
    records: List[Dict[str, Any]] = []

    with h5py.File(dataset, "r") as handle:
        transition_count = _validate_dataset(handle)
        collection_metadata = _collection_metadata(handle)
        terminals = np.asarray(handle["terminals"], dtype=bool).reshape(-1)
        timeouts = np.asarray(handle["timeouts"], dtype=bool).reshape(-1)
        slices = episode_slices(terminals, timeouts)
        if max_episodes is not None:
            slices = slices[:max_episodes]

        stored_environment = _hdf5_scalar(
            handle, "metadata/collection/environment"
        )
        selected_environment = environment or stored_environment or "Walker2d-v4"
        if not isinstance(selected_environment, str):
            selected_environment = str(selected_environment)
        if "Walker2d" not in selected_environment:
            raise ValueError(
                "counterfactual audit is scoped to Walker2d datasets; got "
                f"{selected_environment!r}"
            )

        if "collector_truncations" in handle:
            collector_truncations_all = np.asarray(
                handle["collector_truncations"], dtype=bool
            ).reshape(-1)
            if collector_truncations_all.shape != terminals.shape:
                raise ValueError(
                    "collector_truncations must match the transition count"
                )
        else:
            collector_truncations_all = np.zeros_like(terminals)

        env = factory(selected_environment)
        try:
            for episode_id, (start, end) in enumerate(slices):
                selection = slice(start, end)
                actions = np.asarray(handle["actions"][selection], dtype=np.float64)
                clean_actions = np.asarray(
                    handle["clean_policy_actions"][selection], dtype=np.float64
                )
                rewards = np.asarray(
                    handle["rewards"][selection], dtype=np.float64
                ).reshape(-1)
                next_observations = np.asarray(
                    handle["next_observations"][selection], dtype=np.float64
                )
                episode_terminals = terminals[selection]
                episode_timeouts = timeouts[selection]
                episode_collector_truncations = collector_truncations_all[selection]
                initial_observation = np.asarray(
                    handle["observations"][start], dtype=np.float64
                )
                initial_qpos = np.asarray(
                    handle["infos/qpos"][start], dtype=np.float64
                )
                initial_qvel = np.asarray(
                    handle["infos/qvel"][start], dtype=np.float64
                )

                logged = _replay_logged_actions(
                    env,
                    reset_seed=reset_seed + episode_id,
                    qpos=initial_qpos,
                    qvel=initial_qvel,
                    initial_observation=initial_observation,
                    actions=actions,
                    expected_rewards=rewards,
                    expected_next_observations=next_observations,
                    expected_terminals=episode_terminals,
                    expected_timeouts=episode_timeouts,
                    collector_truncations=episode_collector_truncations,
                    reward_atol=reward_atol,
                    observation_atol=observation_atol,
                )
                clean = _replay_clean_actions(
                    env,
                    reset_seed=reset_seed + episode_id,
                    qpos=initial_qpos,
                    qvel=initial_qvel,
                    actions=clean_actions,
                )
                raw_return = float(np.sum(rewards, dtype=np.float64))
                records.append(
                    {
                        "episode_id": episode_id,
                        "start_row": start,
                        "end_row_exclusive": end,
                        "length": end - start,
                        "raw_noisy_return": raw_return,
                        "logged_action_replay_return": logged["return"],
                        "logged_return_abs_error": abs(logged["return"] - raw_return),
                        "clean_action_replay_return": clean["return"],
                        "clean_minus_raw_return": clean["return"] - raw_return,
                        "logged_replay": logged,
                        "clean_action_replay": clean,
                    }
                )
        finally:
            env.close()

    elapsed = time.perf_counter() - started
    raw_returns = np.asarray(
        [record["raw_noisy_return"] for record in records], dtype=np.float64
    )
    clean_returns = np.asarray(
        [record["clean_action_replay_return"] for record in records],
        dtype=np.float64,
    )
    audited_transition_count = int(sum(record["length"] for record in records))
    result: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "logged-action validation then clean-action open-loop replay",
        "interpretation": (
            "clean_action_replay_return is a deterministic open-loop mechanism "
            "label, not the behavior policy's expected return"
        ),
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_bytes": dataset.stat().st_size,
        "environment": selected_environment,
        "dataset_transition_count": transition_count,
        "dataset_episode_count": len(episode_slices(terminals, timeouts)),
        "audited_episode_count": len(records),
        "audited_episode_fraction": len(records)
        / len(episode_slices(terminals, timeouts)),
        "audited_transition_count": audited_transition_count,
        "audited_transition_fraction": audited_transition_count / transition_count,
        "collection_metadata": collection_metadata,
        "max_episodes": max_episodes,
        "reset_seed": reset_seed,
        "reward_atol": reward_atol,
        "observation_atol": observation_atol,
        "logged_reproduction_pass_all": all(
            record["logged_replay"]["reproduction_pass"] for record in records
        ),
        "logged_reproduction_failure_count": sum(
            not record["logged_replay"]["reproduction_pass"] for record in records
        ),
        "worst_logged_reward_abs_error": max(
            record["logged_replay"]["reward_max_abs_error"] for record in records
        ),
        "worst_logged_next_observation_abs_error": max(
            record["logged_replay"]["next_observation_max_abs_error"]
            for record in records
        ),
        "raw_noisy_return_mean": float(raw_returns.mean()),
        "clean_action_replay_return_mean": float(clean_returns.mean()),
        "clean_minus_raw_return_mean": float((clean_returns - raw_returns).mean()),
        "clean_replay_early_end_count": sum(
            record["clean_action_replay"]["ended_early"] for record in records
        ),
        "wall_seconds": elapsed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "episodes": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a collected Walker2d dataset by replaying logged actions, "
            "then replay its saved clean policy actions open loop."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--env",
        default=None,
        help="environment id; defaults to collection metadata or Walker2d-v4",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--reset-seed", type=int, default=0)
    parser.add_argument("--reward-atol", type=float, default=1e-5)
    parser.add_argument("--observation-atol", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero after saving output if any logged replay fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_dataset(
        args.dataset,
        args.output,
        environment=args.env,
        max_episodes=args.max_episodes,
        reset_seed=args.reset_seed,
        reward_atol=args.reward_atol,
        observation_atol=args.observation_atol,
        overwrite=args.overwrite,
    )
    summary = {key: value for key, value in result.items() if key != "episodes"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict and not result["logged_reproduction_pass_all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
