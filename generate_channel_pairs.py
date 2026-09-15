"""Generate state-free paired command/execution data for channel calibration.

The output intentionally contains no observations, rewards, transitions, or
policy labels beyond the sampled command itself.  Each execution is produced
by the known calibration channel

    a_exec = clip(u_cmd + eps, action_low, action_high),
    eps_d iid Uniform[-beta, beta].
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


PAIR_SCHEMA = "paired_uniform_clip_channel_v1"


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.asarray(values, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _episode_keys(handle: h5py.File) -> List[str]:
    keys = [key for key in handle.keys() if key.startswith("episode_")]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def load_clean_policy_actions(path: Path) -> Tuple[np.ndarray, str]:
    """Load only the clean command channel from flat or episodic HDF5."""

    pieces: List[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        episode_keys = _episode_keys(handle)
        if episode_keys:
            for key in episode_keys:
                group = handle[key]
                if "clean_policy_actions" not in group:
                    raise KeyError(
                        f"{key} is missing required HDF5 field clean_policy_actions"
                    )
                commands = np.asarray(
                    group["clean_policy_actions"], dtype=np.float32
                )
                if commands.ndim < 2:
                    raise ValueError(
                        f"{key}/clean_policy_actions must have shape [N, action...]"
                    )
                pieces.append(commands.reshape(commands.shape[0], -1))
            schema = "episodic"
        else:
            if "clean_policy_actions" not in handle:
                raise KeyError(
                    "dataset is missing required HDF5 field clean_policy_actions"
                )
            commands = np.asarray(handle["clean_policy_actions"], dtype=np.float32)
            if commands.ndim < 2:
                raise ValueError(
                    "clean_policy_actions must have shape [N, action...]"
                )
            pieces.append(commands.reshape(commands.shape[0], -1))
            schema = "flat"
    if not pieces:
        raise ValueError("dataset contains no clean policy actions")
    commands = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if commands.shape[0] == 0 or commands.shape[1] == 0:
        raise ValueError("clean policy action array is empty")
    if not np.all(np.isfinite(commands)):
        raise ValueError("clean_policy_actions contains non-finite values")
    return commands, schema


def generate_pairs(
    source_path: Path,
    *,
    pair_count: int,
    beta: float,
    seed: int,
    action_low: float,
    action_high: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Sample commands without replacement and apply Uniform+clip noise."""

    source_path = source_path.resolve()
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not (
        np.isfinite(action_low)
        and np.isfinite(action_high)
        and action_low < action_high
    ):
        raise ValueError("action bounds must be finite and strictly ordered")
    all_commands, source_schema = load_clean_policy_actions(source_path)
    if pair_count > all_commands.shape[0]:
        raise ValueError(
            f"requested {pair_count} pairs from only {all_commands.shape[0]} commands"
        )
    bound_tolerance = 1e-6
    if np.any(all_commands < action_low - bound_tolerance) or np.any(
        all_commands > action_high + bound_tolerance
    ):
        raise ValueError("source commands lie outside the declared action bounds")

    root_seed = np.random.SeedSequence(int(seed))
    selection_seed, noise_seed = root_seed.spawn(2)
    selection_rng = np.random.default_rng(selection_seed)
    noise_rng = np.random.default_rng(noise_seed)
    indices = selection_rng.choice(
        all_commands.shape[0], size=pair_count, replace=False
    )
    commands = all_commands[indices].astype(np.float32, copy=True)
    perturbations = noise_rng.uniform(
        -beta, beta, size=commands.shape
    ).astype(np.float32)
    executed = np.clip(
        commands + perturbations, action_low, action_high
    ).astype(np.float32, copy=False)

    provenance: Dict[str, object] = {
        "schema": PAIR_SCHEMA,
        "source_dataset_path": str(source_path),
        "source_dataset_sha256": sha256_file(source_path),
        "source_hdf5_schema": source_schema,
        "source_command_count": int(all_commands.shape[0]),
        "selection": "uniform_without_replacement",
        "pair_count": int(pair_count),
        "action_dim": int(commands.shape[1]),
        "action_low": float(action_low),
        "action_high": float(action_high),
        "channel": "iid_uniform_additive_then_clip",
        "beta": float(beta),
        "root_seed": int(seed),
        "bit_generator": "PCG64",
        "selection_spawn_key": list(selection_seed.spawn_key),
        "noise_spawn_key": list(noise_seed.spawn_key),
        "commands_sha256": array_sha256(commands),
        "executed_sha256": array_sha256(executed),
        "contains_state_or_transition_fields": False,
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    return commands, executed, provenance


def save_pair_archive(
    path: Path,
    commands: np.ndarray,
    executed: np.ndarray,
    provenance: Dict[str, object],
) -> None:
    """Atomically save the deliberately minimal, pickle-free NPZ schema."""

    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            commands=np.asarray(commands, dtype=np.float32),
            executed=np.asarray(executed, dtype=np.float32),
            provenance_json=np.asarray(payload),
        )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source HDF5 dataset")
    parser.add_argument("--output", required=True, help="output .npz archive")
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and generate in memory without writing the archive",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    commands, executed, provenance = generate_pairs(
        Path(args.source),
        pair_count=args.pairs,
        beta=args.beta,
        seed=args.seed,
        action_low=args.action_low,
        action_high=args.action_high,
    )
    if not args.dry_run:
        save_pair_archive(output, commands, executed, provenance)
    summary = {
        "status": "dry_run" if args.dry_run else "complete",
        "output": None if args.dry_run else str(output),
        "pair_count": int(commands.shape[0]),
        "action_dim": int(commands.shape[1]),
        "commands_sha256": provenance["commands_sha256"],
        "executed_sha256": provenance["executed_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
