"""CLI for a trajectory-level, strictly out-of-fold reward-noise audit."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import h5py
import numpy as np

from mechanism_core import (
    audit_metrics,
    discounted_return,
    fit_crossfit_heteroscedastic_mlp,
    inject_reward_noise,
    load_d4rl_hdf5,
    make_trajectory_folds,
    reconstruct_trajectories,
    save_json,
    sha256_file,
    trajectory_features,
)


def create_synthetic_hdf5(
    path: Path, *, n_trajectories: int = 160, min_length: int = 6,
    max_length: int = 12, seed: int = 17
) -> None:
    """Create a small reward-predictable transition file for integration tests."""

    if n_trajectories < 4 or min_length < 2 or max_length < min_length:
        raise ValueError("invalid synthetic dataset dimensions")
    rng = np.random.default_rng(seed)
    observations, actions, rewards, terminals, timeouts = [], [], [], [], []
    for trajectory_id in range(n_trajectories):
        length = int(rng.integers(min_length, max_length + 1))
        quality = float(rng.normal())
        phase = float(rng.uniform(-np.pi, np.pi))
        for step in range(length):
            progress = step / max(length - 1, 1)
            observations.append(
                [
                    quality + 0.04 * rng.normal(),
                    progress,
                    np.sin(phase + progress),
                    length / max_length,
                ]
            )
            actions.append([np.tanh(quality) + 0.05 * rng.normal(), progress - 0.5])
            rewards.append(quality + 0.2 * np.cos(phase + progress) + 0.03 * rng.normal())
            terminals.append(step == length - 1 and trajectory_id % 2 == 0)
            timeouts.append(step == length - 1 and trajectory_id % 2 == 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations", data=np.asarray(observations, dtype=np.float32))
        handle.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
        handle.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        handle.create_dataset("terminals", data=np.asarray(terminals, dtype=np.bool_))
        handle.create_dataset("timeouts", data=np.asarray(timeouts, dtype=np.bool_))


def run_audit(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_d4rl_hdf5(dataset_path)
    noisy_rewards, realized_noise = inject_reward_noise(
        arrays["rewards"], args.noise_std, args.noise_seed, args.noise_distribution
    )
    trajectories = reconstruct_trajectories(arrays, noisy_rewards)
    features = trajectory_features(trajectories)
    clean_returns = np.asarray(
        [discounted_return(t.clean_rewards, args.gamma) for t in trajectories], dtype=np.float32
    )
    noisy_returns = np.asarray(
        [discounted_return(t.noisy_rewards, args.gamma) for t in trajectories], dtype=np.float32
    )
    target = noisy_returns if args.target_source == "noisy" else clean_returns
    folds = make_trajectory_folds(len(trajectories), args.folds, args.fold_seed)
    means, stds, provenance = fit_crossfit_heteroscedastic_mlp(
        features,
        target,
        folds,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_seed=args.model_seed,
        device=args.device,
        calibration_fraction=args.calibration_fraction,
        early_stopping_patience=args.early_stopping_patience,
    )
    lcb = means - args.lcb_kappa * stds
    metrics = audit_metrics(clean_returns, noisy_returns, means, stds, lcb)
    np.savez_compressed(
        output_dir / "audit_arrays.npz",
        clean_rewards=arrays["rewards"],
        injected_noise=realized_noise,
        noisy_rewards=noisy_rewards,
        trajectory_start=np.asarray([t.start for t in trajectories], dtype=np.int64),
        trajectory_stop=np.asarray([t.stop for t in trajectories], dtype=np.int64),
        trajectory_length=np.asarray([t.length for t in trajectories], dtype=np.int64),
        trajectory_features=features,
        clean_returns=clean_returns,
        raw_noisy_returns=noisy_returns,
        fold_ids=folds,
        crossfit_mean=means,
        crossfit_std=stds,
        lcb=lcb,
    )
    metadata = {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "transition_count": int(arrays["rewards"].size),
        "trajectory_count": len(trajectories),
        "observation_dim": int(arrays["observations"].shape[1]),
        "action_dim": int(arrays["actions"].shape[1]),
        "gamma": args.gamma,
        "noise": {
            "distribution": args.noise_distribution,
            "std_per_transition": args.noise_std,
            "seed": args.noise_seed,
            "realized_mean": float(realized_noise.mean()),
            "realized_std": float(realized_noise.std()),
        },
        "crossfit": {
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "model_seed": args.model_seed,
            "target_source": args.target_source,
            "feature_definition": "initial/final/mean/std observation; mean/std action; log1p length",
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "calibration_fraction": args.calibration_fraction,
            "early_stopping_patience": args.early_stopping_patience,
            "fold_provenance": provenance,
        },
        "lcb_kappa": args.lcb_kappa,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "implementation_sha256": {
            "mechanism_audit.py": sha256_file(Path(__file__).resolve()),
            "mechanism_core.py": sha256_file(
                Path(__file__).with_name("mechanism_core.py").resolve()
            ),
        },
        "wall_time_seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    try:
        import torch

        metadata["versions"]["torch"] = torch.__version__
        metadata["versions"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            metadata["versions"]["cuda_runtime"] = torch.version.cuda
            metadata["versions"]["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    save_json(output_dir / "audit_metrics.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("make-synthetic", help="write a synthetic D4RL-like HDF5")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--trajectories", type=int, default=160)
    synthetic.add_argument("--min-length", type=int, default=6)
    synthetic.add_argument("--max-length", type=int, default=12)
    synthetic.add_argument("--seed", type=int, default=17)

    audit = subparsers.add_parser("audit", help="run noise injection and OOF audit")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--gamma", type=float, default=0.99)
    audit.add_argument("--noise-std", type=float, default=3.0)
    audit.add_argument("--noise-distribution", choices=("normal", "uniform"), default="normal")
    audit.add_argument("--noise-seed", type=int, default=20260914)
    audit.add_argument("--folds", type=int, default=5)
    audit.add_argument("--fold-seed", type=int, default=31)
    audit.add_argument("--model-seed", type=int, default=47)
    audit.add_argument("--target-source", choices=("noisy", "clean"), default="noisy")
    audit.add_argument("--hidden-dim", type=int, default=64)
    audit.add_argument("--epochs", type=int, default=300)
    audit.add_argument("--batch-size", type=int, default=64)
    audit.add_argument("--learning-rate", type=float, default=3e-3)
    audit.add_argument("--weight-decay", type=float, default=1e-4)
    audit.add_argument("--lcb-kappa", type=float, default=0.5)
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--calibration-fraction", type=float, default=0.2)
    audit.add_argument("--early-stopping-patience", type=int, default=40)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic":
        create_synthetic_hdf5(
            Path(args.output),
            n_trajectories=args.trajectories,
            min_length=args.min_length,
            max_length=args.max_length,
            seed=args.seed,
        )
        print(json.dumps({"created": str(Path(args.output).resolve())}))
        return 0
    metadata = run_audit(args)
    print(json.dumps(metadata["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
