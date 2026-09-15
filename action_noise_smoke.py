"""Reproduce the action-noise mechanism used in HUBL Appendix D.6.3.

This is a clean-room smoke test, not a copy of the unlicensed supplementary
source.  It deliberately separates the environment, policy-action, and noise
random-number streams and records the *applied* action after clipping.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


class UniformActionNoise(gym.Wrapper):
    """Apply ``clip(a + beta * U[-1, 1])`` before every environment step."""

    def __init__(self, env: gym.Env, beta: float, noise_seed: int) -> None:
        super().__init__(env)
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("UniformActionNoise requires a Box action space")
        self.beta = float(beta)
        self.noise_seed = int(noise_seed)
        self.noise_rng = np.random.default_rng(self.noise_seed)
        self.last_commanded_action: Optional[np.ndarray] = None
        self.last_noise: Optional[np.ndarray] = None
        self.last_applied_action: Optional[np.ndarray] = None

    def step(self, action):
        commanded = np.asarray(action, dtype=self.action_space.dtype)
        noise = self.noise_rng.uniform(-1.0, 1.0, size=commanded.shape)
        applied = np.clip(
            commanded + self.beta * noise,
            self.action_space.low,
            self.action_space.high,
        ).astype(self.action_space.dtype, copy=False)
        self.last_commanded_action = commanded.copy()
        self.last_noise = np.asarray(noise, dtype=np.float64)
        self.last_applied_action = applied.copy()
        return self.env.step(applied)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    base_env = gym.make(args.env)
    env = UniformActionNoise(base_env, beta=args.beta, noise_seed=args.noise_seed)
    policy_rng = np.random.default_rng(args.policy_seed)
    observation, _ = env.reset(seed=args.env_seed)
    del observation

    all_noise = []
    commanded = []
    applied = []
    episode_returns = []
    current_return = 0.0
    started = time.perf_counter()
    for step in range(args.steps):
        # Sampling explicitly from a private RNG avoids action_space's hidden RNG.
        action = policy_rng.uniform(
            env.action_space.low, env.action_space.high
        ).astype(env.action_space.dtype)
        _, reward, terminated, truncated, _ = env.step(action)
        current_return += float(reward)
        all_noise.append(env.last_noise.copy())
        commanded.append(env.last_commanded_action.copy())
        applied.append(env.last_applied_action.copy())
        if terminated or truncated:
            episode_returns.append(current_return)
            current_return = 0.0
            env.reset(seed=args.env_seed + step + 1)
    elapsed = time.perf_counter() - started
    env.close()

    noise_array = np.stack(all_noise)
    commanded_array = np.stack(commanded)
    applied_array = np.stack(applied)
    unclipped = commanded_array + args.beta * noise_array
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    clip_fraction = float(((unclipped < low) | (unclipped > high)).mean())
    try:
        import mujoco

        mujoco_version = mujoco.__version__
    except Exception:
        mujoco_version = None

    return {
        "schema_version": 1,
        "env": args.env,
        "beta": args.beta,
        "steps": args.steps,
        "env_seed": args.env_seed,
        "noise_seed": args.noise_seed,
        "policy_seed": args.policy_seed,
        "episodes_completed": len(episode_returns),
        "episode_return_mean": (
            float(np.mean(episode_returns)) if episode_returns else None
        ),
        "episode_return_std": (
            float(np.std(episode_returns)) if episode_returns else None
        ),
        "episode_return_min": (
            float(np.min(episode_returns)) if episode_returns else None
        ),
        "episode_return_max": (
            float(np.max(episode_returns)) if episode_returns else None
        ),
        "noise_mean": float(noise_array.mean()),
        "noise_std": float(noise_array.std()),
        "noise_min": float(noise_array.min()),
        "noise_max": float(noise_array.max()),
        "action_clip_fraction": clip_fraction,
        "wall_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "mujoco": mujoco_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker2d-v4")
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
