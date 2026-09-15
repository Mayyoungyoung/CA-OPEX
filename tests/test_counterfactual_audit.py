from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from counterfactual_audit import audit_dataset, episode_slices, sha256_file


class _FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(1, dtype=np.float64)
        self.qvel = np.zeros(1, dtype=np.float64)


class _FakeWalker:
    """Tiny deterministic stateful env with the MuJoCo state interface."""

    def __init__(self) -> None:
        self.data = _FakeData()
        self.steps = 0
        self.max_steps = 1_000

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        del seed
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.steps = 0
        self.max_steps = 2 if float(self.data.qpos[0]) > 0 else 3

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float64)

    def step(self, action):
        acceleration = float(np.asarray(action).reshape(-1)[0])
        self.data.qvel[:] += acceleration
        self.data.qpos[:] += self.data.qvel
        reward = float(self.data.qpos[0] - 0.1 * acceleration**2)
        self.steps += 1
        terminated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, False, {}

    def close(self):
        pass


def _rollout(initial_qpos, initial_qvel, actions):
    env = _FakeWalker()
    env.reset()
    env.set_state(np.array([initial_qpos]), np.array([initial_qvel]))
    observations = []
    next_observations = []
    rewards = []
    qpos = []
    qvel = []
    for action in actions:
        observations.append(env._get_obs())
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        following, reward, *_ = env.step(np.array([action]))
        next_observations.append(following)
        rewards.append(reward)
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "qpos": np.asarray(qpos, dtype=np.float64),
        "qvel": np.asarray(qvel, dtype=np.float64),
    }


def _write_dataset(path: Path) -> None:
    noisy_episode_actions = ([1.0, -0.5], [0.25, 0.5, -0.25])
    clean_episode_actions = ([0.5, -0.5], [0.0, 0.25, -0.25])
    initial_states = ((2.0, 0.0), (-1.0, 0.5))
    chunks = []
    for state, actions in zip(initial_states, noisy_episode_actions):
        chunks.append(_rollout(*state, actions))

    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observations", data=np.concatenate([x["observations"] for x in chunks])
        )
        handle.create_dataset(
            "next_observations",
            data=np.concatenate([x["next_observations"] for x in chunks]),
        )
        handle.create_dataset(
            "actions",
            data=np.concatenate(
                [np.asarray(x, dtype=np.float32)[:, None] for x in noisy_episode_actions]
            ),
        )
        handle.create_dataset(
            "clean_policy_actions",
            data=np.concatenate(
                [np.asarray(x, dtype=np.float32)[:, None] for x in clean_episode_actions]
            ),
        )
        handle.create_dataset(
            "rewards", data=np.concatenate([x["rewards"] for x in chunks])
        )
        handle.create_dataset(
            "terminals", data=np.array([False, True, False, False, True])
        )
        handle.create_dataset(
            "timeouts", data=np.array([False, False, False, False, False])
        )
        handle.create_dataset(
            "collector_truncations", data=np.array([False] * 5)
        )
        infos = handle.require_group("infos")
        infos.create_dataset("qpos", data=np.concatenate([x["qpos"] for x in chunks]))
        infos.create_dataset("qvel", data=np.concatenate([x["qvel"] for x in chunks]))
        collection = handle.require_group("metadata/collection")
        collection.create_dataset("environment", data=np.bytes_("Walker2d-v4"))


def test_episode_slices_keeps_final_incomplete_segment():
    terminals = np.array([False, True, False, False])
    timeouts = np.array([False, False, False, False])
    assert episode_slices(terminals, timeouts) == [(0, 2), (2, 4)]


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"counterfactual-replay")
    assert sha256_file(path) == hashlib.sha256(b"counterfactual-replay").hexdigest()


def test_logged_replay_and_clean_counterfactual(tmp_path: Path):
    dataset = tmp_path / "flat.hdf5"
    output = tmp_path / "audit.json"
    _write_dataset(dataset)

    result = audit_dataset(
        dataset,
        output,
        max_episodes=1,
        reward_atol=1e-5,
        observation_atol=1e-5,
        env_factory=lambda _: _FakeWalker(),
    )

    assert output.is_file()
    assert json.loads(output.read_text())["dataset_sha256"] == sha256_file(dataset)
    assert result["audited_episode_count"] == 1
    assert result["audited_episode_fraction"] == pytest.approx(0.5)
    assert result["audited_transition_count"] == 2
    assert result["audited_transition_fraction"] == pytest.approx(0.4)
    assert result["dataset_episode_count"] == 2
    assert result["logged_reproduction_pass_all"] is True
    episode = result["episodes"][0]
    assert episode["length"] == 2
    assert episode["logged_return_abs_error"] == pytest.approx(0.0, abs=1e-6)
    assert episode["logged_replay"]["next_observation_max_abs_error"] < 1e-6
    assert episode["clean_action_replay_return"] != pytest.approx(
        episode["raw_noisy_return"]
    )


def test_refuses_overwrite_and_missing_fields(tmp_path: Path):
    dataset = tmp_path / "flat.hdf5"
    output = tmp_path / "audit.json"
    _write_dataset(dataset)
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        audit_dataset(dataset, output, env_factory=lambda _: _FakeWalker())

    broken = tmp_path / "broken.hdf5"
    with h5py.File(broken, "w") as handle:
        handle.create_dataset("actions", data=np.zeros((1, 1)))
    with pytest.raises(KeyError):
        audit_dataset(
            broken,
            tmp_path / "broken.json",
            env_factory=lambda _: _FakeWalker(),
        )
