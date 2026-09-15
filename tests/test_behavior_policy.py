from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from behavior_policy import D4RLTanhGaussianPolicy, audit_embedded_log_probs


def _write_policy_file(path: Path, nonlinearity: str = "relu") -> None:
    rng = np.random.default_rng(19)
    obs_dim, hidden_dim, action_dim = 3, 5, 2
    with h5py.File(path, "w") as handle:
        group = handle.require_group("metadata/policy")
        group["nonlinearity"] = np.bytes_(nonlinearity)
        group["output_distribution"] = np.bytes_("tanh_gaussian")
        group["fc0/weight"] = rng.normal(size=(hidden_dim, obs_dim)).astype(np.float32)
        group["fc0/bias"] = rng.normal(size=hidden_dim).astype(np.float32)
        group["fc1/weight"] = rng.normal(size=(hidden_dim, hidden_dim)).astype(
            np.float32
        )
        group["fc1/bias"] = rng.normal(size=hidden_dim).astype(np.float32)
        group["last_fc/weight"] = rng.normal(size=(action_dim, hidden_dim)).astype(
            np.float32
        )
        group["last_fc/bias"] = rng.normal(size=action_dim).astype(np.float32)
        group["last_fc_log_std/weight"] = np.zeros(
            (action_dim, hidden_dim), dtype=np.float32
        )
        group["last_fc_log_std/bias"] = np.full(action_dim, -0.7, dtype=np.float32)


def test_embedded_policy_log_probability_round_trip(tmp_path):
    path = tmp_path / "policy.hdf5"
    _write_policy_file(path)
    policy = D4RLTanhGaussianPolicy.from_hdf5(path)
    observations = torch.randn(128, policy.observation_dim)
    generator = torch.Generator(device="cpu").manual_seed(71)
    actions, log_probs = policy.sample_action(observations, generator)
    # Re-evaluating sampled actions through atanh should match except for tiny
    # numerical error near the tanh boundary.
    reconstructed = policy.log_prob(observations, actions)
    torch.testing.assert_close(reconstructed, log_probs, atol=2e-3, rtol=2e-3)

    with h5py.File(path, "a") as handle:
        handle["observations"] = observations.numpy()
        handle["actions"] = actions.numpy()
        handle["infos/action_log_probs"] = log_probs.numpy()
    audit = audit_embedded_log_probs(path, sample_count=128, seed=11)
    assert audit.mean_absolute_error < 2e-3
    assert audit.pearson_correlation > 0.99999


def test_loader_obeys_relu_metadata(tmp_path):
    path = tmp_path / "relu.hdf5"
    _write_policy_file(path, nonlinearity="relu")
    policy = D4RLTanhGaussianPolicy.from_hdf5(path)
    observation = torch.tensor([-1.0, 0.5, 2.0])
    mean, _ = policy(observation)
    first = torch.relu(
        torch.nn.functional.linear(
            observation,
            policy._buffer("fc0/weight"),
            policy._buffer("fc0/bias"),
        )
    )
    second = torch.relu(
        torch.nn.functional.linear(
            first,
            policy._buffer("fc1/weight"),
            policy._buffer("fc1/bias"),
        )
    )
    expected = torch.nn.functional.linear(
        second,
        policy._buffer("last_fc/weight"),
        policy._buffer("last_fc/bias"),
    )
    torch.testing.assert_close(mean, expected)


def test_unknown_nonlinearity_fails_closed(tmp_path):
    path = tmp_path / "unknown.hdf5"
    _write_policy_file(path, nonlinearity="mystery_activation")
    with pytest.raises(ValueError, match="unsupported policy nonlinearity"):
        D4RLTanhGaussianPolicy.from_hdf5(path)


def test_sampling_seed_is_reproducible(tmp_path):
    path = tmp_path / "seeded.hdf5"
    _write_policy_file(path)
    policy = D4RLTanhGaussianPolicy.from_hdf5(path)
    observations = torch.zeros(7, policy.observation_dim)
    first, _ = policy.sample_action(
        observations, torch.Generator(device="cpu").manual_seed(123)
    )
    second, _ = policy.sample_action(
        observations, torch.Generator(device="cpu").manual_seed(123)
    )
    torch.testing.assert_close(first, second)
