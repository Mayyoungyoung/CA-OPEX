import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = CODE_DIR / "run_ca_opex_walker_scaleup.sh"


def usable_bash():
    candidate = shutil.which("bash")
    if candidate is None:
        return None
    try:
        probe = subprocess.run(
            [candidate, "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return candidate if probe.returncode == 0 else None


def require_bash():
    candidate = usable_bash()
    if candidate is None:
        pytest.skip("a working POSIX bash is unavailable on this host")
    return candidate


def make_isolated_script(tmp_path):
    fake_research = tmp_path / "research"
    fake_python = tmp_path / "python"
    fake_dataset = tmp_path / "walker.hdf5"
    fake_calibration = (
        fake_research
        / "results/channel_calibration/beta125_n512_seed27001_calibration.json"
    )
    fake_python.write_bytes(b"not executed during dry-run\n")
    fake_dataset.write_bytes(b"frozen-test-dataset\n")
    fake_calibration.parent.mkdir(parents=True)
    fake_calibration.write_bytes(b"frozen-test-calibration\n")
    fake_code = fake_research / "code"
    fake_code.mkdir(parents=True)
    for filename in (
        "train_td3bc.py",
        "evaluate_channel_opex.py",
        "evaluate_td3bc.py",
        "td3bc_core.py",
        "evaluation_controls.py",
        "inverse_residual_core.py",
        "train_inverse_residual_adapter.py",
        "aggregate_ca_opex_walker_scaleup.py",
    ):
        (fake_code / filename).write_text("# test placeholder\n", encoding="utf-8")

    text = SCRIPT.read_text(encoding="utf-8")
    text = text.replace("/root/hubl_research_20260914", fake_research.as_posix())
    text = text.replace("/root/hubl_backup_env/bin/python", fake_python.as_posix())
    text = text.replace(
        "/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5",
        fake_dataset.as_posix(),
    )
    text = text.replace(
        "159a49faaa7786a8444369a1aef758b1a43f26eaaa56146358ace6f9c5b3882a",
        hashlib.sha256(fake_dataset.read_bytes()).hexdigest(),
    )
    text = text.replace(
        "b1769bdeb0eb99aa653ae5d98ef00e3f843e938fb62963821e0f2e348f904354",
        hashlib.sha256(fake_calibration.read_bytes()).hexdigest(),
    )
    isolated = tmp_path / "run_scaleup.sh"
    isolated.write_text(text, encoding="utf-8", newline="\n")
    isolated.chmod(0o755)
    return isolated


def invoke(script, *args, results_root=None):
    bash = require_bash()
    env = os.environ.copy()
    if results_root is not None:
        env["CA_OPEX_SCALEUP_RESULTS_ROOT"] = str(results_root)
    return subprocess.run(
        [bash, str(script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def test_script_contains_frozen_training_and_five_arm_contracts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert 'readonly dataset="/root/hubl_backup_data/action_noise_dev/walker2d_v4_beta1_seed1201.hdf5"' in text
    assert 'readonly calibration="${research_root}/results/channel_calibration/beta125_n512_seed27001_calibration.json"' in text
    assert "--variant hubl_constant" in text
    assert "--action-pairing executed_executed" in text
    assert 'readonly base_updates=25000' in text
    assert 'readonly heuristic_discount="0.391843318939209"' in text
    assert '0) die "training seed 0 is reserved for development' in text
    assert '1|10) die "training seed ${seed} is reserved for formal confirmation' in text
    assert 'readonly base_dir="${run_root}/base"' in text
    assert 'readonly controls_dir="${run_root}/controls"' in text
    assert "run_channel_arm complete inverse calibrated 0.1 8 2 0.25" in text
    assert "run_channel_arm calibrated_identity identity calibrated 0.3 8 2 2.0" in text
    assert "run_channel_arm nominal_k8t2 identity nominal_beta0 0.1 8 2 2.0" in text
    assert "run_channel_arm original_opex identity nominal_beta0 0.1 1 1 2.0" in text
    assert "run_inverse_only" in text
    assert "--command-transform uniform_mean_inverse" in text
    assert '"${code_dir}/aggregate_ca_opex_walker_scaleup.py"' in text
    assert '--output "${run_root}/aggregate.json"' in text
    assert "resume" not in text.lower().replace("no resume mode", "")


def test_seed_namespaces_are_checked_only_against_their_reserved_blocks():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'reject_reserved_block environment "${env_seed}" 39300 79300' in text
    assert 'reject_reserved_block actuator "${noise_seed}" 49300 89300' in text
    assert 'reject_reserved_block gradient "${grad_seed}" 69300 99300' in text
    assert "env_seed" not in text.split("reject_reserved_block actuator", 1)[1].splitlines()[0]


def test_bash_syntax():
    bash = require_bash()
    completed = subprocess.run(
        [bash, "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_dry_run_is_nonmutating_and_renders_train_five_evaluations_and_aggregate(tmp_path):
    isolated = make_isolated_script(tmp_path)
    results_root = tmp_path / "scaleup"
    completed = invoke(
        isolated,
        "--seed", 22,
        "--env-seed", 120000,
        "--noise-seed", 120000,
        "--grad-seed", 120000,
        "--dry-run",
        results_root=results_root,
    )
    assert completed.returncode == 0, completed.stderr
    assert not results_root.exists()
    command_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("DRY-RUN: ")
    ]
    assert len(command_lines) == 7
    train = next(line for line in command_lines if "train_td3bc.py" in line)
    assert "--variant hubl_constant" in train
    assert "--action-pairing executed_executed" in train
    assert "--updates 25000" in train
    assert "--heuristic-discount 0.391843318939209" in train

    channel = [line for line in command_lines if "evaluate_channel_opex.py" in line]
    inverse_only = [line for line in command_lines if "evaluate_td3bc.py" in line]
    aggregate = [
        line
        for line in command_lines
        if "aggregate_ca_opex_walker_scaleup.py" in line
    ]
    assert len(channel) == 4
    assert len(inverse_only) == 1
    assert len(aggregate) == 1
    for line in channel:
        assert "--eval-episodes 50" in line
        assert "--eval-seed 120000" in line
        assert "--eval-noise-seed 120000" in line
        assert "--gradient-noise-seed 120000" in line
        assert "--rollout-action-noise-beta 1.25" in line
    complete = next(line for line in channel if "/controls/complete.json" in line)
    calibrated_identity = next(line for line in channel if "calibrated_identity.json" in line)
    nominal = next(line for line in channel if "nominal_k8t2.json" in line)
    original = next(line for line in channel if "original_opex.json" in line)
    assert "--channel-calibration" in complete and "--baseline-transform inverse" in complete
    assert "--step-size 0.1" in complete and "--delta-max 0.25" in complete
    assert "--channel-calibration" in calibrated_identity and "--step-size 0.3" in calibrated_identity
    assert "--model-action-noise-beta 0" in nominal and "--gradient-noise-samples 8" in nominal
    assert "--model-action-noise-beta 0" in original and "--gradient-noise-samples 1" in original
    assert "--gradient-steps 1" in original
    assert "--command-transform uniform_mean_inverse" in inverse_only[0]
    assert "--eval-seed 120000" in inverse_only[0]
    assert "--eval-noise-seed 120000" in inverse_only[0]
    assert "--run-root" in aggregate[0]
    assert "/walker_seed22/aggregate.json" in aggregate[0]


@pytest.mark.parametrize(
    "flag,start,expected",
    [
        ("--env-seed", 39251, "overlaps formal block"),
        ("--env-seed", 79349, "overlaps supplemental block"),
        ("--noise-seed", 49300, "overlaps formal block"),
        ("--noise-seed", 89251, "overlaps supplemental block"),
        ("--grad-seed", 69349, "overlaps formal block"),
        ("--grad-seed", 99300, "overlaps supplemental block"),
    ],
)
def test_reserved_50_seed_blocks_are_rejected_before_execution(flag, start, expected):
    values = {"--env-seed": 120000, "--noise-seed": 120000, "--grad-seed": 120000}
    values[flag] = start
    completed = invoke(
        SCRIPT,
        "--seed", 22,
        "--env-seed", values["--env-seed"],
        "--noise-seed", values["--noise-seed"],
        "--grad-seed", values["--grad-seed"],
        "--dry-run",
    )
    assert completed.returncode != 0
    assert expected in completed.stderr


def test_existing_per_seed_root_is_rejected_even_for_dry_run(tmp_path):
    isolated = make_isolated_script(tmp_path)
    results_root = tmp_path / "scaleup"
    (results_root / "walker_seed22").mkdir(parents=True)
    sentinel = results_root / "walker_seed22/keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    completed = invoke(
        isolated,
        "--seed", 22,
        "--env-seed", 120000,
        "--noise-seed", 120000,
        "--grad-seed", 120000,
        "--dry-run",
        results_root=results_root,
    )
    assert completed.returncode != 0
    assert "refusing to reuse or overwrite existing run root" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_negative_seed_is_rejected():
    completed = invoke(
        SCRIPT,
        "--seed", 22,
        "--env-seed", -1,
        "--noise-seed", 120000,
        "--grad-seed", 120000,
        "--dry-run",
    )
    assert completed.returncode != 0
    assert "must be a non-negative integer" in completed.stderr


@pytest.mark.parametrize(
    "training_seed,expected",
    [
        (0, "reserved for development"),
        (1, "reserved for formal confirmation"),
        (10, "reserved for formal confirmation"),
    ],
)
def test_development_and_formal_training_seeds_are_rejected(
    training_seed, expected
):
    completed = invoke(
        SCRIPT,
        "--seed", training_seed,
        "--env-seed", 120000,
        "--noise-seed", 120000,
        "--grad-seed", 120000,
        "--dry-run",
    )
    assert completed.returncode != 0
    assert expected in completed.stderr
