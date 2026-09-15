"""Run the immutable post-confirmation equal-compute nominal-Q audit.

The script owns development selection, the conditional one-point endpoint
extension, fresh-seed holdout evaluation, validation, and aggregation.  A
normal invocation requires every output to be absent.  ``--resume-missing``
accepts an interrupted invocation only after validating every existing raw
JSON.  ``--dry-run`` validates hashes and prints commands without executing or
creating files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


SCHEMA = "equal-compute-nominal-control-protocol-v1"
EVIDENCE = "post_confirmation_fresh_rollout_mechanism_audit"


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise AuditError(f"refusing to overwrite output: {path}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise AuditError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def same_float(actual: object, expected: float, label: str) -> None:
    try:
        observed = float(actual)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric") from exc
    if not math.isfinite(observed) or not math.isclose(
        observed, float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise AuditError(f"{label} mismatch: {observed!r} != {expected!r}")


def validate_protocol(p: Mapping[str, Any]) -> None:
    if p.get("schema_version") != SCHEMA or p.get("evidence_label") != EVIDENCE:
        raise AuditError("unsupported protocol schema/evidence label")
    chronology = p.get("chronology", {})
    if chronology.get("formal_results_seen_before_protocol_freeze") is not True:
        raise AuditError("protocol must disclose that formal results were seen")
    if chronology.get("globally_blind_confirmation_claim") is not False:
        raise AuditError("protocol makes an invalid blindness claim")
    expected_grid = [0.01, 0.03, 0.1, 0.3, 1.0]
    if [float(x) for x in p["development"]["step_size_grid"]] != expected_grid:
        raise AuditError("development eta grid mismatch")
    dev = p["development"]
    required_dev = (10, 28300, 38300, 58300)
    observed_dev = (
        dev["episode_count_per_arm"], dev["environment_seed_start"],
        dev["action_noise_seed_start"], dev["gradient_noise_seed_start"],
    )
    if observed_dev != required_dev:
        raise AuditError("development rollout block mismatch")
    hold = p["holdout"]
    required_hold = (50, 79300, 89300, 99300)
    observed_hold = (
        hold["episode_count_per_arm"], hold["environment_seed_start"],
        hold["action_noise_seed_start"], hold["gradient_noise_seed_start"],
    )
    if observed_hold != required_hold:
        raise AuditError("holdout rollout block mismatch")
    if [int(x["training_seed"]) for x in hold["base_checkpoints"]] != [1, 10]:
        raise AuditError("holdout checkpoint seeds must be [1, 10]")
    for controller in p["holdout"]["controllers"].values():
        if int(controller["gradient_noise_samples"]) != 8 or int(
            controller["gradient_steps"]
        ) != 2:
            raise AuditError("all holdout controllers must use equal K=8,T=2")
    stats = p["statistics"]
    if stats["sample_standard_deviation_ddof"] != 1:
        raise AuditError("paired sample standard deviation must use ddof=1")
    if stats["bootstrap"]["replicates"] != 20000:
        raise AuditError("bootstrap replicate count must be 20000")


def validate_inputs(p: Mapping[str, Any]) -> None:
    paths = p["paths"]
    python_bin = Path(paths["python_bin"])
    code_dir = Path(paths["code_dir"])
    if not python_bin.is_file() or not code_dir.is_dir():
        raise AuditError("python_bin or code_dir is missing")
    for filename, expected in p["implementation"].items():
        path = code_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"implementation hash mismatch: {filename}")
    inputs = [p["development"]["base_checkpoint"]]
    inputs.extend(p["holdout"]["base_checkpoints"])
    inputs.append(p["channel_calibration"])
    for item in inputs:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"input artifact hash mismatch: {path}")
    calibration = read_json(Path(p["channel_calibration"]["path"]))
    estimate = calibration.get("estimate")
    if not isinstance(estimate, dict):
        raise AuditError("calibration artifact lacks estimate object")
    same_float(
        estimate.get("beta_mle"),
        p["channel_calibration"]["estimated_beta"],
        "calibration estimate.beta_mle",
    )
    if int(calibration.get("pair_count", -1)) != int(p["channel_calibration"]["pair_count"]):
        raise AuditError("calibration pair count mismatch")


def eta_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def make_spec(
    p: Mapping[str, Any], *, stage: str, controller_id: str,
    checkpoint: Mapping[str, Any], eta: float,
) -> Dict[str, Any]:
    if stage == "development":
        c = p["development"]["controller"]
        block = p["development"]
        alias = c["semantic_alias"]
        filename = f"nominal_q_identity_eta_{eta_token(eta)}_seed0_dev10.json"
        calibrated = False
    else:
        c = p["holdout"]["controllers"][controller_id]
        block = p["holdout"]
        alias = c["semantic_alias"]
        filename = f"{controller_id}_seed{checkpoint['training_seed']}_fresh50.json"
        calibrated = c["model_source"] == "pair_calibration"
    return {
        "stage": stage, "controller_id": controller_id, "semantic_alias": alias,
        "checkpoint": dict(checkpoint), "eta": float(eta),
        "baseline_transform": c["baseline_transform"],
        "model_beta": float(p["channel_calibration"]["estimated_beta"] if calibrated else 0.0),
        "calibrated": calibrated, "K": int(c["gradient_noise_samples"]),
        "T": int(c["gradient_steps"]), "delta": float(c["delta_max"]),
        "episodes": int(block["episode_count_per_arm"]),
        "env_seed": int(block["environment_seed_start"]),
        "action_seed": int(block["action_noise_seed_start"]),
        "gradient_seed": int(block["gradient_noise_seed_start"]),
        "path": Path(p["paths"]["output_root"]) / p["artifacts"][
            "development_directory" if stage == "development" else "holdout_directory"
        ] / filename,
    }


def command(p: Mapping[str, Any], s: Mapping[str, Any]) -> List[str]:
    env = p["environment"]
    result = [
        p["paths"]["python_bin"], str(Path(p["paths"]["code_dir"]) / p["paths"]["evaluator"]),
        "--base-checkpoint", s["checkpoint"]["path"], "--output", str(s["path"]),
        "--env-name", env["name"], "--device", "cuda", "--step-size", str(s["eta"]),
        "--baseline-transform", s["baseline_transform"], "--gradient-noise-samples", str(s["K"]),
        "--gradient-steps", str(s["T"]), "--gradient-noise-seed", str(s["gradient_seed"]),
        "--delta-max", str(s["delta"]), "--rollout-action-noise-beta", str(env["rollout_action_noise_beta"]),
        "--eval-episodes", str(s["episodes"]), "--eval-seed", str(s["env_seed"]),
        "--eval-noise-seed", str(s["action_seed"]), "--reference-min-score", str(env["reference_min_score"]),
        "--reference-max-score", str(env["reference_max_score"]),
    ]
    if s["calibrated"]:
        result += ["--channel-calibration", p["channel_calibration"]["path"]]
    else:
        result += ["--model-action-noise-beta", "0"]
    return result


def validate_raw(p: Mapping[str, Any], s: Mapping[str, Any]) -> Dict[str, Any]:
    raw = read_json(Path(s["path"]))
    if raw.get("raw_schema") != "channel_opex_v1" or raw.get("status") != "complete":
        raise AuditError(f"incomplete raw output: {s['path']}")
    raw_implementation = raw.get("implementation", {})
    implementation_key_map = {
        "evaluate_channel_opex.py": "evaluate_sha256",
        "inverse_residual_core.py": "inverse_residual_core_sha256",
        "td3bc_core.py": "td3bc_core_sha256",
        "train_td3bc.py": "train_td3bc_sha256",
        "evaluation_controls.py": "evaluation_controls_sha256",
        "train_inverse_residual_adapter.py": (
            "train_inverse_residual_adapter_sha256"
        ),
    }
    expected_raw_implementation = {
        implementation_key_map[filename]: digest
        for filename, digest in p["implementation"].items()
    }
    if raw_implementation != expected_raw_implementation:
        raise AuditError(f"raw implementation mismatch: {s['path']}")
    base = raw.get("base_checkpoint", {})
    if base.get("sha256") != s["checkpoint"]["sha256"]:
        raise AuditError("raw base checkpoint mismatch")
    c = raw.get("controller", {})
    expected = {
        "baseline_transform": s["baseline_transform"], "gradient_steps": s["T"],
        "K": s["K"], "execution_noise_samples": s["K"], "delta_max": s["delta"],
        "step_size": s["eta"], "model_beta": s["model_beta"], "q_reducer": "mean_q1",
        "actor_parameter_updates": 0, "critic_parameter_updates": 0,
    }
    for key, value in expected.items():
        if isinstance(value, float): same_float(c.get(key), value, f"controller.{key}")
        elif c.get(key) != value: raise AuditError(f"controller.{key} mismatch")
    calibration = raw.get("calibration", {})
    expected_cal_sha = p["channel_calibration"]["sha256"] if s["calibrated"] else None
    if calibration.get("calibration_sha256") != expected_cal_sha:
        raise AuditError("raw calibration hash mismatch")
    ep = raw.get("evaluation_protocol", {})
    checks = {"episode_count_per_arm": s["episodes"], "environment_seed_start": s["env_seed"],
              "action_noise_seed_start": s["action_seed"], "gradient_noise_seed_start": s["gradient_seed"]}
    for key, value in checks.items():
        if ep.get(key) != value: raise AuditError(f"evaluation_protocol.{key} mismatch")
    expected_env = list(range(s["env_seed"], s["env_seed"] + s["episodes"]))
    expected_action = list(range(s["action_seed"], s["action_seed"] + s["episodes"]))
    expected_gradient = list(range(s["gradient_seed"], s["gradient_seed"] + s["episodes"]))
    for name in ("baseline_only", "adapted"):
        arm = raw.get("arms", {}).get(name, {})
        if arm.get("environment_seeds") != expected_env or arm.get("action_noise_seeds") != expected_action:
            raise AuditError(f"{name} paired seed mismatch")
        if arm.get("gradient_noise_seeds") != expected_gradient or len(arm.get("returns", [])) != s["episodes"]:
            raise AuditError(f"{name} gradient seed/return count mismatch")
        values = np.asarray(arm["returns"], dtype=np.float64)
        if not np.isfinite(values).all(): raise AuditError(f"{name} has nonfinite return")
        same_float(arm.get("return_mean"), float(values.mean()), f"{name}.return_mean")
    adapted = raw["arms"]["adapted"]
    if adapted.get("q1_rows_per_action") != s["K"] * s["T"]:
        raise AuditError("adapted Q-row count mismatch")
    diffs = np.asarray(adapted["returns"]) - np.asarray(raw["arms"]["baseline_only"]["returns"])
    recorded = np.asarray(raw.get("paired", {}).get("adapted_minus_baseline_returns", []))
    if not np.array_equal(diffs, recorded): raise AuditError("raw paired differences mismatch")
    return raw


def execute(p: Mapping[str, Any], s: Mapping[str, Any], *, resume: bool) -> Dict[str, Any]:
    path = Path(s["path"]); log = path.with_suffix(".console.log")
    if path.exists():
        if not resume: raise AuditError(f"refusing to reuse output without --resume-missing: {path}")
        return validate_raw(p, s)
    if path.with_name(path.name + ".tmp").exists() or log.exists():
        raise AuditError(f"partial/conflicting artifact prevents resume: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command(p, s), cwd=p["paths"]["code_dir"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise AuditError(f"evaluator failed for {path} (exit {completed.returncode}):\n{completed.stdout[-4000:]}")
    raw = validate_raw(p, s)
    write_new(log, completed.stdout)
    return raw


def select(candidates: Sequence[tuple[Dict[str, Any], Dict[str, Any]]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return sorted(candidates, key=lambda pair: (-float(pair[1]["arms"]["adapted"]["normalized_score_mean"]), pair[0]["eta"]))[0]


def paired_stats(values: np.ndarray, *, bootstrap_seed: int, replicates: int) -> Dict[str, Any]:
    n = int(values.size); mean = float(values.mean()); sd = float(values.std(ddof=1)); sem = sd / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    boot = values[rng.integers(0, n, size=(replicates, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975]).tolist()
    return {"n_episode_pairs": n, "mean": mean, "sample_std_ddof1": sd, "standard_error": sem,
            "t95_low": mean - critical * sem, "t95_high": mean + critical * sem,
            "bootstrap_replicates": replicates, "bootstrap_seed": bootstrap_seed,
            "bootstrap95_low": float(lo), "bootstrap95_high": float(hi),
            "positive_episode_count": int((values > 0).sum())}


def render_csv(report: Mapping[str, Any]) -> str:
    out = io.StringIO(newline=""); fields = ["comparison_id", "training_seed", "mean", "sample_std_ddof1", "t95_low", "t95_high", "bootstrap95_low", "bootstrap95_high", "positive_episode_count", "n_episode_pairs"]
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n"); writer.writeheader()
    for item in report["paired_comparisons"]: writer.writerow({k: item.get(k) for k in fields})
    return out.getvalue()


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Equal-compute nominal-Q mechanism audit", "", f"Evidence label: `{EVIDENCE}`. Formal results were seen before this fresh-seed audit; this is not blind confirmation.", "", "| Comparison | Checkpoint | mean normalized delta | t95 | bootstrap95 | positive |", "|---|---:|---:|---:|---:|---:|"]
    for x in report["paired_comparisons"]:
        lines.append(f"| {x['comparison_id']} | {x['training_seed']} | {x['mean']:.6f} | [{x['t95_low']:.6f}, {x['t95_high']:.6f}] | [{x['bootstrap95_low']:.6f}, {x['bootstrap95_high']:.6f}] | {x['positive_episode_count']}/{x['n_episode_pairs']} |")
    lines += ["", "Cross-checkpoint values below are descriptive means of two checkpoint-level means only; no training-seed interval or p-value is claimed.", ""]
    for key, value in report["cross_checkpoint_descriptive_means"].items(): lines.append(f"- `{key}`: {value:.6f}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    protocol_path = args.protocol.resolve(); p = read_json(protocol_path)
    validate_protocol(p); validate_inputs(p)
    if args.dry_run:
        print("DRY-RUN ONLY: no commands will execute and no files will be created")
        checkpoint = p["development"]["base_checkpoint"]
        for eta in p["development"]["step_size_grid"]:
            print(shlex.join(command(p, make_spec(p, stage="development", controller_id="nominal_dev", checkpoint=checkpoint, eta=float(eta)))))
        print("CONDITIONAL lower-endpoint command uses eta=0.003; upper-endpoint command uses eta=3.0; exactly one runs only if its endpoint wins.")
        return {"status": "dry_run", "evidence_label": EVIDENCE}
    output_root = Path(p["paths"]["output_root"])
    if not args.resume_missing and output_root.exists() and any(output_root.rglob("*")):
        raise AuditError(f"output root is not empty: {output_root}")
    checkpoint = p["development"]["base_checkpoint"]
    candidates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for eta in p["development"]["step_size_grid"]:
        spec = make_spec(p, stage="development", controller_id="nominal_dev", checkpoint=checkpoint, eta=float(eta))
        candidates.append((spec, execute(p, spec, resume=args.resume_missing)))
    initial_spec, _ = select(candidates)
    extension = None
    if initial_spec["eta"] == 0.01: extension = 0.003
    elif initial_spec["eta"] == 1.0: extension = 3.0
    if extension is not None:
        spec = make_spec(p, stage="development", controller_id="nominal_dev", checkpoint=checkpoint, eta=extension)
        candidates.append((spec, execute(p, spec, resume=args.resume_missing)))
    selected_spec, selected_raw = select(candidates)
    selection = {
        "schema_version": "equal-compute-nominal-control-selection-v1", "status": "complete",
        "evidence_label": EVIDENCE, "protocol_path": str(protocol_path), "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()), "initial_selected_step_size": initial_spec["eta"],
        "endpoint_extension_step_size": extension, "selected_step_size": selected_spec["eta"],
        "candidates": [{"step_size": s["eta"], "adapted_normalized_score_mean": r["arms"]["adapted"]["normalized_score_mean"], "path": str(s["path"]), "sha256": sha256_file(Path(s["path"]))} for s, r in sorted(candidates, key=lambda z: z[0]["eta"])],
        "formal_results_seen_before_protocol_freeze": True, "globally_blind_confirmation_claim": False,
    }
    selection_path = output_root / p["artifacts"]["selection_json"]
    if selection_path.exists():
        if not args.resume_missing or read_json(selection_path) != selection: raise AuditError("existing selection does not match recomputation")
    else: write_new(selection_path, json_text(selection))
    hold_raw: Dict[int, Dict[str, Dict[str, Any]]] = {}
    hold_specs: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for checkpoint in p["holdout"]["base_checkpoints"]:
        seed = int(checkpoint["training_seed"]); specs = [
            make_spec(p, stage="holdout", controller_id="nominal_tuned", checkpoint=checkpoint, eta=selected_spec["eta"]),
            make_spec(p, stage="holdout", controller_id="calibrated_identity_eta_0.3", checkpoint=checkpoint, eta=0.3),
            make_spec(p, stage="holdout", controller_id="complete_calibrated_inverse_eta_0.1", checkpoint=checkpoint, eta=0.1),
        ]
        if selected_spec["eta"] != 0.3:
            specs.insert(1, make_spec(p, stage="holdout", controller_id="nominal_matched_eta_0.3", checkpoint=checkpoint, eta=0.3))
        hold_specs[seed] = {s["controller_id"]: s for s in specs}
        hold_raw[seed] = {
            s["controller_id"]: execute(p, s, resume=args.resume_missing)
            for s in specs
        }
        left = hold_raw[seed]["nominal_tuned"]["arms"]["baseline_only"]
        right = hold_raw[seed]["calibrated_identity_eta_0.3"]["arms"]["baseline_only"]
        for field in p["holdout"]["identity_baseline_elementwise_check"]["fields"]:
            if left[field] != right[field]: raise AuditError(f"identity baselines differ elementwise for seed {seed}: {field}")
    comparisons = []; skipped = []; factor = 100.0 / (p["environment"]["reference_max_score"] - p["environment"]["reference_min_score"])
    for index, comp in enumerate(p["statistics"]["comparisons_in_order"]):
        for seed, raw_by_id in hold_raw.items():
            if comp["right"] not in raw_by_id:
                skipped.append({"comparison_id": comp["id"], "training_seed": seed, "reason": "nominal matched controller duplicates nominal tuned eta=0.3"}); continue
            left = np.asarray(raw_by_id[comp["left"]]["arms"]["adapted"]["returns"], dtype=np.float64)
            right = np.asarray(raw_by_id[comp["right"]]["arms"]["adapted"]["returns"], dtype=np.float64)
            stats = paired_stats((left - right) * factor, bootstrap_seed=p["statistics"]["bootstrap"]["seed_root"] + 1000 * seed + index, replicates=20000)
            comparisons.append({"comparison_id": comp["id"], "left": comp["left"], "right": comp["right"], "training_seed": seed, **stats})
    grouped: Dict[str, List[float]] = {}
    for item in comparisons: grouped.setdefault(item["comparison_id"], []).append(item["mean"])
    report = {
        "schema_version": "equal-compute-nominal-control-aggregate-v1", "status": "complete", "evidence_label": EVIDENCE,
        "protocol_sha256": sha256_file(protocol_path), "selection_sha256": sha256_file(selection_path), "runner_sha256": sha256_file(Path(__file__).resolve()),
        "selected_nominal_step_size": selected_spec["eta"], "paired_comparisons": comparisons, "skipped_comparisons": skipped,
        "cross_checkpoint_descriptive_means": {key: float(np.mean(values)) for key, values in grouped.items()},
        "identity_baseline_elementwise_identical": True,
        "raw_artifacts": [
            {
                "training_seed": seed,
                "controller_id": key,
                "semantic_alias": p["holdout"]["controllers"][key][
                    "semantic_alias"
                ],
                "path": str(hold_specs[seed][key]["path"]),
                "sha256": sha256_file(Path(hold_specs[seed][key]["path"])),
            }
            for seed, raws in hold_raw.items()
            for key in raws
        ],
        "cross_checkpoint_inference_guardrail": p["statistics"]["cross_checkpoint_summary"],
    }
    aggregate = output_root / p["artifacts"]["aggregate_directory"]
    outputs = [(aggregate / p["artifacts"]["aggregate_json"], json_text(report)), (aggregate / p["artifacts"]["aggregate_csv"], render_csv(report)), (aggregate / p["artifacts"]["aggregate_markdown"], render_markdown(report))]
    for path, text in outputs:
        if path.exists():
            if not args.resume_missing or path.read_text(encoding="utf-8") != text: raise AuditError(f"existing aggregate mismatch: {path}")
        else: write_new(path, text)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("equal_compute_nominal_control_protocol.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-missing", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except AuditError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
