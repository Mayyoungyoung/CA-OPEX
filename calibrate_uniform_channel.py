"""Calibrate an additive Uniform+clip execution channel from action pairs.

For an interior executed action, the likelihood contribution is the Uniform
density ``1/(2 beta)``.  An action exactly at a clipping bound is censored and
contributes the corresponding boundary point mass, e.g.

    P(a_exec = low | u) = (beta - (u-low)) / (2 beta)

when ``beta > u-low``.  Treating those atoms as ordinary residual samples is
incorrect and biases the endpoint estimate downward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from generate_channel_pairs import PAIR_SCHEMA, array_sha256, sha256_file


class UnidentifiableChannelError(ValueError):
    """Raised when the observed clipped pairs have no finite beta MLE."""


def load_pair_archive(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as archive:
        expected = {"commands", "executed", "provenance_json"}
        if set(archive.files) != expected:
            raise ValueError(
                f"pair archive must contain exactly {sorted(expected)}, got "
                f"{sorted(archive.files)}"
            )
        commands = np.asarray(archive["commands"], dtype=np.float32)
        executed = np.asarray(archive["executed"], dtype=np.float32)
        provenance_text = str(np.asarray(archive["provenance_json"]).item())
    try:
        provenance = json.loads(provenance_text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid provenance_json") from error
    if not isinstance(provenance, dict) or provenance.get("schema") != PAIR_SCHEMA:
        raise ValueError(f"unsupported pair archive schema; expected {PAIR_SCHEMA}")
    if commands.ndim != 2 or commands.shape != executed.shape:
        raise ValueError("commands and executed must share shape [pairs, action_dim]")
    if commands.size == 0:
        raise ValueError("pair archive is empty")
    if not (np.all(np.isfinite(commands)) and np.all(np.isfinite(executed))):
        raise ValueError("pair archive contains non-finite actions")
    if int(provenance.get("pair_count", -1)) != commands.shape[0]:
        raise ValueError("provenance pair_count does not match commands")
    if int(provenance.get("action_dim", -1)) != commands.shape[1]:
        raise ValueError("provenance action_dim does not match commands")
    if provenance.get("commands_sha256") != array_sha256(commands):
        raise ValueError("commands hash does not match provenance")
    if provenance.get("executed_sha256") != array_sha256(executed):
        raise ValueError("executed hash does not match provenance")
    return commands, executed, provenance


def _observation_partition(
    commands: np.ndarray,
    executed: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    boundary_tolerance: float,
) -> Dict[str, np.ndarray]:
    if boundary_tolerance < 0.0 or not np.isfinite(boundary_tolerance):
        raise ValueError("boundary_tolerance must be finite and non-negative")
    if not (np.isfinite(action_low) and np.isfinite(action_high)):
        raise ValueError("action bounds must be finite")
    if action_low >= action_high:
        raise ValueError("action_low must be smaller than action_high")
    commands = np.asarray(commands, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    if commands.shape != executed.shape or commands.ndim != 2:
        raise ValueError("commands and executed must share shape [pairs, action_dim]")
    if np.any(commands < action_low - boundary_tolerance) or np.any(
        commands > action_high + boundary_tolerance
    ):
        raise ValueError("commands lie outside the declared action bounds")
    if np.any(executed < action_low - boundary_tolerance) or np.any(
        executed > action_high + boundary_tolerance
    ):
        raise ValueError("executed actions lie outside the declared action bounds")

    at_low = np.isclose(
        executed, action_low, rtol=0.0, atol=boundary_tolerance
    )
    at_high = np.isclose(
        executed, action_high, rtol=0.0, atol=boundary_tolerance
    )
    if np.any(at_low & at_high):
        raise ValueError("action bounds are indistinguishable at the tolerance")
    interior = ~(at_low | at_high)
    interior_residuals = np.abs(executed[interior] - commands[interior])
    left_distances = commands[at_low] - action_low
    right_distances = action_high - commands[at_high]
    censor_distances = np.concatenate((left_distances, right_distances))
    return {
        "interior_residuals": interior_residuals,
        "censor_distances": censor_distances,
        "at_low": at_low,
        "at_high": at_high,
        "interior": interior,
    }


def _theta_score(beta: float, total_count: int, censor_distances: np.ndarray) -> float:
    """Derivative of log likelihood with respect to theta=log(beta)."""

    return float(
        -total_count + np.sum(beta / (beta - censor_distances), dtype=np.float64)
    )


def _log_likelihood(
    beta: float,
    interior_count: int,
    censor_distances: np.ndarray,
) -> float:
    if beta <= 0.0:
        return 0.0 if interior_count > 0 else float("-inf")
    if censor_distances.size and np.any(beta <= censor_distances):
        return float("-inf")
    value = -interior_count * np.log(2.0 * beta)
    if censor_distances.size:
        value += np.log((beta - censor_distances) / (2.0 * beta)).sum(
            dtype=np.float64
        )
    return float(value)


def estimate_beta_mle(
    commands: np.ndarray,
    executed: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    boundary_tolerance: float = 1e-7,
) -> Dict[str, object]:
    """Return the exact one-dimensional MLE for the censored likelihood.

    In ``theta=log(beta)``, the finite-beta log likelihood is concave.  Its
    score is monotone, so a bracketed bisection finds the unique interior
    optimum; otherwise the MLE is the largest uncensored residual endpoint.
    """

    partition = _observation_partition(
        commands,
        executed,
        action_low=action_low,
        action_high=action_high,
        boundary_tolerance=boundary_tolerance,
    )
    interior_residuals = partition["interior_residuals"]
    censor_distances = partition["censor_distances"]
    total_count = int(np.asarray(commands).size)
    interior_count = int(interior_residuals.size)
    censored_count = int(censor_distances.size)
    deterministic_residual = np.abs(
        np.asarray(executed, dtype=np.float64)
        - np.clip(np.asarray(commands, dtype=np.float64), action_low, action_high)
    )
    if np.all(deterministic_residual <= boundary_tolerance):
        estimate = 0.0
        optimum = "degenerate_zero_noise"
        log_likelihood = 0.0
    elif interior_count == 0:
        raise UnidentifiableChannelError(
            "all scalar observations are clipped; the likelihood has no finite "
            "beta maximizer without an upper-bound assumption"
        )
    else:
        interior_lower = float(interior_residuals.max(initial=0.0))
        censor_lower = float(censor_distances.max(initial=0.0))
        support_lower = max(interior_lower, censor_lower)
        # A censoring atom has zero probability at beta == its distance.  If
        # the endpoint is set by an interior residual and all censor distances
        # are strictly smaller, the support boundary can itself be the MLE.
        censor_active_at_lower = bool(
            censor_distances.size
            and np.any(
                np.isclose(
                    censor_distances,
                    support_lower,
                    rtol=1e-13,
                    atol=np.finfo(np.float64).eps * max(1.0, support_lower),
                )
            )
        )
        lower_probe = np.nextafter(support_lower, np.inf)
        if lower_probe <= 0.0:
            lower_probe = np.finfo(np.float64).tiny
        lower_score = _theta_score(lower_probe, total_count, censor_distances)
        if not censor_active_at_lower and lower_score <= 0.0:
            estimate = support_lower
            optimum = "uncensored_support_endpoint"
        else:
            upper = max(2.0 * lower_probe, 1e-8)
            for _ in range(256):
                if _theta_score(upper, total_count, censor_distances) <= 0.0:
                    break
                upper *= 2.0
            else:
                raise UnidentifiableChannelError(
                    "failed to bracket a finite censored-likelihood optimum"
                )
            lower = lower_probe
            for _ in range(120):
                midpoint = np.sqrt(lower * upper)
                if _theta_score(midpoint, total_count, censor_distances) > 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            estimate = float(np.sqrt(lower * upper))
            optimum = "censored_likelihood_stationary_point"
        log_likelihood = _log_likelihood(
            estimate, interior_count, censor_distances
        )

    return {
        "beta_mle": float(estimate),
        "log_likelihood_at_mle": float(log_likelihood),
        "optimum_type": optimum,
        "scalar_observation_count": total_count,
        "interior_scalar_count": interior_count,
        "left_censored_scalar_count": int(partition["at_low"].sum()),
        "right_censored_scalar_count": int(partition["at_high"].sum()),
        "censored_scalar_count": censored_count,
        "censored_fraction": censored_count / total_count,
        "boundary_tolerance": float(boundary_tolerance),
        "likelihood_reference_measure": (
            "Lebesgue density on the open action interval plus point masses "
            "at both clipping bounds"
        ),
        "clipping_treatment": "exact boundary observations are censored atoms",
    }


def parametric_bootstrap(
    commands: np.ndarray,
    estimate: float,
    *,
    action_low: float,
    action_high: float,
    boundary_tolerance: float,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> Dict[str, object]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    commands = np.asarray(commands, dtype=np.float32)
    estimates = []
    failure_count = 0
    for _ in range(replicates):
        noise = rng.uniform(-estimate, estimate, size=commands.shape).astype(
            np.float32
        )
        simulated = np.clip(
            commands + noise, action_low, action_high
        ).astype(np.float32, copy=False)
        try:
            result = estimate_beta_mle(
                commands,
                simulated,
                action_low=action_low,
                action_high=action_high,
                boundary_tolerance=boundary_tolerance,
            )
        except UnidentifiableChannelError:
            failure_count += 1
            continue
        estimates.append(float(result["beta_mle"]))
    if not estimates:
        raise UnidentifiableChannelError(
            "all parametric bootstrap samples had unbounded likelihoods"
        )
    values = np.asarray(estimates, dtype=np.float64)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return {
        "method": "conditional_parametric_percentile_bootstrap",
        "conditions_on_observed_commands": True,
        "replicates_requested": int(replicates),
        "replicates_finite": int(values.size),
        "replicates_unbounded": int(failure_count),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "beta_interval": [float(lower), float(upper)],
        "bootstrap_mean": float(values.mean()),
        "bootstrap_std": float(values.std()),
        "beta_estimates": values.tolist(),
        "beta_estimates_sha256": hashlib.sha256(
            np.asarray(values, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def calibrate(
    pair_path: Path,
    *,
    boundary_tolerance: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> Dict[str, object]:
    pair_path = pair_path.resolve()
    commands, executed, provenance = load_pair_archive(pair_path)
    action_low = float(provenance["action_low"])
    action_high = float(provenance["action_high"])
    estimate = estimate_beta_mle(
        commands,
        executed,
        action_low=action_low,
        action_high=action_high,
        boundary_tolerance=boundary_tolerance,
    )
    uncertainty = parametric_bootstrap(
        commands,
        float(estimate["beta_mle"]),
        action_low=action_low,
        action_high=action_high,
        boundary_tolerance=boundary_tolerance,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "status": "complete",
        "estimator": "censored_uniform_plus_clip_mle",
        "estimator_uses_provenance_beta": False,
        "pair_archive_path": str(pair_path),
        "pair_archive_sha256": sha256_file(pair_path),
        "pair_count": int(commands.shape[0]),
        "action_dim": int(commands.shape[1]),
        "commands_sha256": array_sha256(commands),
        "executed_sha256": array_sha256(executed),
        "action_low": action_low,
        "action_high": action_high,
        "estimate": estimate,
        "uncertainty": uncertainty,
        "input_provenance": provenance,
        "calibration_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="paired .npz archive")
    parser.add_argument("--output", required=True, help="calibration JSON")
    parser.add_argument("--boundary-tolerance", type=float, default=1e-7)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=271828)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run validation/calibration without writing JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = calibrate(
        Path(args.pairs),
        boundary_tolerance=args.boundary_tolerance,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    if args.dry_run:
        result["status"] = "dry_run"
    else:
        save_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
