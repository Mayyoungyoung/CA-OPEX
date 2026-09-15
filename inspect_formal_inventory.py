"""Exploratory helper: compact inventory dump for formal + supplemental records.

Read-only inspection used while designing the multi-seed aggregator.
"""

import json
import sys
from pathlib import Path

AGGREGATE = Path(
    "/root/hubl_research_20260914/results/inverse_residual_confirm/aggregate/"
    "frozen_confirmation.json"
)
SUPPLEMENTAL = Path(
    "/root/hubl_research_20260914/results/equal_compute_nominal_control/"
    "aggregate_v2/independent_reaggregation.json"
)
SAMPLE_OPEX = Path(
    "/root/hubl_research_20260914/results/inverse_residual_confirm/external_controls/"
    "ca_opex_inverse_seed1_beta125_50.json"
)
SUPP_NOMINAL = Path(
    "/root/hubl_research_20260914/results/equal_compute_nominal_control/holdout/"
    "nominal_tuned_seed1_fresh50.json"
)


def main() -> int:
    payload = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    print("== formal external_evaluations ==")
    for entry in payload["external_evaluations"]:
        sources = ";".join(
            Path(source["path"]).name for source in entry.get("sources", [])
        )
        print(
            f"{entry.get('run_id'):45s} method={entry.get('method_id'):35s} "
            f"seed={entry.get('training_seed')} files={sources}"
        )

    if SUPPLEMENTAL.is_file():
        supp = json.loads(SUPPLEMENTAL.read_text(encoding="utf-8"))
        print("\n== supplemental v2 keys:", sorted(supp.keys()))
        for key in ("controllers", "controller_rows", "absolute_controller_rows"):
            if key in supp:
                for row in supp[key]:
                    label = row.get("controller_id") or row.get("label")
                    print(
                        f"  {label} seed={row.get('training_seed')} "
                        f"raw={row.get('raw_score_mean')} files="
                        f"{Path(row.get('raw_path', '?')).name if row.get('raw_path') else '?'}"
                    )
    else:
        print("\nsupplemental v2 aggregate missing:", SUPPLEMENTAL)

    sample = json.loads(SAMPLE_OPEX.read_text(encoding="utf-8"))
    print("\n== sample formal opex record keys:", sorted(sample.keys()))
    arms = sample.get("arms", {})
    print("arm keys:", sorted(arms.keys()))
    for arm_name, arm in arms.items():
        seeds = arm.get("environment_seeds", [])
        rets = arm.get("returns", [])
        print(
            f"  arm={arm_name}: n={len(rets)} env_first={seeds[:1]} "
            f"mean={sum(rets) / max(len(rets), 1):.4f}"
        )
    ctrl = sample.get("controller", {})
    print("controller:", {k: ctrl.get(k) for k in ("baseline_transform", "step_size", "gradient_noise_samples", "gradient_steps", "delta_max", "model_beta", "calibration_mode")})

    supp_nom = json.loads(SUPP_NOMINAL.read_text(encoding="utf-8"))
    print("\n== supplemental nominal record keys:", sorted(supp_nom.keys()))
    arms2 = supp_nom.get("arms", {})
    print("arm keys:", sorted(arms2.keys()))
    for arm_name, arm in arms2.items():
        seeds = arm.get("environment_seeds", [])
        rets = arm.get("returns", [])
        print(
            f"  arm={arm_name}: n={len(rets)} env_first={seeds[:1]} "
            f"mean={sum(rets) / max(len(rets), 1):.4f}"
        )
    ctrl2 = supp_nom.get("controller", {})
    print("controller:", {k: ctrl2.get(k) for k in ("baseline_transform", "step_size", "gradient_noise_samples", "gradient_steps", "delta_max", "model_beta", "calibration_mode")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
