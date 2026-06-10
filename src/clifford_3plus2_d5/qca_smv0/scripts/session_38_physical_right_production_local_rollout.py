"""Session 38 certificate for local-force production rollout."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_rollout import (
    sm_physical_right_production_local_rollout_diagnostics,
)


THRESHOLDS = {
    "wilson_epsilon_invariance_residual": 1e-12,
    "family_norm_drift": 2e-6,
    "sm_link_unitarity_residual": 2e-6,
    "higgs_link_unitarity_residual": 2e-6,
}

LOWER_BOUNDS = {
    "site_count": 4.0,
    "higgs_delta_norm": 1e-5,
    "sm_link_delta_norm": 1e-5,
    "sm_momentum_delta_norm": 1e-6,
    "finite_difference_to_local_work_ratio": 700.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_local_rollout_diagnostics()
    print("QCA_SMv0 Session 38 - physical-right production local-force rollout")
    for field in diagnostics._fields:
        raw = getattr(diagnostics, field)
        value = float(raw)
        print(f"{field}: {value:.3e}")
        if field == "final_all_finite" and not bool(raw):
            raise SystemExit("final_all_finite is false")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE38_PHYSICAL_RIGHT_PRODUCTION_LOCAL_ROLLOUT_PASS")


if __name__ == "__main__":
    main()

