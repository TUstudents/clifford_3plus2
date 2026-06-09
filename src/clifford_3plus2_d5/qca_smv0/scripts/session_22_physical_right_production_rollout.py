"""Session 22 certificate for QCA_SMv0 physical-right production rollout."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_rollout_diagnostics,
)


THRESHOLDS = {
    "one_step_direct_residual": 2e-7,
    "loop_scan_final_residual": 5e-7,
    "loop_scan_observation_residual": 1e-6,
    "rollout_family_norm_drift": 1e-5,
    "rollout_max_family_norm_drift": 1e-5,
    "max_sm_link_unitarity_residual": 8e-7,
    "max_higgs_link_unitarity_residual": 8e-7,
}

LOWER_BOUNDS = {
    "higgs_field_total_delta_norm": 1e-7,
    "higgs_momentum_total_delta_norm": 1e-7,
    "sm_momentum_total_delta_norm": 1e-7,
    "production_zero_yukawa_family_difference_norm": 1e-6,
    "production_zero_yukawa_higgs_momentum_difference_norm": 1e-6,
    "record_count": 2.5,
    "scan_all_finite": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_rollout_diagnostics()
    print("QCA_SMv0 Session 22 - physical-right production rollout")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE22_PHYSICAL_RIGHT_PRODUCTION_ROLLOUT_PASS")


if __name__ == "__main__":
    main()
