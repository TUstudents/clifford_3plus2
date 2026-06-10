"""Session 42 certificate for the local physical-right fermion current."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_current import (
    sm_physical_right_production_local_current_diagnostics,
)


THRESHOLDS = {
    "finite_difference_local_max_residual": 1e-4,
    "finite_difference_local_norm_residual": 1e-4,
    "production_local_alias_residual": 1e-12,
    "zero_state_local_current_norm": 1e-7,
    "production_fermion_epsilon_residual": 1e-12,
    "production_family_norm_drift": 8e-5,
    "max_sm_link_unitarity_residual": 8e-7,
    "max_higgs_link_unitarity_residual": 8e-7,
}

LOWER_BOUNDS = {
    "finite_difference_current_norm": 1e-2,
    "local_current_norm": 1e-2,
    "estimated_work_reduction_ratio": 60.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_local_current_diagnostics()
    print("QCA_SMv0 Session 42 - physical-right production local fermion current")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE42_PHYSICAL_RIGHT_PRODUCTION_LOCAL_CURRENT_PASS")


if __name__ == "__main__":
    main()
