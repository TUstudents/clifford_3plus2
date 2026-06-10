"""Session 48 certificate for the Gauss-projected production step."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_projected import (
    sm_physical_right_production_projected_step_diagnostics,
)


THRESHOLDS = {
    "vacuum_unprojected_gauss_norm": 1e-8,
    "vacuum_projected_gauss_norm": 1e-8,
    "vacuum_projection_momentum_delta_norm": 1e-8,
    "history_monotonicity_violation": 5e-7,
    "history_final_gauss_residual": 1e-7,
    "family_state_delta_norm": 1e-8,
    "higgs_delta_norm": 1e-8,
    "higgs_momentum_delta_norm": 1e-8,
    "sm_link_delta_norm": 1e-8,
    "higgs_link_delta_norm": 1e-8,
    "sm_link_unitarity_residual": 1e-6,
    "higgs_link_unitarity_residual": 1e-6,
    "jit_delta_projected_sm_momenta": 1.5e-3,
    "jit_delta_projected_history": 5e-5,
}

LOWER_BOUNDS = {
    "projection_iteration_count": 10.0,
    "unprojected_gauss_norm": 1e-1,
    "gauss_reduction_norm": 1.8e-1,
    "gauss_reduction_fraction": 3e-1,
    "min_history_step_reduction_norm": 1e-3,
    "projection_momentum_delta_norm": 1e-3,
}


def main() -> None:
    diagnostics = sm_physical_right_production_projected_step_diagnostics()
    print("QCA_SMv0 Session 48 - physical-right production Gauss-projected step")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "projected_gauss_norm" and value >= float(diagnostics.unprojected_gauss_norm):
            raise SystemExit(
                f"projected_gauss_norm={value:.3e} does not improve "
                f"unprojected_gauss_norm={float(diagnostics.unprojected_gauss_norm):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE48_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_STEP_PASS")


if __name__ == "__main__":
    main()
