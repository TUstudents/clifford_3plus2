"""Session 47 certificate for iterated production Gauss relaxation."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_solver import (
    sm_physical_right_production_gauss_solver_diagnostics,
)


THRESHOLDS = {
    "vacuum_initial_gauss_norm": 1e-8,
    "vacuum_final_gauss_norm": 1e-8,
    "vacuum_momentum_delta_norm": 1e-8,
    "monotonicity_violation": 5e-7,
    "family_state_delta_norm": 1e-8,
    "higgs_delta_norm": 1e-8,
    "higgs_momentum_delta_norm": 1e-8,
    "sm_link_delta_norm": 1e-8,
    "higgs_link_delta_norm": 1e-8,
    "sm_link_unitarity_residual": 1e-6,
    "higgs_link_unitarity_residual": 1e-6,
    "jit_delta_momenta": 1e-3,
    "jit_delta_history": 5e-5,
}

LOWER_BOUNDS = {
    "iteration_count": 10.0,
    "initial_gauss_norm": 1e-1,
    "total_gauss_reduction_norm": 1.8e-1,
    "total_gauss_reduction_fraction": 3e-1,
    "min_step_reduction_norm": 1e-3,
    "min_line_step": 1e-4,
    "max_line_step": 1e-4,
    "min_gradient_norm": 1e-4,
    "max_gradient_norm": 1e-4,
    "total_momentum_delta_norm": 1e-3,
}


def main() -> None:
    diagnostics = sm_physical_right_production_gauss_solver_diagnostics()
    print("QCA_SMv0 Session 47 - physical-right production iterated Gauss relaxation")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "final_gauss_norm" and value >= float(diagnostics.initial_gauss_norm):
            raise SystemExit(
                f"final_gauss_norm={value:.3e} does not improve "
                f"initial_gauss_norm={float(diagnostics.initial_gauss_norm):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE47_PHYSICAL_RIGHT_PRODUCTION_GAUSS_SOLVER_PASS")


if __name__ == "__main__":
    main()
