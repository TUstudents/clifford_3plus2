"""Session 49 certificate for the Gauss-projected production rollout."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_projected_rollout import (
    sm_physical_right_production_projected_rollout_diagnostics,
)


THRESHOLDS = {
    "vacuum_initial_gauss_norm": 1e-8,
    "vacuum_final_gauss_norm": 1e-8,
    "vacuum_momentum_delta_norm": 1e-8,
    "max_step_history_monotonicity_violation": 5e-7,
    "max_step_history_final_residual": 1e-7,
    "max_projected_sm_link_unitarity_residual": 1e-6,
    "max_projected_higgs_link_unitarity_residual": 1e-6,
}

LOWER_BOUNDS = {
    "step_count": 2.0,
    "projection_iteration_count": 10.0,
    "initial_gauss_norm": 1e-1,
    "raw_final_gauss_norm": 1e-1,
    "final_gauss_reduction_vs_raw_norm": 2e-1,
    "final_gauss_reduction_vs_raw_fraction": 3.5e-1,
    "min_step_projection_reduction_fraction": 5e-2,
    "min_step_projection_momentum_delta_norm": 1.0,
    "history_all_finite": 1.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_projected_rollout_diagnostics()
    print("QCA_SMv0 Session 49 - physical-right production Gauss-projected rollout")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "projected_final_gauss_norm" and value >= float(diagnostics.raw_final_gauss_norm):
            raise SystemExit(
                f"projected_final_gauss_norm={value:.3e} does not improve "
                f"raw_final_gauss_norm={float(diagnostics.raw_final_gauss_norm):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE49_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_ROLLOUT_PASS")


if __name__ == "__main__":
    main()
