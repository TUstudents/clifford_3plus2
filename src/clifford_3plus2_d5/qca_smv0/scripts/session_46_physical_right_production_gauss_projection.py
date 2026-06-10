"""Session 46 certificate for production Gauss relaxation."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_projection import (
    sm_physical_right_production_gauss_projection_diagnostics,
)


THRESHOLDS = {
    "vacuum_initial_gauss_norm": 1e-8,
    "vacuum_relaxed_gauss_norm": 1e-8,
    "vacuum_momentum_delta_norm": 1e-8,
    "family_state_delta_norm": 1e-8,
    "higgs_delta_norm": 1e-8,
    "higgs_momentum_delta_norm": 1e-8,
    "sm_link_delta_norm": 1e-8,
    "higgs_link_delta_norm": 1e-8,
    "sm_link_unitarity_residual": 1e-6,
    "higgs_link_unitarity_residual": 1e-6,
    "jit_delta_momenta": 5e-6,
}

LOWER_BOUNDS = {
    "initial_gauss_norm": 1e-1,
    "gauss_reduction_norm": 1e-4,
    "gauss_reduction_fraction": 1e-3,
    "gradient_norm": 1e-4,
    "line_step": 1e-4,
    "momentum_delta_norm": 1e-4,
}


def main() -> None:
    diagnostics = sm_physical_right_production_gauss_projection_diagnostics()
    print("QCA_SMv0 Session 46 - physical-right production Gauss relaxation")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "relaxed_gauss_norm" and value >= float(diagnostics.initial_gauss_norm):
            raise SystemExit(
                f"relaxed_gauss_norm={value:.3e} does not improve "
                f"initial_gauss_norm={float(diagnostics.initial_gauss_norm):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE46_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PROJECTION_PASS")


if __name__ == "__main__":
    main()
