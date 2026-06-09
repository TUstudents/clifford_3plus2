"""Session 25 certificate for the physical-right production variational audit."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_variational import (
    sm_physical_right_production_variational_diagnostics,
)


THRESHOLDS = {
    "link_force_decomposition_residual": 1e-8,
    "higgs_force_decomposition_residual": 1e-8,
    "link_directional_derivative_residual": 1e-4,
    "higgs_real_directional_derivative_residual": 1e-4,
    "higgs_imag_directional_derivative_residual": 1e-4,
    "vacuum_link_force_norm": 1e-6,
    "vacuum_higgs_force_norm": 1e-6,
}

LOWER_BOUNDS = {
    "deterministic_link_force_norm": 1e-2,
    "deterministic_higgs_force_norm": 1e-2,
    "all_residuals_finite": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_variational_diagnostics()
    print("QCA_SMv0 Session 25 - physical-right production variational audit")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE25_PHYSICAL_RIGHT_PRODUCTION_VARIATIONAL_PASS")


if __name__ == "__main__":
    main()
