"""Session 31 certificate for finite tangent-response echo."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tangent import (
    sm_physical_right_production_tangent_diagnostics,
)


THRESHOLDS = {
    "base_roundtrip_residual": 1e-5,
    "sm_momentum_echo_norm": 2e-4,
    "higgs_momentum_echo_norm": 2e-4,
    "combined_echo_norm": 2e-4,
    "superposition_residual": 2e-6,
    "superposition_relative_residual": 2e-2,
    "combined_inverse_sm_link_unitarity_residual": 1e-5,
    "combined_inverse_higgs_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "steps": 2,
    "perturbation_size": 9e-5,
    "sm_momentum_echo_norm": 5e-5,
    "higgs_momentum_echo_norm": 5e-5,
    "combined_echo_norm": 5e-5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_tangent_diagnostics()
    print("QCA_SMv0 Session 31 - physical-right production tangent echo")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE31_PHYSICAL_RIGHT_PRODUCTION_TANGENT_ECHO_PASS")


if __name__ == "__main__":
    main()
