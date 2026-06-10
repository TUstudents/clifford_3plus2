"""Session 27 certificate for the physical-right production adjoint audit."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_adjoint import (
    sm_physical_right_production_adjoint_diagnostics,
)


THRESHOLDS = {
    "transport_adjoint_residual": 5e-6,
    "frozen_fermion_stage_adjoint_residual": 5e-6,
    "frozen_fermion_norm_drift": 1e-5,
    "local_collision_inverse_residual": 5e-6,
    "forward_sm_link_unitarity_residual": 1e-5,
    "naive_negative_sm_link_unitarity_residual": 1e-5,
    "forward_higgs_link_unitarity_residual": 1e-5,
    "naive_negative_higgs_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "naive_negative_tick_residual": 1e-2,
    "naive_negative_tick_limitation_detected": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_adjoint_diagnostics()
    print("QCA_SMv0 Session 27 - physical-right production adjoint audit")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE27_PHYSICAL_RIGHT_PRODUCTION_ADJOINT_LIMITATION_PASS")


if __name__ == "__main__":
    main()
