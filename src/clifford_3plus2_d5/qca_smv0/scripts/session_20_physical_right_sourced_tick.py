"""Session 20 certificate for QCA_SMv0 physical-right sourced SM tick."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_sourced_tick import (
    sm_physical_right_sourced_tick_diagnostics,
)


THRESHOLDS = {
    "zero_source_force_norm": 5e-8,
    "force_covariance_residual": 5e-5,
    "gauss_covariance_residual": 5e-7,
    "gauss_zero_residual": 1e-7,
    "kick_reversibility_residual": 1e-8,
    "sm_link_unitarity_residual": 7e-7,
    "higgs_link_unitarity_residual": 7e-7,
    "family_norm_drift": 5e-6,
    "jit_delta_family_state": 2e-7,
    "jit_delta_sm_links": 1e-8,
    "jit_delta_higgs_links": 1e-8,
    "jit_delta_sm_momenta": 2e-7,
    "jit_delta_higgs_field": 1e-8,
    "jit_delta_higgs_momenta": 1e-8,
}

LOWER_BOUNDS = {
    "transport_force_difference_norm": 1e-2,
    "nonzero_source_force_norm": 1e-4,
    "kick_delta_norm": 1e-6,
    "higgs_field_delta_norm": 1e-6,
    "transport_tick_state_difference_norm": 1e-3,
}


def main() -> None:
    diagnostics = sm_physical_right_sourced_tick_diagnostics()
    print("QCA_SMv0 Session 20 - physical-right sourced SM tick")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE20_PHYSICAL_RIGHT_SOURCED_TICK_PASS")


if __name__ == "__main__":
    main()
