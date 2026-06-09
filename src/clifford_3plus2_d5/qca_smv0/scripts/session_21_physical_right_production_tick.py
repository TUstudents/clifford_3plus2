"""Session 21 certificate for QCA_SMv0 physical-right production tick."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tick import (
    sm_physical_right_production_tick_diagnostics,
)


THRESHOLDS = {
    "zero_yukawa_state_reduction_residual": 2e-7,
    "zero_yukawa_higgs_reduction_residual": 2e-7,
    "zero_yukawa_higgs_momentum_reduction_residual": 2e-7,
    "zero_yukawa_sm_momentum_reduction_residual": 2e-7,
    "zero_yukawa_sm_link_reduction_residual": 2e-7,
    "zero_yukawa_higgs_link_reduction_residual": 2e-7,
    "zero_yukawa_source_norm": 1e-7,
    "source_kick_reversibility_residual": 1e-8,
    "production_family_norm_drift": 5e-6,
    "sm_link_unitarity_residual": 7e-7,
    "higgs_link_unitarity_residual": 7e-7,
    "jit_delta_family_state": 5e-7,
    "jit_delta_higgs_field": 2e-7,
    "jit_delta_higgs_momenta": 2e-7,
    "jit_delta_sm_links": 2e-7,
    "jit_delta_sm_momenta": 2e-7,
    "jit_delta_higgs_links": 2e-7,
}

LOWER_BOUNDS = {
    "nonzero_yukawa_source_norm": 1e-4,
    "production_state_delta_norm": 1e-6,
    "production_higgs_momentum_delta_norm": 1e-6,
    "transport_production_difference_norm": 1e-3,
    "physical_source_force_difference_norm": 1e-2,
}


def main() -> None:
    diagnostics = sm_physical_right_production_tick_diagnostics()
    print("QCA_SMv0 Session 21 - physical-right production tick")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE21_PHYSICAL_RIGHT_PRODUCTION_TICK_PASS")


if __name__ == "__main__":
    main()
