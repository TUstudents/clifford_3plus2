"""Session 35 certificate for finite-stencil production locality."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_stencil import (
    sm_physical_right_production_stencil_diagnostics,
)


THRESHOLDS = {
    "bcc_transport_radius": 1,
    "higgs_force_radius": 1,
    "two_hop_force_radius": 2,
    "production_tick_radius": 2,
    "inverse_one_tick_radius": 2,
    "inverse_two_tick_radius": 4,
    "radius_growth_per_tick": 2.0,
    "bcc_inverse_closure_residual": 0,
}

LOWER_BOUNDS = {
    "bcc_hop_count": 8,
    "plaquette_pair_count": 6,
    "one_tick_support_count": 8,
    "two_tick_support_count": 8,
    "origin_in_tick_stencil": 1,
}


def main() -> None:
    diagnostics = sm_physical_right_production_stencil_diagnostics()
    print("QCA_SMv0 Session 35 - physical-right production finite stencil")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value > THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE35_PHYSICAL_RIGHT_PRODUCTION_STENCIL_PASS")


if __name__ == "__main__":
    main()
