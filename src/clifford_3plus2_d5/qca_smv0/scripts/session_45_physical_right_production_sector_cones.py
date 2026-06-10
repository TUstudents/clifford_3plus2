"""Session 45 certificate for all-sector two-tick production cones."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_sector_cones import (
    sm_physical_right_production_sector_cones_diagnostics,
)


THRESHOLDS = {
    "outside_step_cone_site_count": 1e-12,
    "outside_step_cone_max_norm": 1e-12,
    "radius_overshoot": 1e-12,
}

LOWER_BOUNDS = {
    "site_count": 120.0,
    "sector_count": 6.0,
    "horizon_steps": 2.0,
    "family_support_count": 8.0,
    "higgs_support_count": 8.0,
    "higgs_momentum_support_count": 1.0,
    "sm_link_support_count": 1.0,
    "sm_momentum_support_count": 1.0,
    "higgs_link_support_count": 1.0,
    "min_support_count": 1.0,
    "max_response_norm": 1e-3,
    "min_detected_response_norm": 1e-8,
}


def main() -> None:
    diagnostics = sm_physical_right_production_sector_cones_diagnostics()
    print("QCA_SMv0 Session 45 - physical-right production all-sector cones")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "max_support_radius" and value > float(diagnostics.horizon_steps):
            raise SystemExit(
                f"max_support_radius={value:.3e} exceeds horizon {float(diagnostics.horizon_steps):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE45_PHYSICAL_RIGHT_PRODUCTION_SECTOR_CONES_PASS")


if __name__ == "__main__":
    main()
