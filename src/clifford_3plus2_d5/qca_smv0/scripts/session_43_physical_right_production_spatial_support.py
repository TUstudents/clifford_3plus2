"""Session 43 certificate for one-tick production spatial support."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_spatial_support import (
    sm_physical_right_production_spatial_support_diagnostics,
)


THRESHOLDS = {
    "outside_predicted_radius_site_count": 1e-12,
    "outside_predicted_radius_max_norm": 1e-12,
}

LOWER_BOUNDS = {
    "site_count": 300.0,
    "family_support_count": 4.0,
    "higgs_support_count": 1.0,
    "higgs_momentum_support_count": 1.0,
    "sm_link_support_count": 1.0,
    "sm_momentum_support_count": 1.0,
    "higgs_link_support_count": 1.0,
    "max_response_norm": 1e-3,
    "min_detected_response_norm": 1e-8,
}


def main() -> None:
    diagnostics = sm_physical_right_production_spatial_support_diagnostics()
    print("QCA_SMv0 Session 43 - physical-right production one-tick spatial support")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field == "max_measured_support_radius" and value > float(diagnostics.predicted_tick_radius):
            raise SystemExit(
                f"max_measured_support_radius={value:.3e} exceeds predicted radius "
                f"{float(diagnostics.predicted_tick_radius):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE43_PHYSICAL_RIGHT_PRODUCTION_SPATIAL_SUPPORT_PASS")


if __name__ == "__main__":
    main()
