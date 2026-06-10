"""Session 41 certificate for local-force spatial support."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_force_support import (
    sm_physical_right_production_force_support_diagnostics,
)


THRESHOLDS = {
    "outside_predicted_radius_max_norm": 1e-12,
    "outside_predicted_radius_site_count": 1e-12,
}

LOWER_BOUNDS = {
    "site_count": 300.0,
    "support_site_count": 10.0,
    "max_force_site_norm": 1e-5,
    "min_support_site_norm": 1e-6,
}


def main() -> None:
    diagnostics = sm_physical_right_production_force_support_diagnostics()
    print("QCA_SMv0 Session 41 - physical-right production local-force support")
    for field in diagnostics._fields:
        raw = getattr(diagnostics, field)
        value = float(raw)
        print(f"{field}: {value:.3e}")
        if field == "center_site_supported" and not bool(raw):
            raise SystemExit("center_site_supported is false")
        if field == "measured_support_radius" and value > float(diagnostics.predicted_force_radius):
            raise SystemExit(
                f"measured_support_radius={value:.3e} exceeds predicted radius "
                f"{float(diagnostics.predicted_force_radius):.3e}",
            )
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE41_PHYSICAL_RIGHT_PRODUCTION_FORCE_SUPPORT_PASS")


if __name__ == "__main__":
    main()

