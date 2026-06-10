"""Session 44 certificate for production family-cone growth."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_cone import (
    sm_physical_right_production_cone_diagnostics,
)


THRESHOLDS = {
    "outside_step_cone_site_count": 1e-12,
    "outside_step_cone_max_norm": 1e-12,
    "radius_growth_residual": 1e-12,
}

LOWER_BOUNDS = {
    "site_count": 300.0,
    "step_one_support_count": 8.0,
    "step_two_support_count": 20.0,
    "step_three_support_count": 40.0,
    "support_count_growth_min_delta": 8.0,
    "max_response_norm": 1e-3,
    "min_detected_response_norm": 1e-8,
}


def main() -> None:
    diagnostics = sm_physical_right_production_cone_diagnostics()
    print("QCA_SMv0 Session 44 - physical-right production family cone")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE44_PHYSICAL_RIGHT_PRODUCTION_CONE_PASS")


if __name__ == "__main__":
    main()
