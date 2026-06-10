"""Session 37 certificate for local Wilson-force replacement."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_force import (
    sm_physical_right_production_local_force_diagnostics,
)


THRESHOLDS = {
    "identity_force_norm": 1e-7,
    "pure_gauge_force_norm": 1e-6,
    "production_local_alias_residual": 1e-12,
    "covariance_residual": 5e-8,
    "ad_coordinate_residual": 2e-8,
    "legacy_fd_relative_residual": 8e-2,
}

LOWER_BOUNDS = {
    "nonflat_force_norm": 5e-5,
    "finite_difference_to_local_work_ratio": 300.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_local_force_diagnostics()
    print("QCA_SMv0 Session 37 - physical-right production local Wilson force")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE37_PHYSICAL_RIGHT_PRODUCTION_LOCAL_FORCE_PASS")


if __name__ == "__main__":
    main()
