"""Session 36 certificate for dense production workload scaling."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_workload import (
    sm_physical_right_production_workload_diagnostics,
)


THRESHOLDS = {
    "state_mib": 4.0,
}

LOWER_BOUNDS = {
    "site_count": 27,
    "state_bytes_per_site": 60_000,
    "sm_link_state_fraction": 0.90,
    "wilson_force_coordinate_count": 2_000,
    "finite_difference_action_evaluations": 5_000,
    "finite_difference_plaquette_holonomies": 800_000,
    "quadratic_work_ratio_to_single_site": 700.0,
    "finite_difference_to_local_work_ratio": 5_000.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_workload_diagnostics()
    print("QCA_SMv0 Session 36 - physical-right production dense workload")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE36_PHYSICAL_RIGHT_PRODUCTION_WORKLOAD_PASS")


if __name__ == "__main__":
    main()
