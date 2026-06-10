"""Session 26 certificate for the physical-right production refinement audit."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_refinement import (
    sm_physical_right_production_refinement_diagnostics,
)


THRESHOLDS = {
    "total_time_match_residual": 1e-12,
    "base_energy_drift_abs": 1e-5,
    "refined_energy_drift_abs": 5e-4,
    "base_family_norm_drift": 1e-5,
    "refined_family_norm_drift": 1e-4,
    "refined_sm_link_unitarity_residual": 1e-5,
    "refined_higgs_link_unitarity_residual": 1e-5,
    "vacuum_refined_energy_drift_abs": 1e-10,
}

LOWER_BOUNDS = {
    "drift_refinement_ratio": 5.0,
    "refinement_limitation_detected": 0.5,
    "refined_energy_drift_controlled": 0.5,
    "histories_all_finite": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_refinement_diagnostics()
    print("QCA_SMv0 Session 26 - physical-right production refinement limitation audit")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE26_PHYSICAL_RIGHT_PRODUCTION_REFINEMENT_LIMITATION_PASS")


if __name__ == "__main__":
    main()
