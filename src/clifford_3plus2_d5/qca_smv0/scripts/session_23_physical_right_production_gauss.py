"""Session 23 certificate for QCA_SMv0 physical-right production Gauss monitor."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss_diagnostics,
)


THRESHOLDS = {
    "vacuum_initial_gauss_norm": 1e-8,
    "vacuum_final_gauss_norm": 1e-8,
    "vacuum_family_norm": 1e-8,
    "rollout_family_norm_drift": 1e-5,
    "max_sm_link_unitarity_residual": 8e-7,
    "max_higgs_link_unitarity_residual": 8e-7,
}

LOWER_BOUNDS = {
    "deterministic_initial_gauss_norm": 1e-1,
    "deterministic_final_gauss_norm": 1e-1,
    "deterministic_max_gauss_norm": 1e-1,
    "deterministic_gauss_delta_norm": 1e-3,
    "zero_yukawa_final_gauss_difference_norm": 5e-7,
    "history_count": 2.5,
    "history_all_finite": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_gauss_diagnostics()
    print("QCA_SMv0 Session 23 - physical-right production Gauss monitor")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE23_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PASS")


if __name__ == "__main__":
    main()
