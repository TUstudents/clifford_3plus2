"""Session 24 certificate for QCA_SMv0 physical-right production energy monitor."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_energy import (
    sm_physical_right_production_energy_diagnostics,
)


THRESHOLDS = {
    "vacuum_initial_total_energy_abs": 1e-7,
    "vacuum_final_total_energy_abs": 1e-7,
    "deterministic_total_energy_delta_abs": 5e-6,
    "rollout_family_norm_drift": 1e-5,
}

LOWER_BOUNDS = {
    "deterministic_initial_total_energy": 1.0,
    "deterministic_final_total_energy": 1.0,
    "deterministic_max_total_energy_abs": 1.0,
    "deterministic_gauge_energy_positive": 1e-8,
    "deterministic_higgs_energy_positive": 1e-3,
    "deterministic_streaming_energy_abs": 1.0,
    "deterministic_yukawa_energy_abs": 1e-3,
    "zero_yukawa_final_total_energy_difference_abs": 1e-3,
    "history_count": 2.5,
    "history_all_finite": 0.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_energy_diagnostics()
    print("QCA_SMv0 Session 24 - physical-right production energy monitor")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE24_PHYSICAL_RIGHT_PRODUCTION_ENERGY_PASS")


if __name__ == "__main__":
    main()
