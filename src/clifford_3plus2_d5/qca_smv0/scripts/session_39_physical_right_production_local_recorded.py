"""Session 39 certificate for multi-step local-force recorded rollout."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_recorded import (
    sm_physical_right_production_local_recorded_diagnostics,
)


THRESHOLDS = {
    "loop_scan_final_residual": 1e-6,
    "loop_scan_observation_residual": 1e-6,
    "family_norm_drift": 3e-6,
    "max_family_norm_drift": 3e-6,
    "max_sm_link_unitarity_residual": 2e-6,
    "max_higgs_link_unitarity_residual": 2e-6,
}

LOWER_BOUNDS = {
    "site_count": 4.0,
    "step_count": 2.0,
    "record_count": 3.0,
    "higgs_field_total_delta_norm": 1e-4,
    "sm_link_total_delta_norm": 1e-4,
    "sm_momentum_total_delta_norm": 1e-5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_local_recorded_diagnostics()
    print("QCA_SMv0 Session 39 - physical-right production local recorded rollout")
    for field in diagnostics._fields:
        raw = getattr(diagnostics, field)
        value = float(raw)
        print(f"{field}: {value:.3e}")
        if field in {"loop_all_finite", "scan_all_finite"} and not bool(raw):
            raise SystemExit(f"{field} is false")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE39_PHYSICAL_RIGHT_PRODUCTION_LOCAL_RECORDED_PASS")


if __name__ == "__main__":
    main()

