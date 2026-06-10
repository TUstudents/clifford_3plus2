"""Session 40 certificate for local-force production profiling."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_profile import (
    sm_physical_right_production_local_profile_diagnostics,
)


THRESHOLDS = {
    "jit_eager_state_residual": 1e-6,
    "family_norm_drift": 2e-6,
    "sm_link_unitarity_residual": 2e-6,
    "higgs_link_unitarity_residual": 2e-6,
}

LOWER_BOUNDS = {
    "site_count": 2.0,
    "eager_profile_python_seconds": 0.0,
    "eager_profile_payload_key_count": 7.0,
    "jit_compile_seconds": 0.0,
    "jit_run_seconds": 0.0,
}


def main() -> None:
    diagnostics = sm_physical_right_production_local_profile_diagnostics()
    print("QCA_SMv0 Session 40 - physical-right production local-force profile")
    for field in diagnostics._fields:
        raw = getattr(diagnostics, field)
        value = float(raw)
        print(f"{field}: {value:.3e}")
        if field in {"eager_profile_all_finite", "final_all_finite"} and not bool(raw):
            raise SystemExit(f"{field} is false")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE40_PHYSICAL_RIGHT_PRODUCTION_LOCAL_PROFILE_PASS")


if __name__ == "__main__":
    main()

