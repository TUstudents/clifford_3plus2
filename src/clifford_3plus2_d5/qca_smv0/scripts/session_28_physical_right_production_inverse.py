"""Session 28 certificate for the explicit physical-right production inverse."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_inverse import (
    sm_physical_right_production_inverse_diagnostics,
)


THRESHOLDS = {
    "inverse_roundtrip_residual": 5e-6,
    "forward_inverse_forward_residual": 5e-6,
    "inverse_family_norm_drift": 1e-5,
    "forward_family_norm_drift": 1e-5,
    "inverse_sm_link_unitarity_residual": 1e-5,
    "inverse_higgs_link_unitarity_residual": 1e-5,
    "jit_delta_family_state": 1e-5,
    "jit_delta_higgs": 1e-8,
    "jit_delta_higgs_momenta": 1e-8,
    "jit_delta_sm_links": 1e-8,
    "jit_delta_sm_momenta": 1e-6,
    "jit_delta_higgs_links": 1e-8,
}

LOWER_BOUNDS = {
    "naive_negative_residual": 1e-2,
    "explicit_inverse_improvement_ratio": 1e3,
}


def main() -> None:
    diagnostics = sm_physical_right_production_inverse_diagnostics()
    print("QCA_SMv0 Session 28 - explicit physical-right production inverse")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE28_PHYSICAL_RIGHT_PRODUCTION_EXPLICIT_INVERSE_PASS")


if __name__ == "__main__":
    main()
