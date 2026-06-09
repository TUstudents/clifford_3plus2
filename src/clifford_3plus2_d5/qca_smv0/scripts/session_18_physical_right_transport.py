"""Session 18 certificate for QCA_SMv0 physical-right bridged transport."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import sm_physical_right_transport_diagnostics


THRESHOLDS = {
    "identity_bridge_residual": 1e-7,
    "finite_bridge_unitarity_residual": 7e-7,
    "bridged_link_covariance_residual": 8e-7,
    "identity_transport_reduction_residual": 1e-7,
    "physical_right_transport_covariance_residual": 8e-7,
    "physical_right_norm_drift": 1e-6,
    "jit_delta_transport": 2e-7,
}

LOWER_BOUNDS = {
    "transport_kernel_difference_norm": 1e-3,
}


def main() -> None:
    diagnostics = sm_physical_right_transport_diagnostics()
    print("QCA_SMv0 Session 18 - physical-right bridged transport")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE18_PHYSICAL_RIGHT_TRANSPORT_PASS")


if __name__ == "__main__":
    main()
