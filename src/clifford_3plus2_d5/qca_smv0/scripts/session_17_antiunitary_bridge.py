"""Session 17 certificate for the QCA_SMv0 antiunitary singlet bridge."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import sm_antiunitary_bridge_diagnostics


THRESHOLDS = {
    "left_linear_generator_residual": 1e-7,
    "right_antilinear_generator_residual": 1e-7,
    "electroweak_yukawa_slice_residual": 1e-7,
    "finite_bridge_residual": 2e-7,
    "finite_bridge_unitarity_residual": 5e-7,
    "full_physical_yukawa_energy_covariance_residual": 1e-7,
    "jit_delta_physical_covariance": 1e-8,
    "jit_delta_transport_noninvariance": 1e-8,
}

LOWER_BOUNDS = {
    "transport_physical_generator_difference_norm": 1.0,
    "full_transport_yukawa_energy_noninvariance_residual": 1e-4,
}


def main() -> None:
    diagnostics = sm_antiunitary_bridge_diagnostics()
    print("QCA_SMv0 Session 17 - antiunitary singlet bridge")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE17_ANTIUNITARY_BRIDGE_PASS")


if __name__ == "__main__":
    main()
