"""Session 16 certificate for the QCA_SMv0 gauge-convention bridge audit."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_gauge_convention_bridge import sm_gauge_convention_bridge_diagnostics


THRESHOLDS = {
    "generator_helper_residual": 1e-7,
    "left_doublet_generator_residual": 1e-7,
    "right_singlet_su2_residual": 1e-7,
    "right_hypercharge_conjugation_residual": 1e-7,
    "physical_yukawa_energy_covariance_residual": 1e-7,
    "jit_delta_physical_covariance": 1e-8,
    "jit_delta_transport_noninvariance": 1e-8,
}

LOWER_BOUNDS = {
    "full_generator_difference_norm": 1.0,
    "hypercharge_spectral_mismatch": 0.1,
    "transport_yukawa_energy_noninvariance_residual": 1e-4,
}


def main() -> None:
    diagnostics = sm_gauge_convention_bridge_diagnostics()
    print("QCA_SMv0 Session 16 - gauge convention bridge audit")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE16_GAUGE_CONVENTION_BRIDGE_PASS")


if __name__ == "__main__":
    main()
