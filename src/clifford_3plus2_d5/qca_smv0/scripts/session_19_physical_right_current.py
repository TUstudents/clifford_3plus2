"""Session 19 certificate for QCA_SMv0 physical-right fermion current."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import sm_physical_right_current_diagnostics


THRESHOLDS = {
    "streaming_energy_reality_residual": 1e-7,
    "zero_state_current_norm": 1e-7,
    "streaming_energy_covariance_residual": 1e-6,
    "current_covariance_residual": 7e-6,
    "charge_covariance_residual": 3e-7,
    "gauss_covariance_residual": 3e-7,
    "momentum_kick_reversibility_residual": 3e-10,
    "kicked_link_unitarity_residual": 7e-7,
    "spectator_norm_drift_after_kick": 8e-6,
    "jit_delta_current": 7e-6,
    "jit_delta_transport": 2e-7,
}

LOWER_BOUNDS = {
    "nonzero_current_norm": 1e-2,
    "transport_current_difference_norm": 1e-2,
    "momentum_kick_delta_norm": 1e-4,
}


def main() -> None:
    diagnostics = sm_physical_right_current_diagnostics()
    print("QCA_SMv0 Session 19 - physical-right fermion current")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value <= LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE19_PHYSICAL_RIGHT_CURRENT_PASS")


if __name__ == "__main__":
    main()
