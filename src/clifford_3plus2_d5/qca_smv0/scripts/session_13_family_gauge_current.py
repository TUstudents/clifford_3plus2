"""Session 13 certificate for family-summed BCC fermion gauge current."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_family_gauge import sm_family_fermion_gauge_current_diagnostics


def main() -> None:
    diagnostics = sm_family_fermion_gauge_current_diagnostics()
    print("QCA_SMv0 Session 13 - family-summed BCC fermion gauge current")
    for name, value in diagnostics._asdict().items():
        print(f"{name}: {float(value):.3e}")

    assert diagnostics.streaming_energy_reality_residual < 1e-8
    assert diagnostics.single_family_energy_reduction_residual < 2e-7
    assert diagnostics.single_family_current_reduction_residual < 1e-5
    assert diagnostics.single_family_charge_reduction_residual < 1e-8
    assert diagnostics.zero_state_current_norm < 1e-8
    assert diagnostics.nonzero_current_norm > 1e-4
    assert diagnostics.streaming_energy_gauge_invariance_residual < 5e-7
    assert diagnostics.current_covariance_residual < 2e-5
    assert diagnostics.charge_covariance_residual < 2e-6
    assert diagnostics.gauss_covariance_residual < 2e-5
    assert diagnostics.momentum_kick_delta_norm > 1e-6
    assert diagnostics.momentum_kick_reversibility_residual < 1e-8
    assert diagnostics.kicked_link_unitarity_residual < 7e-7
    assert diagnostics.spectator_norm_drift_after_kick < 1e-5
    assert diagnostics.jit_delta_current < 1e-5
    assert diagnostics.jit_delta_transport < 1e-8
    print("verdict: QCA_SMV0_STAGE13_FAMILY_GAUGE_CURRENT_PASS")


if __name__ == "__main__":
    main()
