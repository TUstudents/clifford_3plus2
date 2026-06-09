"""Session 12 certificate for the coupled sourced SM tick."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_sourced_tick import sm_sourced_tick_diagnostics


def main() -> None:
    diagnostics = sm_sourced_tick_diagnostics()
    print("QCA_SMv0 Session 12 - coupled sourced SM tick")
    for name, value in diagnostics._asdict().items():
        print(f"{name}: {float(value):.3e}")

    assert diagnostics.zero_source_force_norm < 5e-8
    assert diagnostics.nonzero_source_force_norm > 1e-4
    assert diagnostics.force_covariance_residual < 5e-5
    assert diagnostics.gauss_covariance_residual < 2e-6
    assert diagnostics.gauss_zero_residual < 1e-7
    assert diagnostics.kick_delta_norm > 1e-6
    assert diagnostics.kick_reversibility_residual < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 7e-7
    assert diagnostics.higgs_link_unitarity_residual < 7e-7
    assert diagnostics.fermion_norm_drift < 5e-6
    assert diagnostics.higgs_field_delta_norm > 1e-6
    assert diagnostics.jit_delta_state < 2e-7
    assert diagnostics.jit_delta_sm_links < 1e-8
    assert diagnostics.jit_delta_higgs_links < 1e-8
    assert diagnostics.jit_delta_sm_momenta < 2e-7
    assert diagnostics.jit_delta_higgs_field < 1e-8
    assert diagnostics.jit_delta_higgs_momenta < 1e-8
    print("verdict: QCA_SMV0_STAGE12_SOURCED_SM_TICK_PASS")


if __name__ == "__main__":
    main()
