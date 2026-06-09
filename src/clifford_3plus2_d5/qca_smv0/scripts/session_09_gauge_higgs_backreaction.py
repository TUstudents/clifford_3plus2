"""Session 09 certificate for coupled gauge-Higgs backreaction."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_gauge_higgs import sm_gauge_higgs_backreaction_diagnostics


def main() -> None:
    diagnostics = sm_gauge_higgs_backreaction_diagnostics()
    print("QCA_SMv0 Session 09 - gauge-Higgs backreaction")
    for name, value in diagnostics._asdict().items():
        print(f"{name}: {float(value):.3e}")

    assert diagnostics.projection_roundtrip_residual < 2e-6
    assert diagnostics.higgs_link_force_vacuum_norm < 2e-6
    assert diagnostics.higgs_link_force_nonzero_norm > 1e-5
    assert diagnostics.higgs_link_force_covariance_residual < 5e-6
    assert diagnostics.charge_covariance_residual < 2e-6
    assert diagnostics.gauss_covariance_residual < 5e-6
    assert diagnostics.gauss_vacuum_residual < 1e-7
    assert diagnostics.sm_embedding_roundtrip_residual < 1e-8
    assert diagnostics.coupled_link_unitarity_residual < 5e-7
    assert diagnostics.coupled_hamiltonian_drift < 1e-5
    assert diagnostics.coupled_reversibility_residual < 1e-5
    assert diagnostics.jit_delta_field < 1e-8
    assert diagnostics.jit_delta_links < 1e-8
    assert diagnostics.jit_delta_higgs_momenta < 1e-8
    assert diagnostics.jit_delta_link_momenta < 1e-8
    print("verdict: QCA_SMV0_STAGE9_GAUGE_HIGGS_BACKREACTION_PASS")


if __name__ == "__main__":
    main()
