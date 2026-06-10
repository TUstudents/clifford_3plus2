"""Session 10 certificate for local fermion/Higgs backreaction."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_fermion_higgs_backreaction_diagnostics


def main() -> None:
    diagnostics = sm_fermion_higgs_backreaction_diagnostics()
    print("QCA_SMv0 Session 10 - fermion/Higgs backreaction")
    for name, value in diagnostics._asdict().items():
        print(f"{name}: {float(value):.3e}")

    assert diagnostics.fn_recirculated_source_residual < 1e-8
    assert diagnostics.energy_reality_residual < 1e-8
    assert diagnostics.zero_state_source_norm < 1e-8
    assert diagnostics.nonzero_source_norm > 1e-4
    assert diagnostics.energy_gauge_invariance_residual < 5e-7
    assert diagnostics.source_covariance_residual < 5e-7
    assert diagnostics.kick_delta_norm > 1e-6
    assert diagnostics.kick_reversibility_residual < 1e-8
    assert diagnostics.collision_norm_drift < 1e-6
    assert diagnostics.source_after_collision_norm > 1e-4
    assert diagnostics.jit_delta_source < 1e-8
    assert diagnostics.jit_delta_kick < 1e-8
    print("verdict: QCA_SMV0_STAGE10_FERMION_HIGGS_BACKREACTION_PASS")


if __name__ == "__main__":
    main()
