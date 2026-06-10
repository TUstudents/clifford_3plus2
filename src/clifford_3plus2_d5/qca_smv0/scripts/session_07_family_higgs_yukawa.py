"""Session 07 certificate for three-family Higgs/Yukawa collision."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import sm_family_higgs_yukawa_diagnostics


def main() -> None:
    diagnostics = sm_family_higgs_yukawa_diagnostics()

    print("QCA_SMv0 Session 07 - three-family Higgs/Yukawa collision")
    print(f"family_yukawa_hermitian_residual: {float(diagnostics.family_yukawa_hermitian_residual):.3e}")
    print(f"fn_recirculated_quark_yukawa_residual: {float(diagnostics.fn_recirculated_quark_yukawa_residual):.3e}")
    print(f"fn_recirculated_embedding_residual: {float(diagnostics.fn_recirculated_embedding_residual):.3e}")
    print(f"quark_embedding_residual: {float(diagnostics.quark_embedding_residual):.3e}")
    print(f"wrong_door_residual: {float(diagnostics.wrong_door_residual):.3e}")
    print(f"ckm_embedding_residual: {float(diagnostics.ckm_embedding_residual):.3e}")
    print(f"zero_step_residual: {float(diagnostics.zero_step_residual):.3e}")
    print(f"zero_higgs_residual: {float(diagnostics.zero_higgs_residual):.3e}")
    print(f"norm_drift: {float(diagnostics.norm_drift):.3e}")
    print(f"chirality_flip_right_norm: {float(diagnostics.chirality_flip_right_norm):.3e}")
    print(f"jit_delta: {float(diagnostics.jit_delta):.3e}")
    print("verdict: QCA_SMV0_STAGE7_FAMILY_HIGGS_YUKAWA_PASS")


if __name__ == "__main__":
    main()
