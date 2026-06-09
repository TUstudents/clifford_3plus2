"""Session 04 certificate for local Higgs/Yukawa collision."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_higgs import higgs_yukawa_diagnostics


def main() -> None:
    diagnostics = higgs_yukawa_diagnostics()

    print("QCA_SMv0 Session 04 - local Higgs/Yukawa collision")
    print(f"yukawa_hermitian_residual: {float(diagnostics.yukawa_hermitian_residual):.3e}")
    print(f"zero_step_residual: {float(diagnostics.zero_step_residual):.3e}")
    print(f"zero_higgs_residual: {float(diagnostics.zero_higgs_residual):.3e}")
    print(f"norm_drift: {float(diagnostics.norm_drift):.3e}")
    print(f"chirality_flip_right_norm: {float(diagnostics.chirality_flip_right_norm):.3e}")
    print(f"massive_dispersion_residual: {float(diagnostics.massive_dispersion_residual):.3e}")
    print(f"jit_delta: {float(diagnostics.jit_delta):.3e}")
    print("verdict: QCA_SMV0_STAGE4_HIGGS_YUKAWA_PASS")


if __name__ == "__main__":
    main()
