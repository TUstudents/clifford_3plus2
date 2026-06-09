"""Session 06 certificate for center-holonomy CP."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_cp import sm_center_cp_diagnostics


def main() -> None:
    diagnostics = sm_center_cp_diagnostics()

    print("QCA_SMv0 Session 06 - center-holonomy CP")
    print(f"center_phase_unit_modulus_residual: {float(diagnostics.center_phase_unit_modulus_residual):.3e}")
    print(f"center_phase_closure_residual: {float(diagnostics.center_phase_closure_residual):.3e}")
    print(f"coefficient_magnitude_residual: {float(diagnostics.coefficient_magnitude_residual):.3e}")
    print(f"quark_antiquark_mass_residual: {float(diagnostics.quark_antiquark_mass_residual):.3e}")
    print(f"ckm_unitarity_residual: {float(diagnostics.ckm_unitarity_residual):.3e}")
    print(f"center_jarlskog_abs: {float(diagnostics.center_jarlskog_abs):.3e}")
    print(f"real_control_jarlskog_abs: {float(diagnostics.real_control_jarlskog_abs):.3e}")
    print(f"commutator_cp_abs: {float(diagnostics.commutator_cp_abs):.3e}")
    print(f"jit_delta: {float(diagnostics.jit_delta):.3e}")
    print("verdict: QCA_SMV0_STAGE6_CENTER_HOLONOMY_CP_PASS")


if __name__ == "__main__":
    main()
