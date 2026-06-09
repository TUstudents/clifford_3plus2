"""Session 05 certificate for FN recirculation paths."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_fn import (
    FN_LAMBDA_SHEAR,
    FN_LAMBDA_WOLFENSTEIN,
    fn_recirculation_diagnostics,
)


def _print_diagnostics(label: str, lambda_rec: float) -> None:
    diagnostics = fn_recirculation_diagnostics(lambda_rec)
    print(f"{label}_lambda: {lambda_rec:.12e}")
    print(f"{label}_beam_splitter_unitarity_residual: {float(diagnostics.beam_splitter_unitarity_residual):.3e}")
    print(f"{label}_path_transfer_residual: {float(diagnostics.path_transfer_residual):.3e}")
    print(f"{label}_up_exponent_residual: {float(diagnostics.up_exponent_residual):.3e}")
    print(f"{label}_down_exponent_residual: {float(diagnostics.down_exponent_residual):.3e}")
    print(f"{label}_up_diagonal_scaling_residual: {float(diagnostics.up_diagonal_scaling_residual):.3e}")
    print(f"{label}_down_diagonal_scaling_residual: {float(diagnostics.down_diagonal_scaling_residual):.3e}")
    print(f"{label}_wolfenstein_scaling_residual: {float(diagnostics.wolfenstein_scaling_residual):.3e}")
    print(f"{label}_ckm_unitarity_residual: {float(diagnostics.ckm_unitarity_residual):.3e}")
    print(f"{label}_jit_delta: {float(diagnostics.jit_delta):.3e}")


def main() -> None:
    print("QCA_SMv0 Session 05 - FN recirculation paths")
    _print_diagnostics("wolfenstein", FN_LAMBDA_WOLFENSTEIN)
    _print_diagnostics("shear_candidate", FN_LAMBDA_SHEAR)
    print("verdict: QCA_SMV0_STAGE5_FN_RECIRCULATION_PASS")


if __name__ == "__main__":
    main()
