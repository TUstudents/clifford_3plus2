"""Session 05 certificate for FN recirculation paths."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_fn import (
    FN_LAMBDA_SHEAR,
    FN_LAMBDA_WOLFENSTEIN,
    fn_recirculation_diagnostics,
)


THRESHOLDS = {
    "beam_splitter_unitarity_residual": 1e-7,
    "path_transfer_residual": 1e-7,
    "up_network_unitarity_residual": 1e-6,
    "down_network_unitarity_residual": 1e-6,
    "up_network_transfer_residual": 1e-7,
    "down_network_transfer_residual": 1e-7,
    "up_visible_readout_residual": 1e-10,
    "down_visible_readout_residual": 1e-10,
    "up_unitary_dilation_residual": 2e-6,
    "down_unitary_dilation_residual": 2e-6,
    "up_unitary_dilation_transfer_residual": 1e-7,
    "down_unitary_dilation_transfer_residual": 1e-7,
    "up_collision_norm_drift": 1e-6,
    "down_collision_norm_drift": 1e-6,
    "up_collision_transfer_residual": 1e-7,
    "down_collision_transfer_residual": 1e-7,
    "up_exponent_residual": 1e-8,
    "down_exponent_residual": 1e-8,
    "up_diagonal_scaling_residual": 1e-10,
    "down_diagonal_scaling_residual": 1e-10,
    "wolfenstein_scaling_residual": 1e-10,
    "ckm_unitarity_residual": 2e-6,
    "jit_delta": 1e-10,
}


def _print_diagnostics(label: str, lambda_rec: float) -> None:
    diagnostics = fn_recirculation_diagnostics(lambda_rec)
    print(f"{label}_lambda: {lambda_rec:.12e}")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{label}_{field}: {value:.3e}")
        if value >= THRESHOLDS[field]:
            raise SystemExit(f"{label}_{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")


def main() -> None:
    print("QCA_SMv0 Session 05 - FN recirculation paths")
    _print_diagnostics("wolfenstein", FN_LAMBDA_WOLFENSTEIN)
    _print_diagnostics("shear_candidate", FN_LAMBDA_SHEAR)
    print("verdict: QCA_SMV0_STAGE5_FN_RECIRCULATION_PASS")


if __name__ == "__main__":
    main()
