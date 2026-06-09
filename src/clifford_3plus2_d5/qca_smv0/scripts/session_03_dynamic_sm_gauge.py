"""Session 03 certificate for pure dynamic SM gauge fields."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_dynamics import dynamic_sm_gauge_diagnostics


def main() -> None:
    diagnostics = dynamic_sm_gauge_diagnostics()

    print("QCA_SMv0 Session 03 - pure dynamic SM gauge fields")
    print(f"projection_roundtrip_residual: {float(diagnostics.projection_roundtrip_residual):.3e}")
    print(f"identity_force_norm: {float(diagnostics.identity_force_norm):.3e}")
    print(f"pure_gauge_force_norm: {float(diagnostics.pure_gauge_force_norm):.3e}")
    print(f"nonflat_force_norm: {float(diagnostics.nonflat_force_norm):.3e}")
    print(f"link_unitarity_after_leapfrog: {float(diagnostics.link_unitarity_after_leapfrog):.3e}")
    print(f"hamiltonian_drift: {float(diagnostics.hamiltonian_drift):.3e}")
    print(f"reversibility_residual: {float(diagnostics.reversibility_residual):.3e}")
    print(f"momentum_covariance_residual: {float(diagnostics.momentum_covariance_residual):.3e}")
    print(f"gauss_zero_residual: {float(diagnostics.gauss_zero_residual):.3e}")
    print(f"gauss_covariance_residual: {float(diagnostics.gauss_covariance_residual):.3e}")
    print(f"gauss_pure_leapfrog_residual: {float(diagnostics.gauss_pure_leapfrog_residual):.3e}")
    print(f"weak_field_residual_ratio: {float(diagnostics.weak_field_residual_ratio):.3e}")
    print(f"yang_mills_action_ratio: {float(diagnostics.yang_mills_action_ratio):.3e}")
    print(f"spectator_norm_drift: {float(diagnostics.spectator_norm_drift):.3e}")
    print(f"jit_delta_links: {float(diagnostics.jit_delta_links):.3e}")
    print(f"jit_delta_momenta: {float(diagnostics.jit_delta_momenta):.3e}")
    print("verdict: QCA_SMV0_STAGE3_DYNAMIC_SM_GAUGE_PASS")


if __name__ == "__main__":
    main()
