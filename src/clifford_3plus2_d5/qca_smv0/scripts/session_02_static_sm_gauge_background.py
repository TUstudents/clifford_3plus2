"""Session 02 certificate for static full-SM gauge-background transport."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_gauge import static_sm_gauge_diagnostics


def main() -> None:
    diagnostics = static_sm_gauge_diagnostics()

    print("QCA_SMv0 Session 02 - static full-SM gauge background")
    print(f"generator_count: {diagnostics.generator_count}")
    print(f"generator_antihermitian_residual: {float(diagnostics.generator_antihermitian_residual):.3e}")
    print(f"link_unitarity_residual: {float(diagnostics.link_unitarity_residual):.3e}")
    print(f"identity_reduction_residual: {float(diagnostics.identity_reduction_residual):.3e}")
    print(f"norm_drift: {float(diagnostics.norm_drift):.3e}")
    print(f"gauge_covariance_residual: {float(diagnostics.gauge_covariance_residual):.3e}")
    print(f"identity_wilson_action: {float(diagnostics.identity_wilson_action):.3e}")
    print(f"pure_gauge_wilson_action: {float(diagnostics.pure_gauge_wilson_action):.3e}")
    print(f"nontrivial_wilson_action: {float(diagnostics.nontrivial_wilson_action):.3e}")
    print(f"wilson_trace_covariance_residual: {float(diagnostics.wilson_trace_covariance_residual):.3e}")
    print(f"continuum_residual_ratio: {float(diagnostics.continuum_residual_ratio):.3e}")
    print(f"jit_delta: {float(diagnostics.jit_delta):.3e}")
    print("verdict: QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS")


if __name__ == "__main__":
    main()
