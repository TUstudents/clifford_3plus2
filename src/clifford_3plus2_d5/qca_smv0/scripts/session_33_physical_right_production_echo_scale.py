"""Session 33 certificate for echo-Gram scale stability."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_scale import (
    sm_physical_right_production_echo_scale_diagnostics,
)


THRESHOLDS = {
    "epsilon_ratio": 2.01,
    "small_base_roundtrip_residual": 1e-5,
    "large_base_roundtrip_residual": 1e-5,
    "min_norm_scale_ratio": 1.05,
    "max_norm_scale_ratio": 1.05,
    "min_eigenvalue_scale_ratio": 1.10,
    "max_eigenvalue_scale_ratio": 1.10,
    "condition_number_delta": 0.10,
    "offdiag_correlation_delta": 0.05,
    "max_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "small_epsilon": 9e-5,
    "large_epsilon": 1.9e-4,
    "epsilon_ratio": 1.99,
    "min_norm_scale_ratio": 0.95,
    "max_norm_scale_ratio": 0.95,
    "min_eigenvalue_scale_ratio": 0.90,
    "max_eigenvalue_scale_ratio": 0.90,
}


def main() -> None:
    diagnostics = sm_physical_right_production_echo_scale_diagnostics()
    print("QCA_SMv0 Session 33 - physical-right production echo-Gram scale")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE33_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_SCALE_PASS")


if __name__ == "__main__":
    main()
