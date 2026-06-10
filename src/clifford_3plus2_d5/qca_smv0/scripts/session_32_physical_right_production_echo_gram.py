"""Session 32 certificate for the production echo Gram matrix."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_gram import (
    sm_physical_right_production_echo_gram_diagnostics,
)


THRESHOLDS = {
    "base_roundtrip_residual": 1e-5,
    "min_echo_norm": 2e-4,
    "max_echo_norm": 2e-4,
    "gram_symmetry_residual": 1e-12,
    "gram_max_eigenvalue": 1e-6,
    "gram_condition_number": 10.0,
    "max_offdiag_correlation": 0.1,
    "max_inverse_sm_link_unitarity_residual": 1e-5,
    "max_inverse_higgs_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "basis_count": 3,
    "perturbation_size": 9e-5,
    "min_echo_norm": 5e-5,
    "max_echo_norm": 5e-5,
    "gram_min_eigenvalue": 1e-10,
}


def main() -> None:
    diagnostics = sm_physical_right_production_echo_gram_diagnostics()
    print("QCA_SMv0 Session 32 - physical-right production echo Gram")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE32_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_PASS")


if __name__ == "__main__":
    main()
