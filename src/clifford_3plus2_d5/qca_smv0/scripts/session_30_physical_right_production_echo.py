"""Session 30 certificate for the production Loschmidt echo diagnostic."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo import (
    sm_physical_right_production_echo_diagnostics,
)


THRESHOLDS = {
    "base_roundtrip_residual": 1e-5,
    "apex_perturbation_norm": 2e-4,
    "double_apex_perturbation_norm": 4e-4,
    "echo_residual": 2e-4,
    "double_echo_residual": 4e-4,
    "echo_amplification": 4.0,
    "double_echo_ratio": 2.5,
    "double_echo_linearity_residual": 0.25,
    "perturbed_inverse_sm_link_unitarity_residual": 1e-5,
    "perturbed_inverse_higgs_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "steps": 2,
    "perturbation_size": 9e-5,
    "apex_perturbation_norm": 5e-5,
    "double_apex_perturbation_norm": 1e-4,
    "echo_residual": 5e-5,
    "double_echo_residual": 1e-4,
    "echo_amplification": 0.25,
    "double_echo_ratio": 1.5,
}


def main() -> None:
    diagnostics = sm_physical_right_production_echo_diagnostics()
    print("QCA_SMv0 Session 30 - physical-right production Loschmidt echo")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE30_PHYSICAL_RIGHT_PRODUCTION_LOSCHMIDT_ECHO_PASS")


if __name__ == "__main__":
    main()
