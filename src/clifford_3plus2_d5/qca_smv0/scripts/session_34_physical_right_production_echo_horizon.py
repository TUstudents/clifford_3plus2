"""Session 34 certificate for finite-horizon echo spectrum."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_horizon import (
    sm_physical_right_production_echo_horizon_diagnostics,
)


THRESHOLDS = {
    "short_base_roundtrip_residual": 1e-5,
    "long_base_roundtrip_residual": 1e-5,
    "short_min_gain": 1.05,
    "short_max_gain": 1.05,
    "long_min_gain": 1.05,
    "long_max_gain": 1.05,
    "min_gain_growth_ratio": 1.10,
    "max_gain_growth_ratio": 1.10,
    "max_abs_log_gain_growth_per_tick": 0.05,
    "condition_number_delta": 0.10,
    "offdiag_correlation_delta": 0.05,
    "max_link_unitarity_residual": 1e-5,
}

LOWER_BOUNDS = {
    "short_steps": 1,
    "long_steps": 2,
    "perturbation_size": 9e-5,
    "short_min_gain": 0.95,
    "short_max_gain": 0.95,
    "long_min_gain": 0.95,
    "long_max_gain": 0.95,
    "min_gain_growth_ratio": 0.90,
    "max_gain_growth_ratio": 0.90,
}


def main() -> None:
    diagnostics = sm_physical_right_production_echo_horizon_diagnostics()
    print("QCA_SMv0 Session 34 - physical-right production echo horizon")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE34_PHYSICAL_RIGHT_PRODUCTION_ECHO_HORIZON_PASS")


if __name__ == "__main__":
    main()
