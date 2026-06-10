"""Session 29 certificate for production trajectory reversibility."""

from __future__ import annotations

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_reversibility import (
    sm_physical_right_production_reversibility_diagnostics,
)


THRESHOLDS = {
    "trajectory_roundtrip_residual": 1e-5,
    "trajectory_replay_residual": 1e-5,
    "max_path_restore_residual": 1e-5,
    "inverse_family_norm_drift": 1e-5,
    "forward_family_norm_drift": 1e-5,
    "max_forward_sm_link_unitarity_residual": 1e-5,
    "max_forward_higgs_link_unitarity_residual": 1e-5,
    "max_inverse_sm_link_unitarity_residual": 1e-5,
    "max_inverse_higgs_link_unitarity_residual": 1e-5,
    "jit_inverse_rollout_delta": 1e-5,
}

LOWER_BOUNDS = {
    "steps": 2,
    "naive_negative_trajectory_residual": 1e-2,
    "trajectory_inverse_improvement_ratio": 1e3,
}


def main() -> None:
    diagnostics = sm_physical_right_production_reversibility_diagnostics()
    print("QCA_SMv0 Session 29 - physical-right production trajectory reversibility")
    for field in diagnostics._fields:
        value = float(getattr(diagnostics, field))
        print(f"{field}: {value:.3e}")
        if field in THRESHOLDS and value >= THRESHOLDS[field]:
            raise SystemExit(f"{field}={value:.3e} exceeds threshold {THRESHOLDS[field]:.3e}")
        if field in LOWER_BOUNDS and value < LOWER_BOUNDS[field]:
            raise SystemExit(f"{field}={value:.3e} below lower bound {LOWER_BOUNDS[field]:.3e}")
    print("verdict: QCA_SMV0_STAGE29_PHYSICAL_RIGHT_PRODUCTION_TRAJECTORY_REVERSIBILITY_PASS")


if __name__ == "__main__":
    main()
