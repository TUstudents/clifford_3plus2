"""One-command phenomenology rollout for QCA_SMv0.

This is the constructive front-door smoke run: benchmark quark masses and CKM
go into the calibrated center-CP FN layer, which then drives a local Higgs/FN
collision inside the compact field-QCA rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    deterministic_qca_family_state,
    sm_qca_rollout_config_from_masses_ckm,
    sm_run_qca_rollout,
)


class PhenomenologyRolloutSummary(NamedTuple):
    """Compact numerical summary for the QCA_SMv0 phenomenology rollout."""

    selected_label: str
    passed_center_cp: bool
    ckm_abs_residual: float
    jarlskog_candidate: float
    jarlskog_target: float
    norm_initial: float
    norm_final: float
    norm_drift: float
    max_density_initial: float
    max_density_final: float
    used_higgs_fn_collision: bool
    steps_completed: int


def benchmark_ckm() -> jnp.ndarray:
    """Return the benchmark CKM matrix used by the compact SM rollout."""

    lambda_rec = jnp.asarray(0.22501, dtype=jnp.float32)
    s23 = jnp.asarray(0.04183, dtype=jnp.float32)
    s13 = jnp.asarray(0.003732, dtype=jnp.float32)
    delta = jnp.asarray(1.147, dtype=jnp.float32)
    c12 = jnp.sqrt(1.0 - lambda_rec * lambda_rec)
    c23 = jnp.sqrt(1.0 - s23 * s23)
    c13 = jnp.sqrt(1.0 - s13 * s13)
    exp_pos = jnp.exp(1j * delta)
    exp_neg = jnp.exp(-1j * delta)
    return jnp.asarray(
        [
            [c12 * c13, lambda_rec * c13, s13 * exp_neg],
            [
                -lambda_rec * c23 - c12 * s23 * s13 * exp_pos,
                c12 * c23 - lambda_rec * s23 * s13 * exp_pos,
                s23 * c13,
            ],
            [
                lambda_rec * s23 - c12 * c23 * s13 * exp_pos,
                -c12 * s23 - lambda_rec * c23 * s13 * exp_pos,
                c23 * c13,
            ],
        ],
        dtype=jnp.complex64,
    )


def run_phenomenology_rollout(
    *,
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
    steps: int = 4,
    yukawa_step_size: float = 0.01,
) -> PhenomenologyRolloutSummary:
    """Run the benchmark masses/CKM through the compact QCA_SMv0 rollout."""

    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)
    state = deterministic_qca_family_state(lattice_shape)
    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        benchmark_ckm(),
        lattice_shape,
        center_fit_steps=0,
        yukawa_step_size=yukawa_step_size,
        record_density=True,
    )
    result = sm_run_qca_rollout(calibrated.config, state, steps=steps)
    residuals = calibrated.verdict.fit.residuals
    norm_initial = float(result.norm_history[0])
    norm_final = float(result.norm_history[-1])
    return PhenomenologyRolloutSummary(
        selected_label=calibrated.verdict.selected_label,
        passed_center_cp=bool(calibrated.verdict.passed),
        ckm_abs_residual=float(residuals.ckm_abs_residual),
        jarlskog_candidate=float(residuals.candidate_jarlskog),
        jarlskog_target=float(residuals.target_jarlskog),
        norm_initial=norm_initial,
        norm_final=norm_final,
        norm_drift=abs(norm_final - norm_initial),
        max_density_initial=float(result.max_density_history[0]),
        max_density_final=float(result.max_density_history[-1]),
        used_higgs_fn_collision=bool(result.used_higgs_fn_collision),
        steps_completed=int(result.steps_completed),
    )


def main() -> None:
    """Print the benchmark rollout summary."""

    summary = run_phenomenology_rollout()
    print("QCA_SMv0 phenomenology rollout")
    print(f"  selected_center_cp_texture: {summary.selected_label}")
    print(f"  center_cp_passed: {summary.passed_center_cp}")
    print(f"  ckm_abs_residual: {summary.ckm_abs_residual:.6g}")
    print(f"  jarlskog_candidate: {summary.jarlskog_candidate:.6g}")
    print(f"  jarlskog_target: {summary.jarlskog_target:.6g}")
    print(f"  norm_initial: {summary.norm_initial:.9g}")
    print(f"  norm_final: {summary.norm_final:.9g}")
    print(f"  norm_drift: {summary.norm_drift:.6g}")
    print(f"  max_density_initial: {summary.max_density_initial:.9g}")
    print(f"  max_density_final: {summary.max_density_final:.9g}")
    print(f"  used_higgs_fn_collision: {summary.used_higgs_fn_collision}")
    print(f"  steps_completed: {summary.steps_completed}")


if __name__ == "__main__":
    main()
