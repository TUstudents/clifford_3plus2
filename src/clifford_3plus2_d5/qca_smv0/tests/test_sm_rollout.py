"""Tests for the compact QCA_SMv0 rollout runner."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_free_dirac_internal_step, sm_identity_links
from clifford_3plus2_d5.qca_smv0.sm_rollout import (
    QCASMRolloutConfig,
    deterministic_qca_family_state,
    deterministic_qca_sm_state,
    sm_qca_center_cp_rollout_config,
    sm_qca_rollout_config_from_masses_ckm,
    sm_qca_total_norm,
    sm_run_qca_rollout,
)


def _benchmark_ckm() -> jnp.ndarray:
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


def test_free_bcc_rollout_matches_dirac_step_and_records_observables() -> None:
    state = jnp.zeros((2, 1, 1, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[1, 0, 0, 3].set(-0.5 + 0.125j)

    result = sm_run_qca_rollout(QCASMRolloutConfig(record_density=True), state, steps=1)

    assert result.steps_completed == 1
    assert not result.used_gauge_links
    assert not result.used_higgs_fn_collision
    assert result.density_history.shape == (2, 2, 1, 1)
    assert jnp.max(jnp.abs(result.final_state - bcc_dirac_step(state))) < 2e-7
    assert jnp.max(jnp.abs(result.norm_history - result.norm_history[0])) < 2e-7


def test_identity_gauge_rollout_matches_free_internal_step() -> None:
    state = deterministic_qca_sm_state((2, 1, 1))
    links = sm_identity_links((2, 1, 1), dtype=state.dtype)

    result = sm_run_qca_rollout(QCASMRolloutConfig(links=links), state, steps=1)

    assert result.used_gauge_links
    assert jnp.max(jnp.abs(result.final_state - sm_free_dirac_internal_step(state))) < 2e-7
    assert jnp.max(jnp.abs(result.norm_history - result.norm_history[0])) < 2e-7


def test_center_cp_higgs_fn_rollout_preserves_norm_and_changes_family_state() -> None:
    state = deterministic_qca_family_state((1, 1, 1))
    config = sm_qca_center_cp_rollout_config((1, 1, 1), yukawa_step_size=0.02, record_density=True)

    result = sm_run_qca_rollout(config, state, steps=1)
    stream_only = sm_run_qca_rollout(QCASMRolloutConfig(record_density=True), state, steps=1)

    assert result.used_higgs_fn_collision
    assert result.density_history.shape == (2, 1, 1, 1)
    assert jnp.abs(sm_qca_total_norm(result.final_state) - sm_qca_total_norm(state)) < 2e-6
    assert jnp.max(jnp.abs(result.final_state - stream_only.final_state)) > 1e-8


def test_calibrated_center_cp_rollout_config_from_masses_ckm_runs_family_rollout() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_qca_family_state(lattice_shape)
    up_masses = jnp.asarray([0.00216 / 172.57, 1.2730 / 172.57, 1.0], dtype=jnp.float32)
    down_masses = jnp.asarray([0.00467 / 4.183, 0.0935 / 4.183, 1.0], dtype=jnp.float32)

    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        _benchmark_ckm(),
        lattice_shape,
        center_fit_steps=0,
        yukawa_step_size=0.01,
    )
    result = sm_run_qca_rollout(calibrated.config, state, steps=1)

    assert calibrated.verdict.passed
    assert calibrated.verdict.selected_label == "wilson_flux_rule"
    assert result.used_higgs_fn_collision
    assert jnp.abs(result.norm_history[-1] - result.norm_history[0]) < 2e-6
