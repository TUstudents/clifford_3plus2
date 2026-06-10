"""Tests for the Stage 49 Gauss-projected production rollout."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_projected_rollout import (
    sm_physical_right_production_projected_rollout,
    sm_physical_right_production_projected_rollout_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_projected_rollout_records_raw_and_projected_histories() -> None:
    initial = sm_physical_right_production_initial_state()
    final, history = sm_physical_right_production_projected_rollout(
        initial,
        steps=2,
        step_size=0.001,
        projection_iterations=2,
    )

    assert history.raw_gauss_norms.shape == (3,)
    assert history.projected_gauss_norms.shape == (3,)
    assert history.step_post_tick_gauss_norms.shape == (2,)
    assert history.step_projected_gauss_norms.shape == (2,)
    assert jnp.all(history.step_projected_gauss_norms < history.step_post_tick_gauss_norms)
    assert jnp.all(history.step_projection_reduction_fractions > 0)
    assert jnp.all(history.step_projection_momentum_delta_norms > 1e-3)
    assert history.projected_gauss_norms[-1] == history.step_projected_gauss_norms[-1]
    assert jnp.linalg.norm(final.sm_momenta - initial.sm_momenta) > 1e-3


def test_projected_rollout_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_projected_rollout_diagnostics()

    assert diagnostics.step_count == 2
    assert diagnostics.projection_iteration_count == 10
    assert diagnostics.vacuum_initial_gauss_norm < 1e-8
    assert diagnostics.vacuum_final_gauss_norm < 1e-8
    assert diagnostics.vacuum_momentum_delta_norm < 1e-8
    assert diagnostics.initial_gauss_norm > 1e-1
    assert diagnostics.raw_final_gauss_norm > 1e-1
    assert diagnostics.projected_final_gauss_norm < diagnostics.raw_final_gauss_norm
    assert diagnostics.final_gauss_reduction_vs_raw_norm > 2e-1
    assert diagnostics.final_gauss_reduction_vs_raw_fraction > 3.5e-1
    assert diagnostics.min_step_projection_reduction_fraction > 5e-2
    assert diagnostics.min_step_projection_momentum_delta_norm > 1.0
    assert diagnostics.max_step_history_monotonicity_violation < 5e-7
    assert diagnostics.max_step_history_final_residual < 1e-7
    assert diagnostics.max_projected_sm_link_unitarity_residual < 1e-6
    assert diagnostics.max_projected_higgs_link_unitarity_residual < 1e-6
    assert diagnostics.history_all_finite


def test_projected_rollout_validates_steps() -> None:
    initial = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        sm_physical_right_production_projected_rollout(initial, steps=-1, step_size=0.001)

    _, history = sm_physical_right_production_projected_rollout(initial, steps=0, step_size=0.001)
    assert history.raw_gauss_norms.shape == (1,)
    assert history.projected_gauss_norms.shape == (1,)
    assert history.step_post_tick_gauss_norms.shape == (0,)
