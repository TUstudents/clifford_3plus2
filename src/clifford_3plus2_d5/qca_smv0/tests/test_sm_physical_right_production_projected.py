"""Tests for the Stage 48 Gauss-projected production step."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_projected import (
    sm_physical_right_production_projected_step,
    sm_physical_right_production_projected_step_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)


def test_projected_step_relaxes_only_post_tick_sm_momenta() -> None:
    initial = sm_physical_right_production_initial_state()
    unprojected = sm_physical_right_production_step(initial, step_size=0.001)
    projected, history = sm_physical_right_production_projected_step(
        initial,
        step_size=0.001,
        projection_iterations=4,
    )

    assert history.gauss_norms.shape == (5,)
    assert history.line_steps.shape == (4,)
    assert history.gradient_norms.shape == (4,)
    assert jnp.linalg.norm(sm_physical_right_production_gauss(projected)) < jnp.linalg.norm(
        sm_physical_right_production_gauss(unprojected),
    )
    assert jnp.linalg.norm(projected.sm_momenta - unprojected.sm_momenta) > 1e-3
    assert jnp.linalg.norm(projected.family_state - unprojected.family_state) < 1e-8
    assert jnp.linalg.norm(projected.higgs - unprojected.higgs) < 1e-8
    assert jnp.linalg.norm(projected.higgs_momenta - unprojected.higgs_momenta) < 1e-8
    assert jnp.linalg.norm(projected.sm_links - unprojected.sm_links) < 1e-8
    assert jnp.linalg.norm(projected.higgs_links - unprojected.higgs_links) < 1e-8


def test_projected_step_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_projected_step_diagnostics()

    assert diagnostics.projection_iteration_count == 10
    assert diagnostics.vacuum_unprojected_gauss_norm < 1e-8
    assert diagnostics.vacuum_projected_gauss_norm < 1e-8
    assert diagnostics.vacuum_projection_momentum_delta_norm < 1e-8
    assert diagnostics.unprojected_gauss_norm > 1e-1
    assert diagnostics.projected_gauss_norm < diagnostics.unprojected_gauss_norm
    assert diagnostics.gauss_reduction_norm > 1.8e-1
    assert diagnostics.gauss_reduction_fraction > 3e-1
    assert diagnostics.history_monotonicity_violation < 5e-7
    assert diagnostics.history_final_gauss_residual < 1e-7
    assert diagnostics.min_history_step_reduction_norm > 1e-3
    assert diagnostics.projection_momentum_delta_norm > 1e-3
    assert diagnostics.family_state_delta_norm < 1e-8
    assert diagnostics.higgs_delta_norm < 1e-8
    assert diagnostics.higgs_momentum_delta_norm < 1e-8
    assert diagnostics.sm_link_delta_norm < 1e-8
    assert diagnostics.higgs_link_delta_norm < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 1e-6
    assert diagnostics.higgs_link_unitarity_residual < 1e-6
    assert diagnostics.jit_delta_projected_sm_momenta < 1.5e-3
    assert diagnostics.jit_delta_projected_history < 5e-5


def test_projected_step_validates_projection_inputs() -> None:
    initial = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="iterations must be positive"):
        sm_physical_right_production_projected_step(
            initial,
            step_size=0.001,
            projection_iterations=0,
        )

    with pytest.raises(ValueError, match="relaxation_scale must be positive"):
        sm_physical_right_production_projected_step(
            initial,
            step_size=0.001,
            projection_relaxation_scale=0.0,
        )
