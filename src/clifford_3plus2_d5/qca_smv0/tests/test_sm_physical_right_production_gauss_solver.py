"""Tests for the Stage 47 iterated Gauss relaxation solver."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_solver import (
    sm_physical_right_production_gauss_relaxation_solve,
    sm_physical_right_production_gauss_solver_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_gauss_relaxation_solve_records_monotone_history() -> None:
    initial = sm_physical_right_production_initial_state()
    relaxed, history = sm_physical_right_production_gauss_relaxation_solve(initial, iterations=4)
    reductions = history.gauss_norms[:-1] - history.gauss_norms[1:]

    assert history.gauss_norms.shape == (5,)
    assert history.line_steps.shape == (4,)
    assert history.gradient_norms.shape == (4,)
    assert jnp.all(reductions >= -5e-7)
    assert history.gauss_norms[-1] < history.gauss_norms[0]
    assert jnp.min(reductions) > 1e-3
    assert jnp.linalg.norm(relaxed.sm_momenta - initial.sm_momenta) > 1e-3
    assert jnp.linalg.norm(relaxed.family_state - initial.family_state) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs - initial.higgs) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs_momenta - initial.higgs_momenta) < 1e-8
    assert jnp.linalg.norm(relaxed.sm_links - initial.sm_links) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs_links - initial.higgs_links) < 1e-8


def test_gauss_solver_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_gauss_solver_diagnostics()

    assert diagnostics.iteration_count == 10
    assert diagnostics.vacuum_initial_gauss_norm < 1e-8
    assert diagnostics.vacuum_final_gauss_norm < 1e-8
    assert diagnostics.vacuum_momentum_delta_norm < 1e-8
    assert diagnostics.initial_gauss_norm > 1e-1
    assert diagnostics.final_gauss_norm < diagnostics.initial_gauss_norm
    assert diagnostics.total_gauss_reduction_norm > 1.8e-1
    assert diagnostics.total_gauss_reduction_fraction > 3e-1
    assert diagnostics.monotonicity_violation < 5e-7
    assert diagnostics.min_step_reduction_norm > 1e-3
    assert diagnostics.min_line_step > 1e-4
    assert diagnostics.max_line_step > 1e-4
    assert diagnostics.min_gradient_norm > 1e-4
    assert diagnostics.max_gradient_norm > 1e-4
    assert diagnostics.total_momentum_delta_norm > 1e-3
    assert diagnostics.family_state_delta_norm < 1e-8
    assert diagnostics.higgs_delta_norm < 1e-8
    assert diagnostics.higgs_momentum_delta_norm < 1e-8
    assert diagnostics.sm_link_delta_norm < 1e-8
    assert diagnostics.higgs_link_delta_norm < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 1e-6
    assert diagnostics.higgs_link_unitarity_residual < 1e-6
    assert diagnostics.jit_delta_momenta < 1e-3
    assert diagnostics.jit_delta_history < 5e-5


def test_gauss_relaxation_solve_validates_inputs() -> None:
    initial = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="iterations must be positive"):
        sm_physical_right_production_gauss_relaxation_solve(initial, iterations=0)

    with pytest.raises(ValueError, match="relaxation_scale must be positive"):
        sm_physical_right_production_gauss_relaxation_solve(initial, relaxation_scale=0.0)
