"""Tests for the Stage 46 production Gauss relaxation."""

import pytest
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_projection import (
    sm_physical_right_production_gauss_projection_diagnostics,
    sm_physical_right_production_gauss_relaxation_step,
)


def test_gauss_relaxation_reduces_deterministic_residual_and_only_changes_momenta() -> None:
    initial = sm_physical_right_production_initial_state()
    relaxed, line_step, gradient = sm_physical_right_production_gauss_relaxation_step(initial)

    assert line_step > 1e-4
    assert jnp.linalg.norm(gradient) > 1e-4
    assert sm_physical_right_production_gauss(relaxed).shape == sm_physical_right_production_gauss(initial).shape
    assert (
        sm_physical_right_production_gauss(relaxed).reshape((-1,)).dot(
            sm_physical_right_production_gauss(relaxed).reshape((-1,)),
        )
        < sm_physical_right_production_gauss(initial).reshape((-1,)).dot(
            sm_physical_right_production_gauss(initial).reshape((-1,)),
        )
    )
    assert jnp.linalg.norm(relaxed.family_state - initial.family_state) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs - initial.higgs) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs_momenta - initial.higgs_momenta) < 1e-8
    assert jnp.linalg.norm(relaxed.sm_links - initial.sm_links) < 1e-8
    assert jnp.linalg.norm(relaxed.higgs_links - initial.higgs_links) < 1e-8


def test_gauss_relaxation_keeps_vacuum_unchanged() -> None:
    vacuum = sm_physical_right_production_vacuum_state()
    relaxed, line_step, gradient = sm_physical_right_production_gauss_relaxation_step(vacuum)

    assert line_step == 0.0
    assert jnp.linalg.norm(gradient) < 1e-8
    assert jnp.linalg.norm(relaxed.sm_momenta - vacuum.sm_momenta) < 1e-8
    assert jnp.linalg.norm(sm_physical_right_production_gauss(relaxed)) < 1e-8


def test_gauss_projection_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_gauss_projection_diagnostics()

    assert diagnostics.vacuum_initial_gauss_norm < 1e-8
    assert diagnostics.vacuum_relaxed_gauss_norm < 1e-8
    assert diagnostics.vacuum_momentum_delta_norm < 1e-8
    assert diagnostics.initial_gauss_norm > 1e-1
    assert diagnostics.relaxed_gauss_norm < diagnostics.initial_gauss_norm
    assert diagnostics.gauss_reduction_norm > 1e-4
    assert diagnostics.gauss_reduction_fraction > 1e-3
    assert diagnostics.gradient_norm > 1e-4
    assert diagnostics.line_step > 1e-4
    assert diagnostics.momentum_delta_norm > 1e-4
    assert diagnostics.family_state_delta_norm < 1e-8
    assert diagnostics.higgs_delta_norm < 1e-8
    assert diagnostics.higgs_momentum_delta_norm < 1e-8
    assert diagnostics.sm_link_delta_norm < 1e-8
    assert diagnostics.higgs_link_delta_norm < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 1e-6
    assert diagnostics.higgs_link_unitarity_residual < 1e-6
    assert diagnostics.jit_delta_momenta < 5e-6


def test_gauss_relaxation_validates_inputs() -> None:
    with pytest.raises(ValueError, match="relaxation_scale must be positive"):
        sm_physical_right_production_gauss_relaxation_step(
            sm_physical_right_production_initial_state(),
            relaxation_scale=0.0,
        )
