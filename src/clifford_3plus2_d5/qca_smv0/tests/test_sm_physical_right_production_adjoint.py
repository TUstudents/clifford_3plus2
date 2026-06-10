"""Tests for the physical-right production adjoint audit."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_adjoint import (
    sm_family_physical_right_gauged_dirac_adjoint_step,
    sm_physical_right_production_adjoint_diagnostics,
    sm_physical_right_production_frozen_fermion_stage,
    sm_physical_right_production_frozen_fermion_stage_adjoint,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import (
    sm_family_physical_right_gauged_dirac_step,
)


def test_physical_right_transport_adjoint_restores_state() -> None:
    state = sm_physical_right_production_initial_state()
    transported = sm_family_physical_right_gauged_dirac_step(state.family_state, state.sm_links)
    restored = sm_family_physical_right_gauged_dirac_adjoint_step(transported, state.sm_links)

    assert jnp.max(jnp.abs(restored - state.family_state)) < 5e-6


def test_physical_right_frozen_fermion_stage_has_adjoint() -> None:
    state = sm_physical_right_production_initial_state()
    step_size = 1e-3
    forward = sm_physical_right_production_step(state, step_size=step_size)
    fermion_forward = sm_physical_right_production_frozen_fermion_stage(
        state.family_state,
        state.higgs,
        forward.higgs,
        forward.sm_links,
        step_size=step_size,
    )
    restored = sm_physical_right_production_frozen_fermion_stage_adjoint(
        fermion_forward,
        state.higgs,
        forward.higgs,
        forward.sm_links,
        step_size=step_size,
    )

    assert jnp.max(jnp.abs(restored - state.family_state)) < 5e-6


def test_physical_right_production_adjoint_diagnostics_record_limitation() -> None:
    diagnostics = sm_physical_right_production_adjoint_diagnostics()

    assert diagnostics.transport_adjoint_residual < 5e-6
    assert diagnostics.frozen_fermion_stage_adjoint_residual < 5e-6
    assert diagnostics.frozen_fermion_norm_drift < 1e-5
    assert diagnostics.local_collision_inverse_residual < 5e-6
    assert diagnostics.naive_negative_tick_residual > 1e-2
    assert bool(diagnostics.naive_negative_tick_limitation_detected)
    assert diagnostics.forward_sm_link_unitarity_residual < 1e-5
    assert diagnostics.naive_negative_sm_link_unitarity_residual < 1e-5
    assert diagnostics.forward_higgs_link_unitarity_residual < 1e-5
    assert diagnostics.naive_negative_higgs_link_unitarity_residual < 1e-5


def test_physical_right_transport_adjoint_validates_inputs() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="family SM Dirac state"):
        sm_family_physical_right_gauged_dirac_adjoint_step(state.family_state[..., 0], state.sm_links)
