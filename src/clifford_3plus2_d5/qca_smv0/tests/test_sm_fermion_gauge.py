"""Tests for QCA_SMv0 BCC streaming fermion gauge current."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    deterministic_sm_momenta,
    sm_project_to_coordinates,
    sm_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_gauge import (
    sm_apply_fermion_gauge_momentum_kick,
    sm_dirac_streaming_energy_density,
    sm_fermion_gauge_current_diagnostics,
    sm_fermion_gauss_constraint,
    sm_fermion_gauge_kick_then_transport,
    sm_fermion_left_gauge_current,
    sm_streaming_fermion_charge_density,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    deterministic_sm_state,
    sm_algebra_matrix_field,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
    sm_transform_state,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def test_streaming_current_vanishes_for_zero_state_and_not_for_seed() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    zero_state = jnp.zeros_like(state)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))

    assert jnp.linalg.norm(sm_fermion_left_gauge_current(zero_state, links)) < 1e-8
    assert jnp.linalg.norm(sm_fermion_left_gauge_current(state, links)) > 1e-4


def test_streaming_energy_current_charge_and_gauss_are_gauge_covariant() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = deterministic_sm_momenta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape, scale=0.06))

    transformed_state = sm_transform_state(state, site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_momenta = sm_transform_momenta(momenta, site_gauge)
    energy = sm_dirac_streaming_energy_density(state, links)
    transformed_energy = sm_dirac_streaming_energy_density(transformed_state, transformed_links)
    current = sm_fermion_left_gauge_current(state, links)
    transformed_current = sm_fermion_left_gauge_current(transformed_state, transformed_links)
    expected_current = sm_transform_momenta(current, site_gauge)
    charge = sm_streaming_fermion_charge_density(state)
    transformed_charge = sm_streaming_fermion_charge_density(transformed_state)
    expected_charge = sm_project_to_coordinates(
        site_gauge
        @ sm_algebra_matrix_field(charge)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )
    gauss = sm_fermion_gauss_constraint(links, momenta, state)
    transformed_gauss = sm_fermion_gauss_constraint(transformed_links, transformed_momenta, transformed_state)
    expected_gauss = sm_project_to_coordinates(
        site_gauge
        @ sm_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )

    assert jnp.abs(transformed_energy - energy) < 5e-7
    assert jnp.max(jnp.abs(transformed_current - expected_current)) < 2e-5
    assert jnp.max(jnp.abs(transformed_charge - expected_charge)) < 2e-6
    assert jnp.max(jnp.abs(transformed_gauss - expected_gauss)) < 2e-5


def test_fermion_gauss_constraint_zero_for_zero_state_and_zero_momenta() -> None:
    lattice_shape = (1, 1, 1)
    state = jnp.zeros_like(deterministic_sm_state(lattice_shape))
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = jnp.zeros((*lattice_shape, 8, 12), dtype=jnp.float32)

    assert jnp.linalg.norm(sm_fermion_gauss_constraint(links, momenta, state)) < 1e-7


def test_fermion_gauge_momentum_kick_is_reversible() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = deterministic_sm_momenta(lattice_shape)

    kicked = sm_apply_fermion_gauge_momentum_kick(momenta, state, links, step_size=0.01)
    restored = sm_apply_fermion_gauge_momentum_kick(kicked, state, links, step_size=-0.01)

    assert jnp.linalg.norm(kicked - momenta) > 1e-6
    assert jnp.max(jnp.abs(restored - momenta)) < 1e-8


def test_fermion_gauge_kick_then_transport_keeps_links_unitary_and_norm() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = deterministic_sm_momenta(lattice_shape)

    updated_state, updated_links, updated_momenta = sm_fermion_gauge_kick_then_transport(
        state,
        links,
        momenta,
        step_size=0.005,
    )

    assert updated_state.shape == state.shape
    assert updated_links.shape == links.shape
    assert updated_momenta.shape == momenta.shape
    assert sm_link_unitarity_residual(updated_links) < 5e-7
    assert jnp.abs(state_norm_squared(updated_state) - state_norm_squared(state)) < 3e-5


def test_fermion_gauge_current_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_fermion_gauge_current_diagnostics()

    assert diagnostics.streaming_energy_reality_residual < 1e-8
    assert diagnostics.zero_state_current_norm < 1e-8
    assert diagnostics.nonzero_current_norm > 1e-4
    assert diagnostics.streaming_energy_gauge_invariance_residual < 5e-7
    assert diagnostics.current_covariance_residual < 2e-5
    assert diagnostics.charge_covariance_residual < 2e-6
    assert diagnostics.gauss_covariance_residual < 2e-5
    assert diagnostics.gauss_zero_state_residual < 1e-7
    assert diagnostics.momentum_kick_delta_norm > 1e-6
    assert diagnostics.momentum_kick_reversibility_residual < 1e-8
    assert diagnostics.kicked_link_unitarity_residual < 5e-7
    assert diagnostics.spectator_norm_drift_after_kick < 3e-5
    assert diagnostics.jit_delta_current < 1e-5
    assert diagnostics.jit_delta_kick < 1e-7


def test_fermion_gauge_current_and_kick_are_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = deterministic_sm_momenta(lattice_shape)
    expected_current = sm_fermion_left_gauge_current(state, links)
    expected_kick = sm_apply_fermion_gauge_momentum_kick(momenta, state, links, step_size=0.01)
    jitted_current = jax.jit(sm_fermion_left_gauge_current)
    jitted_kick = jax.jit(sm_apply_fermion_gauge_momentum_kick, static_argnames=("step_size", "epsilon"))

    assert jnp.max(jnp.abs(jitted_current(state, links) - expected_current)) < 1e-5
    assert jnp.max(jnp.abs(jitted_kick(momenta, state, links, step_size=0.01) - expected_kick)) < 1e-7
