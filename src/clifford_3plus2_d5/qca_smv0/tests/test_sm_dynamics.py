"""Tests for QCA_SMv0 pure dynamic SM gauge fields."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    deterministic_sm_curved_link_theta,
    deterministic_sm_momenta,
    sm_apply_momentum_update,
    sm_electric_divergence,
    sm_gauge_hamiltonian_density,
    sm_left_wilson_force,
    sm_leapfrog_step,
    sm_linearized_plaquette_field_strength,
    sm_linearized_yang_mills_action_density,
    sm_momentum_algebra,
    sm_no_backreaction_fermion_gauge_step,
    sm_project_to_coordinates,
    sm_transform_momenta,
    sm_weak_field_holonomy_residual,
    sm_weak_field_yang_mills_action_ratio,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    deterministic_sm_state,
    sm_algebra_matrix_field,
    sm_gauged_dirac_step,
    sm_identity_links,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_pure_gauge_links_from_site_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def test_sm_momentum_projection_roundtrip() -> None:
    lattice_shape = (1, 1, 1)
    momenta = deterministic_sm_momenta(lattice_shape)

    projected = sm_project_to_coordinates(sm_momentum_algebra(momenta))

    assert projected.shape == momenta.shape
    assert jnp.max(jnp.abs(projected - momenta)) < 1e-6


def test_sm_momentum_transforms_by_target_site_adjoint() -> None:
    lattice_shape = (1, 1, 1)
    momenta = deterministic_sm_momenta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))
    transformed_momenta = sm_transform_momenta(momenta, site_gauge)

    actual = sm_momentum_algebra(transformed_momenta)
    expected = jnp.einsum(
        "...ab,...hbc,...cd->...had",
        site_gauge,
        sm_momentum_algebra(momenta),
        jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )

    assert jnp.max(jnp.abs(actual - expected)) < 2e-6


def test_sm_left_wilson_force_detects_only_nonflat_curvature() -> None:
    lattice_shape = (1, 1, 1)
    identity_links = sm_identity_links(lattice_shape)
    pure_links = sm_pure_gauge_links_from_site_algebra(deterministic_sm_site_theta(lattice_shape))
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=2.0))

    assert jnp.linalg.norm(sm_left_wilson_force(identity_links)) < 1e-7
    assert jnp.linalg.norm(sm_left_wilson_force(pure_links)) < 1e-4
    assert jnp.linalg.norm(sm_left_wilson_force(links)) > 2e-4


def test_sm_momentum_update_and_leapfrog_keep_links_unitary_and_nearly_hamiltonian() -> None:
    lattice_shape = (1, 1, 1)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.6))
    momenta = deterministic_sm_momenta(lattice_shape)

    directly_updated = sm_apply_momentum_update(links, momenta, step_size=0.02)
    updated_links, updated_momenta = sm_leapfrog_step(links, momenta, step_size=0.02)
    hamiltonian_before = sm_gauge_hamiltonian_density(links, momenta)
    hamiltonian_after = sm_gauge_hamiltonian_density(updated_links, updated_momenta)

    assert sm_link_unitarity_residual(directly_updated) < 1e-5
    assert sm_link_unitarity_residual(updated_links) < 1e-5
    assert jnp.abs(hamiltonian_after - hamiltonian_before) < 1e-5


def test_sm_leapfrog_is_reversible_under_momentum_flip() -> None:
    lattice_shape = (1, 1, 1)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.6))
    momenta = deterministic_sm_momenta(lattice_shape)

    updated_links, updated_momenta = sm_leapfrog_step(links, momenta, step_size=0.02)
    reversed_links, reversed_momenta = sm_leapfrog_step(updated_links, -updated_momenta, step_size=0.02)

    assert jnp.max(jnp.abs(reversed_links - links)) < 2e-5
    assert jnp.max(jnp.abs(reversed_momenta + momenta)) < 2e-5


def test_sm_electric_divergence_is_zero_for_zero_momenta_and_covariant() -> None:
    lattice_shape = (1, 1, 1)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.6))
    momenta = deterministic_sm_momenta(lattice_shape)
    zero_momenta = jnp.zeros_like(momenta)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))

    transformed_links = sm_transform_links(links, site_gauge)
    transformed_momenta = sm_transform_momenta(momenta, site_gauge)
    divergence = sm_electric_divergence(links, momenta)
    transformed_divergence = sm_electric_divergence(transformed_links, transformed_momenta)
    expected = site_gauge @ sm_algebra_matrix_field(divergence) @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2)

    assert jnp.linalg.norm(sm_electric_divergence(links, zero_momenta)) < 1e-7
    assert jnp.max(jnp.abs(sm_algebra_matrix_field(transformed_divergence) - expected)) < 2e-6


def test_sm_pure_gauge_zero_momentum_leapfrog_preserves_gauss_constraint() -> None:
    lattice_shape = (1, 1, 1)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))
    pure_links = sm_transform_links(sm_identity_links(lattice_shape), site_gauge)
    zero_momenta = jnp.zeros((*lattice_shape, 8, 12), dtype=jnp.float32)

    updated_links, updated_momenta = sm_leapfrog_step(pure_links, zero_momenta, step_size=0.02)

    assert jnp.linalg.norm(sm_electric_divergence(updated_links, updated_momenta)) < 1e-6


def test_sm_weak_field_plaquette_holonomy_linearizes_to_field_strength() -> None:
    theta = deterministic_sm_curved_link_theta()
    field_strength = sm_linearized_plaquette_field_strength(theta)
    residual = sm_weak_field_holonomy_residual(theta, epsilon=0.04)
    residual_half = sm_weak_field_holonomy_residual(theta, epsilon=0.02)
    ratio = residual_half / residual

    assert field_strength.shape == (*theta.shape[:3], 6, 32, 32)
    assert jnp.linalg.norm(field_strength) > 1.0
    assert residual > residual_half > 0
    assert 0.18 < ratio < 0.32


def test_sm_wilson_action_matches_weak_field_yang_mills_density() -> None:
    theta = deterministic_sm_curved_link_theta()
    linearized_density = sm_linearized_yang_mills_action_density(theta)
    ratio = sm_weak_field_yang_mills_action_ratio(theta, epsilon=0.04)

    assert linearized_density > 0
    assert 0.97 < ratio < 1.03


def test_no_backreaction_fermion_gauge_step_preserves_spectator_norm() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.6))
    momenta = deterministic_sm_momenta(lattice_shape)

    updated_state, updated_links, updated_momenta = sm_no_backreaction_fermion_gauge_step(
        state,
        links,
        momenta,
        step_size=0.02,
    )

    assert updated_state.shape == state.shape
    assert updated_links.shape == links.shape
    assert updated_momenta.shape == momenta.shape
    assert jnp.abs(state_norm_squared(updated_state) - state_norm_squared(state)) < 2e-5
    assert jnp.max(jnp.abs(updated_state - sm_gauged_dirac_step(state, updated_links))) < 1e-6


def test_sm_leapfrog_step_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.6))
    momenta = deterministic_sm_momenta(lattice_shape)
    expected_links, expected_momenta = sm_leapfrog_step(links, momenta, step_size=0.02)
    jitted = jax.jit(sm_leapfrog_step, static_argnames=("step_size", "beta", "force_epsilon"))
    actual_links, actual_momenta = jitted(links, momenta, step_size=0.02)

    assert actual_links.shape == links.shape
    assert actual_momenta.shape == momenta.shape
    assert jnp.max(jnp.abs(actual_links - expected_links)) < 1e-6
    assert jnp.max(jnp.abs(actual_momenta - expected_momenta)) < 1e-6
