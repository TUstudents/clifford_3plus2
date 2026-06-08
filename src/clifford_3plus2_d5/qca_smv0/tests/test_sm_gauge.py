"""Tests for QCA_SMv0 static Standard-Model gauge backgrounds."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    deterministic_sm_state,
    sm_average_normalized_wilson_loop,
    sm_average_wilson_action_density,
    sm_continuum_linearization_residual,
    sm_free_dirac_internal_step,
    sm_gauged_dirac_step,
    sm_gauged_norm_drift,
    sm_generator_antihermitian_residual,
    sm_generators,
    sm_identity_links,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_pure_gauge_links_from_site_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
    sm_transform_state,
)


def test_sm_generators_are_antihermitian() -> None:
    generators = sm_generators()

    assert generators.shape == (SM_GENERATOR_COUNT, SM_INTERNAL_DIM, SM_INTERNAL_DIM)
    assert sm_generator_antihermitian_residual() < 1e-6


def test_sm_links_are_unitary_and_zero_algebra_is_identity() -> None:
    lattice_shape = (1, 1, 1)
    theta_zero = jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)
    identity = jnp.eye(SM_INTERNAL_DIM, dtype=jnp.complex64)
    zero_links = sm_link_field_from_algebra(theta_zero)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape))

    assert jnp.max(jnp.abs(zero_links - identity)) < 1e-6
    assert sm_link_unitarity_residual(links) < 1e-5


def test_identity_gauge_background_reduces_to_free_dirac_step() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    identity_links = sm_identity_links(lattice_shape)

    actual = sm_gauged_dirac_step(state, identity_links)
    expected = sm_free_dirac_internal_step(state)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-6


def test_static_sm_links_preserve_norm_on_periodic_lattice() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape))

    assert jnp.abs(sm_gauged_norm_drift(state, links)) < 2e-5


def test_static_sm_transport_is_locally_gauge_covariant() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape))
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))

    transformed_state = sm_transform_state(state, site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    actual = sm_gauged_dirac_step(transformed_state, transformed_links)
    expected = sm_transform_state(sm_gauged_dirac_step(state, links), site_gauge)

    assert jnp.max(jnp.abs(actual - expected)) < 2e-6


def test_wilson_loop_response_is_gauge_invariant_and_nontrivial() -> None:
    lattice_shape = (2, 1, 1)
    identity_links = sm_identity_links(lattice_shape)
    pure_links = sm_pure_gauge_links_from_site_algebra(deterministic_sm_site_theta(lattice_shape))
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape))
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))
    transformed_links = sm_transform_links(links, site_gauge)

    assert sm_average_wilson_action_density(identity_links) < 1e-6
    assert sm_average_wilson_action_density(pure_links) < 1e-6
    assert sm_average_wilson_action_density(links) > 1e-7
    assert (
        jnp.abs(sm_average_normalized_wilson_loop(links) - sm_average_normalized_wilson_loop(transformed_links))
        < 1e-6
    )


def test_continuum_covariant_derivative_linearization_scales_quadratically() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    theta = deterministic_sm_link_theta(lattice_shape)

    residual = sm_continuum_linearization_residual(state, theta, epsilon=0.05)
    residual_half = sm_continuum_linearization_residual(state, theta, epsilon=0.025)
    ratio = residual_half / residual

    assert residual > residual_half > 0
    assert 0.18 < ratio < 0.32


def test_sm_gauged_dirac_step_is_jittable() -> None:
    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape))

    expected = sm_gauged_dirac_step(state, links)
    actual = jax.jit(sm_gauged_dirac_step)(state, links)

    assert jnp.max(jnp.abs(actual - expected)) < 1e-6
