"""Tests for QCA_SMv0 dynamic Higgs-field evolution."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    deterministic_higgs_theta,
    sm_higgs_dynamics_diagnostics,
    sm_higgs_force,
    sm_higgs_generator_antihermitian_residual,
    sm_higgs_gradient_density,
    sm_higgs_hamiltonian_density,
    sm_higgs_leapfrog_step,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
    sm_higgs_site_gauge_from_algebra,
    sm_identity_higgs_links,
    sm_pure_gauge_higgs_links_from_site_algebra,
    sm_transform_higgs_field,
    sm_transform_higgs_links,
)


def test_higgs_generators_and_links_are_unitary() -> None:
    lattice_shape = (2, 1, 1)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))

    assert sm_higgs_generator_antihermitian_residual() < 1e-7
    assert sm_higgs_link_unitarity_residual(links) < 5e-7


def test_higgs_vacuum_and_pure_gauge_have_no_gradient_force() -> None:
    lattice_shape = (2, 1, 1)
    vacuum = sm_constant_higgs(lattice_shape)
    identity_links = sm_identity_higgs_links(lattice_shape)
    site_theta = deterministic_higgs_site_theta(lattice_shape)
    site_gauge = sm_higgs_site_gauge_from_algebra(site_theta)
    pure_links = sm_pure_gauge_higgs_links_from_site_algebra(site_theta)
    pure_field = sm_transform_higgs_field(vacuum, site_gauge)

    assert jnp.linalg.norm(sm_higgs_force(vacuum, identity_links)) < 1e-7
    assert sm_higgs_gradient_density(pure_field, pure_links) < 1e-10


def test_higgs_force_and_energy_are_gauge_covariant() -> None:
    lattice_shape = (2, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    site_gauge = sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape, scale=0.17))

    transformed_field = sm_transform_higgs_field(field, site_gauge)
    transformed_momenta = sm_transform_higgs_field(momenta, site_gauge)
    transformed_links = sm_transform_higgs_links(links, site_gauge)
    transformed_force = sm_higgs_force(transformed_field, transformed_links)
    expected_force = sm_transform_higgs_field(sm_higgs_force(field, links), site_gauge)
    energy = sm_higgs_hamiltonian_density(field, momenta, links)
    transformed_energy = sm_higgs_hamiltonian_density(transformed_field, transformed_momenta, transformed_links)

    assert jnp.max(jnp.abs(transformed_force - expected_force)) < 2e-7
    assert jnp.abs(transformed_energy - energy) < 1e-7


def test_higgs_leapfrog_is_reversible_and_nearly_hamiltonian() -> None:
    lattice_shape = (2, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    step_size = 0.01

    updated_field, updated_momenta = sm_higgs_leapfrog_step(field, momenta, links, step_size=step_size)
    reversed_field, reversed_momenta = sm_higgs_leapfrog_step(updated_field, -updated_momenta, links, step_size=step_size)
    before = sm_higgs_hamiltonian_density(field, momenta, links)
    after = sm_higgs_hamiltonian_density(updated_field, updated_momenta, links)

    assert jnp.max(jnp.abs(reversed_field - field)) < 1e-7
    assert jnp.max(jnp.abs(reversed_momenta + momenta)) < 1e-7
    assert jnp.abs(after - before) < 5e-6


def test_higgs_dynamics_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_higgs_dynamics_diagnostics()

    assert diagnostics.generator_antihermitian_residual < 1e-7
    assert diagnostics.link_unitarity_residual < 5e-7
    assert diagnostics.vacuum_force_norm < 1e-7
    assert diagnostics.pure_gauge_gradient_energy < 1e-10
    assert diagnostics.force_covariance_residual < 2e-7
    assert diagnostics.energy_gauge_invariance_residual < 1e-7
    assert diagnostics.hamiltonian_drift < 5e-6
    assert diagnostics.reversibility_residual < 1e-7
    assert diagnostics.jit_delta_higgs < 1e-8
    assert diagnostics.jit_delta_momenta < 1e-8


def test_higgs_leapfrog_step_is_jittable() -> None:
    lattice_shape = (2, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    expected_field, expected_momenta = sm_higgs_leapfrog_step(field, momenta, links, step_size=0.01)
    jitted = jax.jit(sm_higgs_leapfrog_step, static_argnames=("step_size",))
    actual_field, actual_momenta = jitted(field, momenta, links, step_size=0.01)

    assert jnp.max(jnp.abs(actual_field - expected_field)) < 1e-8
    assert jnp.max(jnp.abs(actual_momenta - expected_momenta)) < 1e-8
