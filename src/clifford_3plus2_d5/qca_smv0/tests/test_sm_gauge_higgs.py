"""Tests for QCA_SMv0 coupled gauge-Higgs backreaction."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    deterministic_higgs_theta,
    sm_higgs_algebra_matrix_field,
    sm_higgs_link_field_from_algebra,
    sm_higgs_site_gauge_from_algebra,
    sm_identity_higgs_links,
    sm_pure_gauge_higgs_links_from_site_algebra,
    sm_transform_higgs_field,
    sm_transform_higgs_links,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge_higgs import (
    deterministic_higgs_link_momenta,
    sm_coupled_higgs_gauge_hamiltonian_density,
    sm_coupled_higgs_gauge_leapfrog_step,
    sm_embed_higgs_to_sm_momenta,
    sm_extract_higgs_from_sm_momenta,
    sm_gauge_higgs_backreaction_diagnostics,
    sm_higgs_charge_density,
    sm_higgs_electric_divergence,
    sm_higgs_gauss_constraint,
    sm_higgs_left_gauge_force,
    sm_higgs_link_momentum_algebra,
    sm_higgs_project_to_coordinates,
    sm_transform_higgs_link_momenta,
)


def test_higgs_link_momentum_projection_and_sm_embedding_roundtrip() -> None:
    lattice_shape = (1, 1, 1)
    momenta = deterministic_higgs_link_momenta(lattice_shape)

    projected = sm_higgs_project_to_coordinates(sm_higgs_link_momentum_algebra(momenta))
    embedded = sm_embed_higgs_to_sm_momenta(momenta)

    assert projected.shape == momenta.shape
    assert jnp.max(jnp.abs(projected - momenta)) < 2e-6
    assert embedded.shape == (*lattice_shape, 8, 12)
    assert jnp.max(jnp.abs(embedded[..., :8])) < 1e-8
    assert jnp.max(jnp.abs(sm_extract_higgs_from_sm_momenta(embedded) - momenta)) < 1e-8


def test_higgs_link_force_vanishes_for_covariantly_constant_vacuum_and_detects_field() -> None:
    lattice_shape = (1, 1, 1)
    vacuum = sm_constant_higgs(lattice_shape)
    site_theta = deterministic_higgs_site_theta(lattice_shape)
    pure_links = sm_pure_gauge_higgs_links_from_site_algebra(site_theta)
    pure_field = sm_transform_higgs_field(vacuum, sm_higgs_site_gauge_from_algebra(site_theta))
    field = deterministic_higgs_field(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))

    assert jnp.linalg.norm(sm_higgs_left_gauge_force(pure_field, pure_links)) < 2e-6
    assert jnp.linalg.norm(sm_higgs_left_gauge_force(field, links)) > 1e-5


def test_higgs_link_force_charge_and_gauss_are_gauge_covariant() -> None:
    lattice_shape = (1, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    link_momenta = deterministic_higgs_link_momenta(lattice_shape)
    site_gauge = sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape, scale=0.17))

    transformed_field = sm_transform_higgs_field(field, site_gauge)
    transformed_higgs_momenta = sm_transform_higgs_field(higgs_momenta, site_gauge)
    transformed_links = sm_transform_higgs_links(links, site_gauge)
    transformed_link_momenta = sm_transform_higgs_link_momenta(link_momenta, site_gauge)
    force = sm_higgs_left_gauge_force(field, links)
    transformed_force = sm_higgs_left_gauge_force(transformed_field, transformed_links)
    expected_force = sm_transform_higgs_link_momenta(force, site_gauge)
    charge = sm_higgs_charge_density(field, higgs_momenta)
    transformed_charge = sm_higgs_charge_density(transformed_field, transformed_higgs_momenta)
    expected_charge = sm_higgs_project_to_coordinates(
        site_gauge
        @ sm_higgs_algebra_matrix_field(charge)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )
    gauss = sm_higgs_gauss_constraint(links, link_momenta, field, higgs_momenta)
    transformed_gauss = sm_higgs_gauss_constraint(
        transformed_links,
        transformed_link_momenta,
        transformed_field,
        transformed_higgs_momenta,
    )
    expected_gauss = sm_higgs_project_to_coordinates(
        site_gauge
        @ sm_higgs_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )

    assert jnp.max(jnp.abs(transformed_force - expected_force)) < 5e-6
    assert jnp.max(jnp.abs(transformed_charge - expected_charge)) < 2e-6
    assert jnp.max(jnp.abs(transformed_gauss - expected_gauss)) < 5e-6


def test_higgs_gauss_constraint_is_zero_for_vacuum_zero_momenta() -> None:
    lattice_shape = (1, 1, 1)
    vacuum = sm_constant_higgs(lattice_shape)
    zero_higgs_momenta = jnp.zeros_like(vacuum)
    identity_links = sm_identity_higgs_links(lattice_shape)
    zero_link_momenta = jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32)

    assert jnp.linalg.norm(sm_higgs_electric_divergence(identity_links, zero_link_momenta)) < 1e-7
    assert jnp.linalg.norm(sm_higgs_gauss_constraint(identity_links, zero_link_momenta, vacuum, zero_higgs_momenta)) < 1e-7


def test_coupled_higgs_gauge_leapfrog_is_unitary_reversible_and_nearly_hamiltonian() -> None:
    lattice_shape = (1, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    link_momenta = deterministic_higgs_link_momenta(lattice_shape)
    step_size = 0.005

    before = sm_coupled_higgs_gauge_hamiltonian_density(field, higgs_momenta, links, link_momenta)
    updated_field, updated_higgs_momenta, updated_links, updated_link_momenta = sm_coupled_higgs_gauge_leapfrog_step(
        field,
        higgs_momenta,
        links,
        link_momenta,
        step_size=step_size,
    )
    after = sm_coupled_higgs_gauge_hamiltonian_density(
        updated_field,
        updated_higgs_momenta,
        updated_links,
        updated_link_momenta,
    )
    reversed_field, reversed_higgs_momenta, reversed_links, reversed_link_momenta = (
        sm_coupled_higgs_gauge_leapfrog_step(
            updated_field,
            -updated_higgs_momenta,
            updated_links,
            -updated_link_momenta,
            step_size=step_size,
        )
    )

    assert jnp.max(jnp.abs(jnp.einsum("...ji,...jk->...ik", jnp.conj(updated_links), updated_links) - jnp.eye(2))) < 5e-7
    assert jnp.abs(after - before) < 1e-5
    assert jnp.max(jnp.abs(reversed_field - field)) < 1e-5
    assert jnp.max(jnp.abs(reversed_higgs_momenta + higgs_momenta)) < 1e-5
    assert jnp.max(jnp.abs(reversed_links - links)) < 1e-5
    assert jnp.max(jnp.abs(reversed_link_momenta + link_momenta)) < 1e-5


def test_gauge_higgs_backreaction_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_gauge_higgs_backreaction_diagnostics()

    assert diagnostics.projection_roundtrip_residual < 2e-6
    assert diagnostics.higgs_link_force_vacuum_norm < 2e-6
    assert diagnostics.higgs_link_force_nonzero_norm > 1e-5
    assert diagnostics.higgs_link_force_covariance_residual < 5e-6
    assert diagnostics.charge_covariance_residual < 2e-6
    assert diagnostics.gauss_covariance_residual < 5e-6
    assert diagnostics.gauss_vacuum_residual < 1e-7
    assert diagnostics.sm_embedding_roundtrip_residual < 1e-8
    assert diagnostics.coupled_link_unitarity_residual < 5e-7
    assert diagnostics.coupled_hamiltonian_drift < 1e-5
    assert diagnostics.coupled_reversibility_residual < 1e-5
    assert diagnostics.jit_delta_field < 1e-8
    assert diagnostics.jit_delta_links < 1e-8
    assert diagnostics.jit_delta_higgs_momenta < 1e-8
    assert diagnostics.jit_delta_link_momenta < 1e-8


def test_coupled_higgs_gauge_leapfrog_step_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    field = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    link_momenta = deterministic_higgs_link_momenta(lattice_shape)
    expected = sm_coupled_higgs_gauge_leapfrog_step(field, higgs_momenta, links, link_momenta, step_size=0.005)
    jitted = jax.jit(sm_coupled_higgs_gauge_leapfrog_step, static_argnames=("step_size", "parameters", "force_epsilon"))
    actual = jitted(field, higgs_momenta, links, link_momenta, step_size=0.005)

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.max(jnp.abs(actual_part - expected_part)) < 1e-8
