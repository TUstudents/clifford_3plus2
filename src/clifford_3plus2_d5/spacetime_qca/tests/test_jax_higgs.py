"""JAX tests for the Session 39 site-local Higgs field layer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca.jax_higgs import (
    jax_constant_higgs_field,
    jax_higgs_covariant_difference,
    jax_higgs_energy_density,
    jax_higgs_generators,
    jax_higgs_kinetic_energy_density,
    jax_higgs_link_field_from_algebra,
    jax_higgs_potential_density,
    jax_higgs_pure_gauge_links_from_site_algebra,
    jax_higgs_site_gauge_from_algebra,
    jax_higgs_yukawa_hamiltonian_field,
    jax_higgs_yukawa_internal_control_field,
    jax_transform_higgs_field,
    jax_transform_higgs_links,
)
from clifford_3plus2_d5.spacetime_qca.jax_yukawa import (
    jax_neutral_yukawa_hamiltonian,
    jax_neutral_yukawa_internal_control,
)


def _sample_phi() -> jnp.ndarray:
    lattice_shape = (2, 2, 2)
    values = jnp.arange(np.prod(lattice_shape), dtype=jnp.float32).reshape(lattice_shape)
    return jnp.stack(
        (
            0.1 * values + 0.05j * (values + 1),
            1.0 + 0.07 * values - 0.02j * values,
        ),
        axis=-1,
    )


def _sample_site_theta() -> jnp.ndarray:
    lattice_shape = (2, 2, 2)
    values = jnp.arange(np.prod(lattice_shape) * 4, dtype=jnp.float32).reshape((*lattice_shape, 4))
    return 0.03 * values


def _sample_link_theta() -> jnp.ndarray:
    lattice_shape = (2, 2, 2)
    values = jnp.arange(np.prod(lattice_shape) * 8 * 4, dtype=jnp.float32).reshape((*lattice_shape, 8, 4))
    return 0.005 * values


def test_higgs_generators_are_antihermitian_and_pin_em_charges() -> None:
    generators = jax_higgs_generators()
    assert generators.shape == (4, 2, 2)
    for generator in np.asarray(generators):
        np.testing.assert_allclose(generator.conj().T, -generator, atol=1e-6)

    electromagnetic_charge = 1j * (generators[2] + generators[3])
    np.testing.assert_allclose(np.asarray(electromagnetic_charge), np.diag((1.0, 0.0)), atol=1e-6)


def test_higgs_field_and_link_shapes_and_link_unitarity() -> None:
    phi = jax_constant_higgs_field((2, 1, 2), phi_plus=1.0 + 2.0j, phi_zero=3.0 - 1.0j)
    links = jax_higgs_link_field_from_algebra(jnp.zeros((2, 1, 2, 8, 4), dtype=jnp.float32))
    assert phi.shape == (2, 1, 2, 2)
    assert links.shape == (2, 1, 2, 8, 2, 2)

    identity = jnp.eye(2, dtype=links.dtype)
    unitarity = jnp.einsum("...ba,...bc->...ac", jnp.conj(links), links)
    np.testing.assert_allclose(
        np.asarray(unitarity),
        np.broadcast_to(np.asarray(identity), (2, 1, 2, 8, 2, 2)),
        atol=1e-6,
    )


def test_higgs_covariant_difference_is_gauge_covariant() -> None:
    phi = _sample_phi()
    links = jax_higgs_link_field_from_algebra(_sample_link_theta())
    site_gauge = jax_higgs_site_gauge_from_algebra(_sample_site_theta())

    transformed_phi = jax_transform_higgs_field(phi, site_gauge)
    transformed_links = jax_transform_higgs_links(links, site_gauge)

    original_difference = jax_higgs_covariant_difference(phi, links)
    transformed_difference = jax_higgs_covariant_difference(transformed_phi, transformed_links)
    expected = jnp.einsum("...ab,...hb->...ha", site_gauge, original_difference)
    np.testing.assert_allclose(np.asarray(transformed_difference), np.asarray(expected), atol=2e-5)


def test_higgs_norm_kinetic_and_total_energy_are_gauge_invariant() -> None:
    phi = _sample_phi()
    links = jax_higgs_link_field_from_algebra(_sample_link_theta())
    site_gauge = jax_higgs_site_gauge_from_algebra(_sample_site_theta())

    transformed_phi = jax_transform_higgs_field(phi, site_gauge)
    transformed_links = jax_transform_higgs_links(links, site_gauge)

    original_norm = jnp.sum(jnp.conj(phi) * phi, axis=-1)
    transformed_norm = jnp.sum(jnp.conj(transformed_phi) * transformed_phi, axis=-1)
    np.testing.assert_allclose(np.asarray(transformed_norm), np.asarray(original_norm), atol=2e-5)
    np.testing.assert_allclose(
        np.asarray(jax_higgs_kinetic_energy_density(transformed_phi, transformed_links)),
        np.asarray(jax_higgs_kinetic_energy_density(phi, links)),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax_higgs_energy_density(transformed_phi, transformed_links, vev_squared=0.7, quartic=1.3)),
        np.asarray(jax_higgs_energy_density(phi, links, vev_squared=0.7, quartic=1.3)),
        atol=2e-5,
    )


def test_pure_gauge_links_make_transformed_constant_higgs_covariantly_constant() -> None:
    site_theta = _sample_site_theta()
    site_gauge = jax_higgs_site_gauge_from_algebra(site_theta)
    phi0 = jax_constant_higgs_field((2, 2, 2), phi_plus=0.0 + 0.0j, phi_zero=1.0 + 0.0j)
    phi = jax_transform_higgs_field(phi0, site_gauge)
    links = jax_higgs_pure_gauge_links_from_site_algebra(site_theta)

    difference = jax_higgs_covariant_difference(phi, links)
    np.testing.assert_allclose(np.asarray(difference), np.zeros((2, 2, 2, 8, 2), dtype=np.complex64), atol=2e-5)
    np.testing.assert_allclose(
        np.asarray(jax_higgs_kinetic_energy_density(phi, links)),
        np.zeros((2, 2, 2), dtype=np.float32),
        atol=2e-5,
    )


def test_higgs_potential_depends_only_on_norm_and_vanishes_at_neutral_vev() -> None:
    neutral = jax_constant_higgs_field((1, 1, 1), phi_plus=0.0 + 0.0j, phi_zero=1.0 + 0.0j)
    charged = jax_constant_higgs_field((1, 1, 1), phi_plus=1.0 + 0.0j, phi_zero=0.0 + 0.0j)
    bigger = jax_constant_higgs_field((1, 1, 1), phi_plus=1.0 + 1.0j, phi_zero=0.0 + 0.0j)

    np.testing.assert_allclose(np.asarray(jax_higgs_potential_density(neutral)), np.zeros((1, 1, 1)), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(jax_higgs_potential_density(charged, vev_squared=1.0, quartic=2.0)),
        np.asarray(jax_higgs_potential_density(neutral, vev_squared=1.0, quartic=2.0)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(jax_higgs_potential_density(bigger, vev_squared=1.0, quartic=2.0)),
        np.asarray(jnp.full((1, 1, 1), 2.0)),
        atol=1e-6,
    )


@pytest.mark.slow
def test_site_local_higgs_yukawa_bridge_matches_neutral_static_helpers() -> None:
    phi = jax_constant_higgs_field((2, 1, 1), phi_plus=0.0 + 0.0j, phi_zero=1.25 + 0.0j)
    internal = jax_higgs_yukawa_internal_control_field(phi)
    hamiltonian = jax_higgs_yukawa_hamiltonian_field(phi)
    expected_internal = jax_neutral_yukawa_internal_control(1.25)
    expected_hamiltonian = jax_neutral_yukawa_hamiltonian(1.25)

    assert internal.shape == (2, 1, 1, 32, 32)
    assert hamiltonian.shape == (2, 1, 1, 128, 128)
    np.testing.assert_allclose(
        np.asarray(internal),
        np.broadcast_to(np.asarray(expected_internal), (2, 1, 1, 32, 32)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(hamiltonian),
        np.broadcast_to(np.asarray(expected_hamiltonian), (2, 1, 1, 128, 128)),
        atol=1e-6,
    )


@pytest.mark.slow
def test_zero_higgs_field_gives_zero_site_local_yukawa_matrices() -> None:
    phi = jax_constant_higgs_field((1, 2, 1), phi_plus=0.0 + 0.0j, phi_zero=0.0 + 0.0j)
    internal = jax_higgs_yukawa_internal_control_field(phi)
    hamiltonian = jax_higgs_yukawa_hamiltonian_field(phi)
    np.testing.assert_allclose(np.asarray(internal), np.zeros((1, 2, 1, 32, 32), dtype=np.complex64))
    np.testing.assert_allclose(np.asarray(hamiltonian), np.zeros((1, 2, 1, 128, 128), dtype=np.complex64))
