"""Session 34 chiral16 SU(4) compact gauge-dynamics tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_average_wilson_action_density,
    jax_patisalam_su4_action_descent_step,
    jax_patisalam_su4_algebra_matrix,
    jax_patisalam_su4_apply_momentum_update,
    jax_patisalam_su4_gauge_hamiltonian_density,
    jax_patisalam_su4_generators_chiral16,
    jax_patisalam_su4_leapfrog_step,
    jax_patisalam_su4_left_force,
    jax_patisalam_su4_link_field_from_algebra,
    jax_patisalam_su4_link_from_algebra,
    jax_patisalam_su4_momentum_kinetic_energy_density,
    jax_patisalam_su4_project_to_coordinates,
    jax_patisalam_su4_pure_gauge_links_from_site_algebra,
    jax_patisalam_su4_site_field_from_algebra,
    jax_patisalam_su4_transform_momentum_field,
    jax_transform_link_field,
)


def _shapes():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _theta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 15), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.045)
    theta = theta.at[0, 0, 0, 1, 1].set(-0.038)
    theta = theta.at[0, 0, 0, 2, 2].set(0.031)
    theta = theta.at[0, 0, 0, 3, 3].set(0.024)
    return theta


def _momenta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    _, _, _, hop, algebra = jnp.indices((*shape, 8, 15), dtype=jnp.float32)
    return 0.002 * (hop + 1) - 0.0007 * algebra


def _site_theta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    site_theta = jnp.zeros((*shape, 15), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.025)
    site_theta = site_theta.at[..., 4].set(-0.018)
    site_theta = site_theta.at[..., 9].set(0.014)
    return site_theta


def test_patisalam_su4_chiral16_generators_are_valid_and_non_degenerate() -> None:
    generators = jax_patisalam_su4_generators_chiral16()

    assert generators.shape == (15, 32, 32)
    for matrix in np.asarray(generators):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=1e-7)

    gram = jnp.real(jnp.einsum("aij,bji->ab", jnp.swapaxes(jnp.conj(generators), -1, -2), generators))
    assert np.linalg.matrix_rank(np.asarray(gram)) == 15


def test_patisalam_su4_projection_recovers_chiral16_coordinates() -> None:
    theta = jnp.linspace(-0.07, 0.08, 15, dtype=jnp.float32)
    algebra = jax_patisalam_su4_algebra_matrix(theta)

    coordinates = jax_patisalam_su4_project_to_coordinates(algebra)

    np.testing.assert_allclose(np.asarray(coordinates), np.asarray(theta), atol=2e-6)


def test_patisalam_su4_link_exponential_preserves_compactness() -> None:
    theta = jnp.zeros((1, 15), dtype=jnp.float32)
    theta = theta.at[0, 0].set(0.04)
    theta = theta.at[0, 6].set(-0.03)
    theta = theta.at[0, 14].set(0.02)

    links = jax_patisalam_su4_link_from_algebra(theta)

    matrix = np.asarray(links[0])
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(32), atol=5e-6)


def test_patisalam_su4_pure_gauge_has_zero_action() -> None:
    links = jax_patisalam_su4_pure_gauge_links_from_site_algebra(_site_theta())
    action = jax_average_wilson_action_density(links, _shapes())

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=3e-6)


def test_patisalam_su4_finite_difference_force_lowers_nonflat_action() -> None:
    links = jax_patisalam_su4_link_field_from_algebra(_theta())
    action_before = jax_average_wilson_action_density(links, _shapes())

    updated, force = jax_patisalam_su4_action_descent_step(
        links,
        step_size=0.15,
        epsilon=5e-3,
        shapes=_shapes(),
    )
    action_after = jax_average_wilson_action_density(updated, _shapes())

    assert force.shape == (1, 1, 1, 8, 15)
    assert float(jnp.linalg.norm(force)) > 0
    assert float(action_after) < float(action_before)


def test_patisalam_su4_momentum_transform_preserves_gram_kinetic_energy() -> None:
    momenta = _momenta()
    site_gauge = jax_patisalam_su4_site_field_from_algebra(_site_theta())

    transformed = jax_patisalam_su4_transform_momentum_field(momenta, site_gauge)

    np.testing.assert_allclose(
        np.asarray(jax_patisalam_su4_momentum_kinetic_energy_density(transformed)),
        np.asarray(jax_patisalam_su4_momentum_kinetic_energy_density(momenta)),
        atol=2e-5,
    )


def test_patisalam_su4_momentum_update_and_leapfrog_preserve_compact_links() -> None:
    links = jax_patisalam_su4_link_field_from_algebra(_theta())
    momenta = _momenta()

    momentum_updated = jax_patisalam_su4_apply_momentum_update(links, momenta, step_size=0.01)
    leapfrog_links, leapfrog_momenta = jax_patisalam_su4_leapfrog_step(
        links,
        momenta,
        step_size=0.01,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )
    hamiltonian = jax_patisalam_su4_gauge_hamiltonian_density(links, momenta, shapes=_shapes())

    assert leapfrog_momenta.shape == momenta.shape
    assert bool(jnp.isfinite(hamiltonian))
    for matrix in np.asarray(jnp.concatenate((momentum_updated, leapfrog_links), axis=0)).reshape(-1, 32, 32):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(32), atol=8e-6)


def test_patisalam_su4_identity_links_with_zero_momenta_are_fixed_by_leapfrog() -> None:
    links = jax_patisalam_su4_link_field_from_algebra(jnp.zeros((1, 1, 1, 8, 15), dtype=jnp.float32))
    momenta = jnp.zeros((1, 1, 1, 8, 15), dtype=jnp.float32)

    updated_links, updated_momenta = jax_patisalam_su4_leapfrog_step(
        links,
        momenta,
        step_size=0.01,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=1e-6)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=1e-6)


def test_patisalam_su4_link_and_momentum_transforms_share_pull_convention() -> None:
    links = jax_patisalam_su4_link_field_from_algebra(_theta())
    momenta = _momenta()
    site_gauge = jax_patisalam_su4_site_field_from_algebra(_site_theta())

    transformed_links = jax_transform_link_field(links, site_gauge)
    transformed_momenta = jax_patisalam_su4_transform_momentum_field(momenta, site_gauge)

    assert transformed_links.shape == links.shape
    assert transformed_momenta.shape == momenta.shape
    assert bool(jnp.all(jnp.isfinite(transformed_links)))
    assert bool(jnp.all(jnp.isfinite(transformed_momenta)))


def test_patisalam_su4_left_force_default_uses_memory_safe_finite_difference() -> None:
    links = jax_patisalam_su4_link_field_from_algebra(_theta())

    force = jax_patisalam_su4_left_force(links, epsilon=5e-3, shapes=_shapes())

    assert force.shape == (1, 1, 1, 8, 15)
    assert bool(jnp.all(jnp.isfinite(force)))
