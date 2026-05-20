"""Session 34 generic basis-based compact Lie dynamics tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_average_wilson_action_density,
    jax_compact_lie_action_descent_step,
    jax_compact_lie_algebra_matrix,
    jax_compact_lie_apply_momentum_update,
    jax_compact_lie_gauge_hamiltonian_density,
    jax_compact_lie_left_force,
    jax_compact_lie_link_field_from_algebra,
    jax_compact_lie_link_from_algebra,
    jax_compact_lie_momentum_kinetic_energy_density,
    jax_compact_lie_project_to_coordinates,
    jax_compact_lie_pure_gauge_links_from_site_algebra,
    jax_compact_lie_site_field_from_algebra,
    jax_compact_lie_transform_momentum_field,
    jax_su2_generators,
    jax_transform_link_field,
)


def _shapes():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _theta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 3), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.23)
    theta = theta.at[0, 0, 0, 1, 1].set(-0.17)
    theta = theta.at[0, 0, 0, 2, 2].set(0.11)
    theta = theta.at[0, 0, 0, 3, 0].set(-0.09)
    return theta


def _momenta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    _, _, _, hop, algebra = jnp.indices((*shape, 8, 3), dtype=jnp.float32)
    return 0.012 * (hop + 1) - 0.004 * algebra


def _site_theta(shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    site_theta = jnp.zeros((*shape, 3), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.08)
    site_theta = site_theta.at[..., 1].set(-0.05)
    return site_theta


def test_compact_lie_projection_handles_non_unit_generator_norms() -> None:
    generators = jax_su2_generators()
    scaled_generators = generators * jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float32)[:, None, None]
    theta = jnp.asarray([0.2, -0.3, 0.4], dtype=jnp.float32)

    algebra = jax_compact_lie_algebra_matrix(theta, scaled_generators)
    coordinates = jax_compact_lie_project_to_coordinates(algebra, scaled_generators)

    np.testing.assert_allclose(np.asarray(coordinates), np.asarray(theta), atol=1e-6)


def test_compact_lie_link_exponential_preserves_unitarity() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_link_from_algebra(
        jnp.asarray([[0.2, -0.1, 0.05]], dtype=jnp.float32),
        generators,
    )

    matrix = np.asarray(links[0])
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=2e-6)
    np.testing.assert_allclose(np.linalg.det(matrix), 1 + 0j, atol=2e-6)


def test_compact_lie_pure_gauge_has_zero_action_and_force() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_pure_gauge_links_from_site_algebra(_site_theta(), generators)

    force = jax_compact_lie_left_force(links, generators, shapes=_shapes())
    action = jax_average_wilson_action_density(links, _shapes())

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(force)), np.asarray(0, dtype=np.float32), atol=2e-6)


def test_compact_lie_descent_lowers_nonflat_action() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_link_field_from_algebra(_theta(), generators)
    action_before = jax_average_wilson_action_density(links, _shapes())

    updated, force = jax_compact_lie_action_descent_step(
        links,
        generators,
        step_size=0.5,
        shapes=_shapes(),
    )
    action_after = jax_average_wilson_action_density(updated, _shapes())

    assert float(jnp.linalg.norm(force)) > 0
    assert float(action_after) < float(action_before)


def test_compact_lie_batched_finite_difference_matches_scalar() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_link_field_from_algebra(_theta(), generators)

    scalar = jax_compact_lie_left_force(
        links,
        generators,
        shapes=_shapes(),
        method="finite_difference",
        epsilon=5e-3,
    )
    batched = jax_compact_lie_left_force(
        links,
        generators,
        shapes=_shapes(),
        method="finite_difference_batched",
        epsilon=5e-3,
        chunk_size=5,
    )

    np.testing.assert_allclose(np.asarray(batched), np.asarray(scalar), atol=1e-5)


def test_compact_lie_momentum_transform_preserves_gram_kinetic_energy() -> None:
    generators = jax_su2_generators()
    momenta = _momenta()
    site_gauge = jax_compact_lie_site_field_from_algebra(_site_theta(), generators)

    transformed = jax_compact_lie_transform_momentum_field(momenta, site_gauge, generators)

    np.testing.assert_allclose(
        np.asarray(jax_compact_lie_momentum_kinetic_energy_density(transformed, generators)),
        np.asarray(jax_compact_lie_momentum_kinetic_energy_density(momenta, generators)),
        atol=2e-7,
    )


def test_compact_lie_hamiltonian_and_momentum_update_are_shape_stable() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_link_field_from_algebra(_theta(), generators)
    momenta = _momenta()

    updated = jax_compact_lie_apply_momentum_update(links, momenta, generators, step_size=0.02)
    hamiltonian = jax_compact_lie_gauge_hamiltonian_density(links, momenta, generators, shapes=_shapes())

    assert updated.shape == links.shape
    assert bool(jnp.isfinite(hamiltonian))
    for matrix in np.asarray(updated).reshape(-1, 2, 2):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=2e-6)


def test_compact_lie_gauge_transform_uses_same_link_convention() -> None:
    generators = jax_su2_generators()
    links = jax_compact_lie_link_field_from_algebra(_theta(), generators)
    site_gauge = jax_compact_lie_site_field_from_algebra(_site_theta(), generators)

    transformed = jax_transform_link_field(links, site_gauge)

    assert transformed.shape == links.shape
    assert bool(jnp.all(jnp.isfinite(transformed)))
