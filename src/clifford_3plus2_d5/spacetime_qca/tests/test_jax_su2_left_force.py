"""Session 31 left-trivialized SU(2) Wilson-force tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_average_wilson_action_density,
    jax_su2_action_descent_step,
    jax_su2_apply_left_update,
    jax_su2_generators,
    jax_su2_left_force,
    jax_su2_left_force_from_algebra,
    jax_su2_link_field_from_algebra,
    jax_su2_project_antihermitian_to_algebra,
    jax_su2_pure_gauge_links_from_site_algebra,
    jax_su2_site_field_from_algebra,
    jax_transform_link_field,
)

pytestmark = pytest.mark.slow


def _nontrivial_su2_theta(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 3), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.31)
    theta = theta.at[1, 0, 1, 3, 1].set(-0.23)
    theta = theta.at[0, 1, 1, 5, 2].set(0.17)
    theta = theta.at[1, 1, 0, 7, 0].set(0.11)
    return theta


def _site_algebra(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    site_theta = jnp.zeros((*shape, 3), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.05 * x - 0.02 * y)
    site_theta = site_theta.at[..., 1].set(0.03 * y + 0.04 * z)
    site_theta = site_theta.at[..., 2].set(0.07 * z - 0.01 * x)
    return site_theta


def _left_force_direction(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    x, y, z, h, a = jnp.indices((*shape, 8, 3), dtype=jnp.float32)
    raw = 0.02 * (x + 1) - 0.015 * (y + 2) + 0.01 * (z + h + 1) - 0.007 * a
    return raw / jnp.linalg.norm(raw)


def _shapes():
    return tuple(canonical_bcc_plaquette_shapes())


def test_su2_projection_recovers_generator_coordinates() -> None:
    theta = jnp.asarray([0.2, -0.3, 0.4], dtype=jnp.float32)
    algebra = jnp.einsum("a,aij->ij", theta, jax_su2_generators())

    coordinates = jax_su2_project_antihermitian_to_algebra(algebra)
    zero_coordinates = jax_su2_project_antihermitian_to_algebra(
        jnp.zeros((2, 2), dtype=jnp.complex64),
    )

    np.testing.assert_allclose(np.asarray(coordinates), np.asarray(theta), atol=1e-7)
    np.testing.assert_allclose(np.asarray(zero_coordinates), np.zeros(3), atol=1e-7)


def test_su2_left_force_zero_field_is_zero() -> None:
    theta = jnp.zeros((2, 2, 2, 8, 3), dtype=jnp.float32)

    force = jax_su2_left_force_from_algebra(theta, shapes=_shapes())

    assert force.shape == theta.shape
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(force)), np.asarray(0, dtype=np.float32), atol=1e-7)


def test_su2_left_force_finite_pure_gauge_is_zero() -> None:
    links = jax_su2_pure_gauge_links_from_site_algebra(_site_algebra())

    force = jax_su2_left_force(links, shapes=_shapes())

    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(force)), np.asarray(0, dtype=np.float32), atol=2e-6)


def test_su2_left_force_matches_centered_left_directional_difference() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    force = jax_su2_left_force(links, shapes=_shapes())
    direction = _left_force_direction()
    epsilon = jnp.asarray(1e-2, dtype=jnp.float32)

    def action_along(scale: jnp.ndarray) -> jnp.ndarray:
        perturbed = jax_su2_apply_left_update(links, -direction, step_size=scale)
        return jax_average_wilson_action_density(perturbed, _shapes())

    directional_derivative = jnp.sum(force * direction)
    finite_difference = (action_along(epsilon) - action_along(-epsilon)) / (2 * epsilon)

    np.testing.assert_allclose(
        np.asarray(directional_derivative),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=3e-4,
    )


def test_su2_left_force_nonflat_field_is_nonzero_and_descent_lowers_action() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    action_before = jax_average_wilson_action_density(links, _shapes())

    updated, force = jax_su2_action_descent_step(
        links,
        step_size=0.5,
        shapes=_shapes(),
    )
    action_after = jax_average_wilson_action_density(updated, _shapes())

    assert float(jnp.linalg.norm(force)) > 0
    assert float(action_after) < float(action_before)


def test_su2_left_update_preserves_compactness() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    force = jax_su2_left_force(links, shapes=_shapes())

    updated = jax_su2_apply_left_update(links, force, step_size=0.5)

    for matrix in np.asarray(updated).reshape(-1, 2, 2):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(matrix), 1 + 0j, atol=2e-6)


def test_su2_left_descent_action_is_gauge_invariant() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    site_gauge = jax_su2_site_field_from_algebra(_site_algebra())
    transformed = jax_transform_link_field(links, site_gauge)

    updated, _ = jax_su2_action_descent_step(links, step_size=0.5, shapes=_shapes())
    updated_transformed, _ = jax_su2_action_descent_step(
        transformed,
        step_size=0.5,
        shapes=_shapes(),
    )

    action = jax_average_wilson_action_density(updated, _shapes())
    transformed_action = jax_average_wilson_action_density(updated_transformed, _shapes())
    np.testing.assert_allclose(np.asarray(action), np.asarray(transformed_action), atol=3e-6)


def test_su2_left_force_is_jittable_and_shape_preserving() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())

    force = jax.jit(jax_su2_left_force)(links)

    assert force.shape == (2, 2, 2, 8, 3)
    assert bool(jnp.all(jnp.isfinite(force)))
