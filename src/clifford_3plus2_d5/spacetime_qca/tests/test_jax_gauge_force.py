"""Session 29 JAX Wilson-action gradient tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    jax_centered_finite_difference,
    jax_so2_link_field_from_angles,
    jax_so2_pure_gauge_angles,
    jax_so2_rotation,
    jax_so2_wilson_action_density,
    jax_so2_wilson_action_gradient,
)


def _nontrivial_theta(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0].set(0.35)
    theta = theta.at[1, 0, 2, 3].set(-0.20)
    theta = theta.at[2, 1, 1, 5].set(0.15)
    theta = theta.at[1, 2, 0, 7].set(0.10)
    return theta


def _site_angles(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    return 0.11 * x - 0.07 * y + 0.05 * z


def _direction(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    x, y, z, h = jnp.indices((*shape, 8), dtype=jnp.float32)
    raw = 0.03 * (x + 1) - 0.02 * (y + 2) + 0.01 * (z + h)
    return raw / jnp.linalg.norm(raw)


def test_so2_rotation_and_link_field_shapes_are_orthogonal() -> None:
    angles = jnp.asarray([0.0, 0.4, -0.7], dtype=jnp.float32)
    rotations = jax_so2_rotation(angles)
    links = jax_so2_link_field_from_angles(jnp.zeros((2, 2, 2, 8), dtype=jnp.float32))

    np.testing.assert_allclose(np.asarray(rotations.shape), np.asarray((3, 2, 2)))
    np.testing.assert_allclose(np.asarray(links.shape), np.asarray((2, 2, 2, 8, 2, 2)))
    for matrix in np.asarray(rotations):
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=1e-6)


def test_zero_angles_have_zero_action_and_zero_gradient() -> None:
    theta = jnp.zeros((3, 3, 3, 8), dtype=jnp.float32)

    action = jax_so2_wilson_action_density(theta)
    gradient = jax_so2_wilson_action_gradient(theta)

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(gradient)), np.asarray(0, dtype=np.float32), atol=1e-7)


def test_pure_gauge_angles_have_zero_action_and_zero_gradient() -> None:
    theta = jax_so2_pure_gauge_angles(_site_angles())

    action = jax_so2_wilson_action_density(theta)
    gradient = jax_so2_wilson_action_gradient(theta)

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(gradient)), np.asarray(0, dtype=np.float32), atol=1e-6)


def test_nontrivial_angles_have_positive_action_and_nonzero_gradient() -> None:
    theta = _nontrivial_theta()

    action = jax_so2_wilson_action_density(theta)
    gradient = jax_so2_wilson_action_gradient(theta)

    assert float(action) > 0
    assert float(jnp.linalg.norm(gradient)) > 0


def test_gradient_matches_centered_directional_finite_difference() -> None:
    theta = _nontrivial_theta()
    direction = _direction()

    gradient = jax_so2_wilson_action_gradient(theta)
    directional_derivative = jnp.sum(gradient * direction)
    finite_difference = jax_centered_finite_difference(
        jax_so2_wilson_action_density,
        theta,
        direction,
        epsilon=1e-2,
    )

    np.testing.assert_allclose(
        np.asarray(directional_derivative),
        np.asarray(finite_difference),
        rtol=2e-2,
        atol=2e-4,
    )


def test_gradient_is_orthogonal_to_pure_gauge_directions() -> None:
    theta = _nontrivial_theta()
    gauge_direction = jax_so2_pure_gauge_angles(_site_angles())

    gradient = jax_so2_wilson_action_gradient(theta)
    gauge_component = jnp.sum(gradient * gauge_direction)

    np.testing.assert_allclose(np.asarray(gauge_component), np.asarray(0, dtype=np.float32), atol=2e-6)


def test_gradient_is_jittable_and_shape_preserving() -> None:
    theta = _nontrivial_theta()

    gradient = jax.jit(jax_so2_wilson_action_gradient)(theta)

    assert gradient.shape == theta.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
