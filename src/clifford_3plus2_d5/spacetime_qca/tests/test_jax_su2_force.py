"""Session 30 JAX SU(2) Wilson-force tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    jax_average_wilson_action_density,
    jax_centered_finite_difference,
    jax_so2_pure_gauge_angles,
    jax_su2_generators,
    jax_su2_link_field_from_algebra,
    jax_su2_link_from_algebra,
    jax_su2_pure_gauge_links_from_site_algebra,
    jax_su2_site_field_from_algebra,
    jax_su2_wilson_action_density,
    jax_su2_wilson_action_gradient,
    jax_transform_link_field,
)


def _nontrivial_su2_theta(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 3), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.31)
    theta = theta.at[1, 0, 2, 3, 1].set(-0.23)
    theta = theta.at[2, 1, 1, 5, 2].set(0.17)
    theta = theta.at[1, 2, 0, 7, 0].set(0.11)
    theta = theta.at[2, 2, 1, 4, 1].set(0.09)
    return theta


def _site_algebra(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    site_theta = jnp.zeros((*shape, 3), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.05 * x - 0.02 * y)
    site_theta = site_theta.at[..., 1].set(0.03 * y + 0.04 * z)
    site_theta = site_theta.at[..., 2].set(0.07 * z - 0.01 * x)
    return site_theta


def _cartan_pure_gauge_theta(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    site_angles = 0.11 * x - 0.07 * y + 0.05 * z
    theta = jnp.zeros((*shape, 8, 3), dtype=jnp.float32)
    return theta.at[..., 2].set(jax_so2_pure_gauge_angles(site_angles))


def _direction(shape: tuple[int, int, int] = (3, 3, 3)) -> jnp.ndarray:
    x, y, z, h, a = jnp.indices((*shape, 8, 3), dtype=jnp.float32)
    raw = 0.02 * (x + 1) - 0.015 * (y + 2) + 0.01 * (z + h + 1) - 0.007 * a
    return raw / jnp.linalg.norm(raw)


def test_su2_generators_are_antihermitian_traceless_and_close() -> None:
    generators = jax_su2_generators()

    for generator in np.asarray(generators):
        np.testing.assert_allclose(generator.conj().T, -generator, atol=1e-7)
        np.testing.assert_allclose(np.trace(generator), 0, atol=1e-7)

    commutator = generators[0] @ generators[1] - generators[1] @ generators[0]
    np.testing.assert_allclose(np.asarray(commutator), np.asarray(generators[2]), atol=1e-7)


def test_su2_link_exponential_is_compact_and_linearizes() -> None:
    zero_link = jax_su2_link_from_algebra(jnp.zeros(3, dtype=jnp.float32))
    theta = jnp.asarray([0.3, -0.2, 0.1], dtype=jnp.float32)
    link = jax_su2_link_from_algebra(theta)
    epsilon = jnp.asarray(1e-4, dtype=jnp.float32)
    small_theta = epsilon * theta
    small_link = jax_su2_link_from_algebra(small_theta)
    expected_linear = jnp.eye(2, dtype=jnp.complex64) + jnp.einsum(
        "a,aij->ij",
        small_theta,
        jax_su2_generators(),
    )

    np.testing.assert_allclose(np.asarray(zero_link), np.eye(2, dtype=np.complex64), atol=1e-7)
    np.testing.assert_allclose(np.asarray(link.conj().T @ link), np.eye(2, dtype=np.complex64), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.det(link)), np.asarray(1 + 0j, dtype=np.complex64), atol=1e-6)
    np.testing.assert_allclose(np.asarray(small_link), np.asarray(expected_linear), atol=1e-7)


def test_su2_zero_field_has_zero_action_and_zero_gradient() -> None:
    theta = jnp.zeros((3, 3, 3, 8, 3), dtype=jnp.float32)

    action = jax_su2_wilson_action_density(theta)
    gradient = jax_su2_wilson_action_gradient(theta)

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(gradient)), np.asarray(0, dtype=np.float32), atol=1e-7)


def test_su2_cartan_pure_gauge_coordinates_have_zero_action_and_gradient() -> None:
    theta = _cartan_pure_gauge_theta()

    action = jax_su2_wilson_action_density(theta)
    gradient = jax_su2_wilson_action_gradient(theta)

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(gradient)), np.asarray(0, dtype=np.float32), atol=1e-6)


def test_su2_nontrivial_field_has_positive_action_and_nonzero_gradient() -> None:
    theta = _nontrivial_su2_theta()

    action = jax_su2_wilson_action_density(theta)
    gradient = jax_su2_wilson_action_gradient(theta)

    assert float(action) > 0
    assert float(jnp.linalg.norm(gradient)) > 0


def test_su2_gradient_matches_centered_directional_finite_difference() -> None:
    theta = _nontrivial_su2_theta()
    direction = _direction()

    gradient = jax_su2_wilson_action_gradient(theta)
    directional_derivative = jnp.sum(gradient * direction)
    finite_difference = jax_centered_finite_difference(
        jax_su2_wilson_action_density,
        theta,
        direction,
        epsilon=1e-2,
    )

    np.testing.assert_allclose(
        np.asarray(directional_derivative),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=3e-4,
    )


def test_su2_finite_gauge_transform_preserves_action_density() -> None:
    theta = _nontrivial_su2_theta()
    links = jax_su2_link_field_from_algebra(theta)
    site_gauge = jax_su2_site_field_from_algebra(_site_algebra())
    transformed = jax_transform_link_field(links, site_gauge)

    original_action = jax_average_wilson_action_density(links)
    transformed_action = jax_average_wilson_action_density(transformed)

    np.testing.assert_allclose(np.asarray(original_action), np.asarray(transformed_action), atol=2e-6)


def test_su2_finite_pure_gauge_links_have_zero_action() -> None:
    links = jax_su2_pure_gauge_links_from_site_algebra(_site_algebra())

    action = jax_average_wilson_action_density(links)

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=2e-6)


def test_su2_gradient_is_jittable_and_shape_preserving() -> None:
    theta = _nontrivial_su2_theta()

    gradient = jax.jit(jax_su2_wilson_action_gradient)(theta)

    assert gradient.shape == theta.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
