"""Session 33 SU(3) compact force and reversible dynamics tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_average_wilson_action_density,
    jax_su3_action_descent_step,
    jax_su3_apply_left_update,
    jax_su3_apply_momentum_update,
    jax_su3_gauge_hamiltonian_density,
    jax_su3_generators,
    jax_su3_leapfrog_step,
    jax_su3_left_force,
    jax_su3_left_force_from_algebra,
    jax_su3_link_field_from_algebra,
    jax_su3_link_from_algebra,
    jax_su3_momentum_kinetic_energy_density,
    jax_su3_project_antihermitian_to_algebra,
    jax_su3_pure_gauge_links_from_site_algebra,
    jax_su3_site_field_from_algebra,
    jax_su3_transform_momentum_field,
    jax_transform_link_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_dynamics import jax_su3_algebra_matrix

pytestmark = pytest.mark.slow


def _nontrivial_su3_theta(shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 8), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.21)
    theta = theta.at[0, 0, 0, 1, 1].set(-0.16)
    theta = theta.at[0, 0, 0, 2, 2].set(0.13)
    theta = theta.at[0, 0, 0, 3, 3].set(0.09)
    theta = theta.at[0, 0, 0, 4, 7].set(-0.07)
    theta = theta.at[0, 0, 0, 5, 5].set(0.05)
    theta = theta.at[0, 0, 0, 6, 4].set(-0.04)
    theta = theta.at[1, 0, 0, 0, 3].set(-0.08)
    theta = theta.at[1, 0, 0, 2, 6].set(0.06)
    theta = theta.at[1, 0, 0, 7, 1].set(0.04)
    return theta


def _momenta(shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    x, y, z, h, a = jnp.indices((*shape, 8, 8), dtype=jnp.float32)
    return 0.01 * (x + 1) - 0.007 * (y + 2) + 0.005 * (z + h + 1) - 0.003 * a


def _site_algebra(shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    site_theta = jnp.zeros((*shape, 8), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.07 + 0.04 * x - 0.015 * y)
    site_theta = site_theta.at[..., 1].set(-0.05 + 0.025 * y + 0.03 * z)
    site_theta = site_theta.at[..., 2].set(0.035 * z - 0.01 * x)
    site_theta = site_theta.at[..., 3].set(0.03 + 0.02 * (x + y))
    site_theta = site_theta.at[..., 4].set(-0.02 - 0.018 * (y + z))
    site_theta = site_theta.at[..., 7].set(0.04 + 0.015 * (z - x))
    return site_theta


def _left_force_direction(shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    x, y, z, h, a = jnp.indices((*shape, 8, 8), dtype=jnp.float32)
    raw = 0.013 * (x + 1) - 0.011 * (y + 2) + 0.007 * (z + h + 1) - 0.004 * a
    return raw / jnp.linalg.norm(raw)


def _shapes():
    return tuple(canonical_bcc_plaquette_shapes())


def test_su3_generators_are_antihermitian_traceless_and_normalized() -> None:
    generators = jax_su3_generators()

    for matrix in np.asarray(generators):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=1e-7)
        np.testing.assert_allclose(np.trace(matrix), 0, atol=1e-7)

    inner = jnp.einsum("aij,bji->ab", jnp.swapaxes(jnp.conj(generators), -1, -2), generators)
    np.testing.assert_allclose(np.asarray(inner), 0.5 * np.eye(8), atol=1e-7)


def test_su3_projection_recovers_generator_coordinates() -> None:
    theta = jnp.asarray([0.2, -0.3, 0.4, 0.1, -0.08, 0.05, 0.07, -0.09], dtype=jnp.float32)
    algebra = jnp.einsum("a,aij->ij", theta, jax_su3_generators())

    coordinates = jax_su3_project_antihermitian_to_algebra(algebra)
    zero_coordinates = jax_su3_project_antihermitian_to_algebra(
        jnp.zeros((3, 3), dtype=jnp.complex64),
    )

    np.testing.assert_allclose(np.asarray(coordinates), np.asarray(theta), atol=1e-7)
    np.testing.assert_allclose(np.asarray(zero_coordinates), np.zeros(8), atol=1e-7)


def test_su3_link_exponential_preserves_compactness() -> None:
    links = jax_su3_link_from_algebra(
        jnp.asarray([[0.2, -0.1, 0.05, 0.04, -0.03, 0.02, 0.01, -0.07]], dtype=jnp.float32),
    )

    for matrix in np.asarray(links).reshape(-1, 3, 3):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3), atol=3e-6)
        np.testing.assert_allclose(np.linalg.det(matrix), 1 + 0j, atol=3e-6)


def test_su3_zero_field_has_zero_action_and_force() -> None:
    theta = jnp.zeros((1, 1, 1, 8, 8), dtype=jnp.float32)

    force = jax_su3_left_force_from_algebra(theta, shapes=_shapes())
    action = jax_average_wilson_action_density(jax_su3_link_field_from_algebra(theta), _shapes())

    assert force.shape == theta.shape
    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(force)), np.asarray(0, dtype=np.float32), atol=1e-7)


def test_su3_pure_gauge_has_zero_action_and_force() -> None:
    links = jax_su3_pure_gauge_links_from_site_algebra(_site_algebra())

    force = jax_su3_left_force(links, shapes=_shapes())
    action = jax_average_wilson_action_density(links, _shapes())

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=4e-6)
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(force)), np.asarray(0, dtype=np.float32), atol=2e-5)


def test_su3_left_force_matches_centered_left_directional_difference() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    force = jax_su3_left_force(links, shapes=_shapes())
    direction = _left_force_direction()
    epsilon = jnp.asarray(1e-2, dtype=jnp.float32)

    def action_along(scale: jnp.ndarray) -> jnp.ndarray:
        perturbed = jax_su3_apply_left_update(links, -direction, step_size=scale)
        return jax_average_wilson_action_density(perturbed, _shapes())

    directional_derivative = jnp.sum(force * direction)
    finite_difference = (action_along(epsilon) - action_along(-epsilon)) / (2 * epsilon)

    np.testing.assert_allclose(
        np.asarray(directional_derivative),
        np.asarray(finite_difference),
        rtol=8e-2,
        atol=8e-4,
    )


def test_su3_left_force_nonflat_field_is_nonzero_and_descent_lowers_action() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    action_before = jax_average_wilson_action_density(links, _shapes())

    updated, force = jax_su3_action_descent_step(
        links,
        step_size=0.5,
        shapes=_shapes(),
    )
    action_after = jax_average_wilson_action_density(updated, _shapes())

    assert float(jnp.linalg.norm(force)) > 0
    assert float(action_after) < float(action_before)


def test_su3_algebra_matrix_is_antihermitian_and_traceless() -> None:
    algebra = jax_su3_algebra_matrix(_momenta())

    for matrix in np.asarray(algebra).reshape(-1, 3, 3):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=1e-7)
        np.testing.assert_allclose(np.trace(matrix), 0, atol=1e-7)


def test_su3_momentum_gauge_transform_preserves_kinetic_energy_density() -> None:
    momenta = _momenta()
    site_gauge = jax_su3_site_field_from_algebra(_site_algebra())

    transformed = jax_su3_transform_momentum_field(momenta, site_gauge)

    np.testing.assert_allclose(
        np.asarray(jax_su3_momentum_kinetic_energy_density(transformed)),
        np.asarray(jax_su3_momentum_kinetic_energy_density(momenta)),
        atol=2e-7,
    )


def test_su3_identity_links_with_zero_momenta_are_fixed_by_leapfrog() -> None:
    links = jax_su3_link_field_from_algebra(jnp.zeros((1, 1, 1, 8, 8), dtype=jnp.float32))
    momenta = jnp.zeros((1, 1, 1, 8, 8), dtype=jnp.float32)

    updated_links, updated_momenta = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=1e-7)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=1e-7)


def test_su3_pure_gauge_links_with_zero_momenta_are_fixed_by_leapfrog() -> None:
    links = jax_su3_pure_gauge_links_from_site_algebra(_site_algebra())
    momenta = jnp.zeros((2, 1, 1, 8, 8), dtype=jnp.float32)

    updated_links, updated_momenta = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(jax_su3_left_force(links, shapes=_shapes())), np.asarray(momenta), atol=2e-5)
    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=2e-5)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=2e-5)


def test_su3_momentum_update_and_leapfrog_preserve_compact_links() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    momenta = _momenta()

    momentum_updated = jax_su3_apply_momentum_update(links, momenta, step_size=0.05)
    leapfrog_updated, _ = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    for matrix in np.asarray(jnp.concatenate((momentum_updated, leapfrog_updated), axis=0)).reshape(-1, 3, 3):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3), atol=4e-6)
        np.testing.assert_allclose(np.linalg.det(matrix), 1 + 0j, atol=4e-6)


def test_su3_leapfrog_is_reversible() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    momenta = _momenta()

    forward_links, forward_momenta = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.02,
        shapes=_shapes(),
    )
    recovered_links, recovered_momenta = jax_su3_leapfrog_step(
        forward_links,
        forward_momenta,
        step_size=-0.02,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(recovered_links), np.asarray(links), atol=6e-5)
    np.testing.assert_allclose(np.asarray(recovered_momenta), np.asarray(momenta), atol=6e-5)


def test_su3_leapfrog_energy_drift_is_small_and_step_size_sensitive() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    momenta = _momenta()
    initial = jax_su3_gauge_hamiltonian_density(links, momenta, shapes=_shapes())

    links_small, momenta_small = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.01,
        shapes=_shapes(),
    )
    links_large, momenta_large = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.04,
        shapes=_shapes(),
    )
    small_drift = jnp.abs(jax_su3_gauge_hamiltonian_density(links_small, momenta_small, shapes=_shapes()) - initial)
    large_drift = jnp.abs(jax_su3_gauge_hamiltonian_density(links_large, momenta_large, shapes=_shapes()) - initial)

    assert float(small_drift) < 2e-5
    assert float(small_drift) < float(large_drift)


def test_su3_leapfrog_is_gauge_covariant() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    momenta = _momenta()
    site_gauge = jax_su3_site_field_from_algebra(_site_algebra())
    transformed_links = jax_transform_link_field(links, site_gauge)
    transformed_momenta = jax_su3_transform_momentum_field(momenta, site_gauge)

    evolved_links, evolved_momenta = jax_su3_leapfrog_step(
        links,
        momenta,
        step_size=0.02,
        shapes=_shapes(),
    )
    evolved_transformed_links, evolved_transformed_momenta = jax_su3_leapfrog_step(
        transformed_links,
        transformed_momenta,
        step_size=0.02,
        shapes=_shapes(),
    )

    expected_links = jax_transform_link_field(evolved_links, site_gauge)
    expected_momenta = jax_su3_transform_momentum_field(evolved_momenta, site_gauge)
    np.testing.assert_allclose(np.asarray(evolved_transformed_links), np.asarray(expected_links), atol=2e-4)
    np.testing.assert_allclose(np.asarray(evolved_transformed_momenta), np.asarray(expected_momenta), atol=2e-4)


def test_su3_compact_momentum_update_is_jittable_and_shape_preserving() -> None:
    links = jax_su3_link_field_from_algebra(_nontrivial_su3_theta())
    momenta = _momenta()

    updated_links = jax.jit(lambda link_field, momentum_field: jax_su3_apply_momentum_update(
        link_field,
        momentum_field,
        step_size=0.02,
    ))(links, momenta)

    assert updated_links.shape == links.shape
    assert bool(jnp.all(jnp.isfinite(updated_links)))
