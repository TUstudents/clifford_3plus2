"""Session 32 SU(2) reversible gauge-dynamics tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_su2_left_force,
    jax_su2_link_field_from_algebra,
    jax_su2_pure_gauge_links_from_site_algebra,
    jax_su2_site_field_from_algebra,
    jax_transform_link_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_dynamics import (
    jax_su2_algebra_matrix,
    jax_su2_apply_momentum_update,
    jax_su2_gauge_hamiltonian_density,
    jax_su2_leapfrog_step,
    jax_su2_momentum_kinetic_energy_density,
    jax_su2_transform_momentum_field,
)

pytestmark = pytest.mark.slow


def _nontrivial_su2_theta(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    theta = jnp.zeros((*shape, 8, 3), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0, 0].set(0.31)
    theta = theta.at[1, 0, 1, 3, 1].set(-0.23)
    theta = theta.at[0, 1, 1, 5, 2].set(0.17)
    theta = theta.at[1, 1, 0, 7, 0].set(0.11)
    return theta


def _momenta(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    x, y, z, h, a = jnp.indices((*shape, 8, 3), dtype=jnp.float32)
    return 0.015 * (x + 1) - 0.011 * (y + 2) + 0.008 * (z + h + 1) - 0.006 * a


def _site_algebra(shape: tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    x, y, z = jnp.indices(shape, dtype=jnp.float32)
    site_theta = jnp.zeros((*shape, 3), dtype=jnp.float32)
    site_theta = site_theta.at[..., 0].set(0.05 * x - 0.02 * y)
    site_theta = site_theta.at[..., 1].set(0.03 * y + 0.04 * z)
    site_theta = site_theta.at[..., 2].set(0.07 * z - 0.01 * x)
    return site_theta


def _shapes():
    return tuple(canonical_bcc_plaquette_shapes())


def test_su2_algebra_matrix_is_antihermitian_and_traceless() -> None:
    algebra = jax_su2_algebra_matrix(_momenta())

    for matrix in np.asarray(algebra).reshape(-1, 2, 2):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=1e-7)
        np.testing.assert_allclose(np.trace(matrix), 0, atol=1e-7)


def test_momentum_gauge_transform_preserves_kinetic_energy_density() -> None:
    momenta = _momenta()
    site_gauge = jax_su2_site_field_from_algebra(_site_algebra())

    transformed = jax_su2_transform_momentum_field(momenta, site_gauge)

    np.testing.assert_allclose(
        np.asarray(jax_su2_momentum_kinetic_energy_density(transformed)),
        np.asarray(jax_su2_momentum_kinetic_energy_density(momenta)),
        atol=1e-7,
    )


def test_identity_links_with_zero_momenta_are_fixed_by_leapfrog() -> None:
    links = jax_su2_link_field_from_algebra(jnp.zeros((2, 2, 2, 8, 3), dtype=jnp.float32))
    momenta = jnp.zeros((2, 2, 2, 8, 3), dtype=jnp.float32)

    updated_links, updated_momenta = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=1e-7)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=1e-7)


def test_pure_gauge_links_with_zero_momenta_are_fixed_by_leapfrog() -> None:
    links = jax_su2_pure_gauge_links_from_site_algebra(_site_algebra())
    momenta = jnp.zeros((2, 2, 2, 8, 3), dtype=jnp.float32)

    updated_links, updated_momenta = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(jax_su2_left_force(links, shapes=_shapes())), np.asarray(momenta), atol=2e-6)
    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=2e-6)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=2e-6)


def test_momentum_update_and_leapfrog_preserve_compact_links() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    momenta = _momenta()

    momentum_updated = jax_su2_apply_momentum_update(links, momenta, step_size=0.05)
    leapfrog_updated, _ = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.05,
        shapes=_shapes(),
    )

    for matrix in np.asarray(jnp.concatenate((momentum_updated, leapfrog_updated), axis=0)).reshape(-1, 2, 2):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(2), atol=2e-6)
        np.testing.assert_allclose(np.linalg.det(matrix), 1 + 0j, atol=2e-6)


def test_leapfrog_is_reversible() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    momenta = _momenta()

    forward_links, forward_momenta = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.02,
        shapes=_shapes(),
    )
    recovered_links, recovered_momenta = jax_su2_leapfrog_step(
        forward_links,
        forward_momenta,
        step_size=-0.02,
        shapes=_shapes(),
    )

    np.testing.assert_allclose(np.asarray(recovered_links), np.asarray(links), atol=3e-6)
    np.testing.assert_allclose(np.asarray(recovered_momenta), np.asarray(momenta), atol=3e-6)


def test_leapfrog_energy_drift_is_small_and_step_size_sensitive() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    momenta = _momenta()
    initial = jax_su2_gauge_hamiltonian_density(links, momenta, shapes=_shapes())

    links_small, momenta_small = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.01,
        shapes=_shapes(),
    )
    links_large, momenta_large = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.04,
        shapes=_shapes(),
    )
    small_drift = jnp.abs(jax_su2_gauge_hamiltonian_density(links_small, momenta_small, shapes=_shapes()) - initial)
    large_drift = jnp.abs(jax_su2_gauge_hamiltonian_density(links_large, momenta_large, shapes=_shapes()) - initial)

    assert float(small_drift) < 2e-5
    assert float(small_drift) < float(large_drift)


def test_leapfrog_is_gauge_covariant() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    momenta = _momenta()
    site_gauge = jax_su2_site_field_from_algebra(_site_algebra())
    transformed_links = jax_transform_link_field(links, site_gauge)
    transformed_momenta = jax_su2_transform_momentum_field(momenta, site_gauge)

    evolved_links, evolved_momenta = jax_su2_leapfrog_step(
        links,
        momenta,
        step_size=0.02,
        shapes=_shapes(),
    )
    evolved_transformed_links, evolved_transformed_momenta = jax_su2_leapfrog_step(
        transformed_links,
        transformed_momenta,
        step_size=0.02,
        shapes=_shapes(),
    )

    expected_links = jax_transform_link_field(evolved_links, site_gauge)
    expected_momenta = jax_su2_transform_momentum_field(evolved_momenta, site_gauge)
    np.testing.assert_allclose(np.asarray(evolved_transformed_links), np.asarray(expected_links), atol=4e-5)
    np.testing.assert_allclose(np.asarray(evolved_transformed_momenta), np.asarray(expected_momenta), atol=4e-5)


def test_leapfrog_is_jittable_and_shape_preserving() -> None:
    links = jax_su2_link_field_from_algebra(_nontrivial_su2_theta())
    momenta = _momenta()

    updated_links, updated_momenta = jax.jit(lambda link_field, momentum_field: jax_su2_leapfrog_step(
        link_field,
        momentum_field,
        step_size=0.02,
    ))(links, momenta)

    assert updated_links.shape == links.shape
    assert updated_momenta.shape == momenta.shape
    assert bool(jnp.all(jnp.isfinite(updated_links)))
    assert bool(jnp.all(jnp.isfinite(updated_momenta)))
