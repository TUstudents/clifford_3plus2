"""JAX gauge-force helpers for compact BCC link fields.

Session 29 introduced the smallest compact gauge testbed: every BCC link is an
SO(2) rotation parameterized by one real angle.  Session 30 extends the same
Wilson-gradient audit to fundamental SU(2) links, the smallest compact
nonabelian testbed.  Session 31 adds a left-trivialized SU(2) force audit and
compact action-descent update.  Session 33 mirrors the compact force stack for
fundamental SU(3) color links.

* zero field has zero action and zero gradient;
* pure gauges have zero action and zero gradient;
* non-flat fields have non-zero curvature and a non-zero action gradient.

This is a force audit, not a production HMC implementation.  The
left-trivialized force is computed by differentiating compact left
perturbations ``exp(omega_a T_a) U``.  Focused finite-difference tests pin the
convention before a future vectorized staple formula.  Generic SU(N) force
projection and heatbath dynamics remain future work.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_matrix_expm

from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.links import (
    jax_identity_link_field,
    jax_transform_link_field as _generic_jax_transform_link_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_step import jax_bcc_displacements
from clifford_3plus2_d5.spacetime_qca.jax_wilson import jax_average_wilson_action_density
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape


def _validate_so2_angles(theta: jnp.ndarray) -> None:
    if theta.ndim != 4 or theta.shape[-1] != 8:
        raise ValueError("SO(2) link angles must have shape (nx, ny, nz, 8)")


def _validate_su2_link_algebra(theta: jnp.ndarray) -> None:
    if theta.ndim != 5 or theta.shape[-2:] != (8, 3):
        raise ValueError("SU(2) link algebra coordinates must have shape (nx, ny, nz, 8, 3)")


def _validate_su2_site_algebra(site_theta: jnp.ndarray) -> None:
    if site_theta.ndim != 4 or site_theta.shape[-1] != 3:
        raise ValueError("SU(2) site algebra coordinates must have shape (nx, ny, nz, 3)")


def _validate_su2_links(links: jnp.ndarray) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 2, 2):
        raise ValueError("SU(2) BCC links must have shape (nx, ny, nz, 8, 2, 2)")


def _validate_su3_link_algebra(theta: jnp.ndarray) -> None:
    if theta.ndim != 5 or theta.shape[-2:] != (8, 8):
        raise ValueError("SU(3) link algebra coordinates must have shape (nx, ny, nz, 8, 8)")


def _validate_su3_site_algebra(site_theta: jnp.ndarray) -> None:
    if site_theta.ndim != 4 or site_theta.shape[-1] != 8:
        raise ValueError("SU(3) site algebra coordinates must have shape (nx, ny, nz, 8)")


def _validate_su3_links(links: jnp.ndarray) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 3, 3):
        raise ValueError("SU(3) BCC links must have shape (nx, ny, nz, 8, 3, 3)")


def _jax_matrix_expm_batch(matrices: jnp.ndarray) -> jnp.ndarray:
    if matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrix exponentials require square trailing dimensions")

    dimension = matrices.shape[-1]
    flat = jnp.reshape(matrices, (-1, dimension, dimension))
    exponentials = jax.vmap(jax_matrix_expm)(flat)
    return jnp.reshape(exponentials, matrices.shape)


def jax_so2_rotation(theta: jnp.ndarray) -> jnp.ndarray:
    """Return SO(2) rotation matrices for scalar or array-valued angles.

    A scalar angle returns a ``(2, 2)`` matrix.  An angle array with shape
    ``(...,)`` returns matrices with shape ``(..., 2, 2)``.
    """

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return jnp.stack(
        (
            jnp.stack((cos_theta, -sin_theta), axis=-1),
            jnp.stack((sin_theta, cos_theta), axis=-1),
        ),
        axis=-2,
    )


def jax_so2_link_field_from_angles(theta: jnp.ndarray) -> jnp.ndarray:
    """Return a BCC JAX link field from SO(2) link angles.

    Input shape is ``(nx, ny, nz, 8)``.  Output shape is
    ``(nx, ny, nz, 8, 2, 2)``, matching the numerical Wilson backend layout.
    """

    _validate_so2_angles(theta)
    return jax_so2_rotation(theta)


def jax_so2_pure_gauge_angles(site_angles: jnp.ndarray) -> jnp.ndarray:
    """Return SO(2) link angles for a pure gauge ``g(x) g(x+h)^-1`` field.

    ``site_angles`` has shape ``(nx, ny, nz)`` and represents
    ``g(x) = R(phi[x])``.  The BCC pull-link convention stores the link at
    target site ``x`` for the source site ``x+h``.  The pure-gauge link is
    therefore ``R(phi[x] - phi[x+h])``.
    """

    if site_angles.ndim != 3:
        raise ValueError("site gauge angles must have shape (nx, ny, nz)")

    components = []
    for displacement in jax_bcc_displacements():
        source_angles = source_roll(site_angles, displacement)
        components.append(site_angles - source_angles)
    return jnp.stack(components, axis=-1)


def jax_so2_wilson_action_density(
    theta: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return Wilson action density for SO(2)-parameterized BCC links."""

    return jax_average_wilson_action_density(
        jax_so2_link_field_from_angles(theta),
        shapes,
    )


def jax_so2_wilson_action_gradient(
    theta: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return ``d S_W / d theta`` for SO(2)-parameterized BCC links."""

    return jax.grad(lambda angles: jax_so2_wilson_action_density(angles, shapes))(theta)


def jax_centered_finite_difference(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
    direction: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
) -> jnp.ndarray:
    """Return the centered directional finite difference of ``fn`` at ``theta``."""

    step = jnp.asarray(epsilon, dtype=theta.dtype)
    return (fn(theta + step * direction) - fn(theta - step * direction)) / (2 * step)


def jax_su2_generators(dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return anti-Hermitian fundamental SU(2) generators ``T_a = -i sigma_a/2``."""

    zero = jnp.asarray(0, dtype=dtype)
    one = jnp.asarray(1, dtype=dtype)
    imag = jnp.asarray(1j, dtype=dtype)
    sigma_x = jnp.asarray(((zero, one), (one, zero)), dtype=dtype)
    sigma_y = jnp.asarray(((zero, -imag), (imag, zero)), dtype=dtype)
    sigma_z = jnp.asarray(((one, zero), (zero, -one)), dtype=dtype)
    return -0.5j * jnp.stack((sigma_x, sigma_y, sigma_z), axis=0)


def jax_su3_generators(dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return anti-Hermitian fundamental SU(3) generators ``T_a = -i lambda_a/2``."""

    zero = jnp.asarray(0, dtype=dtype)
    one = jnp.asarray(1, dtype=dtype)
    two = jnp.asarray(2, dtype=dtype)
    imag = jnp.asarray(1j, dtype=dtype)
    inv_sqrt_three = one / jnp.sqrt(jnp.asarray(3, dtype=dtype))
    lambda_1 = jnp.asarray(((zero, one, zero), (one, zero, zero), (zero, zero, zero)), dtype=dtype)
    lambda_2 = jnp.asarray(((zero, -imag, zero), (imag, zero, zero), (zero, zero, zero)), dtype=dtype)
    lambda_3 = jnp.asarray(((one, zero, zero), (zero, -one, zero), (zero, zero, zero)), dtype=dtype)
    lambda_4 = jnp.asarray(((zero, zero, one), (zero, zero, zero), (one, zero, zero)), dtype=dtype)
    lambda_5 = jnp.asarray(((zero, zero, -imag), (zero, zero, zero), (imag, zero, zero)), dtype=dtype)
    lambda_6 = jnp.asarray(((zero, zero, zero), (zero, zero, one), (zero, one, zero)), dtype=dtype)
    lambda_7 = jnp.asarray(((zero, zero, zero), (zero, zero, -imag), (zero, imag, zero)), dtype=dtype)
    lambda_8 = inv_sqrt_three * jnp.asarray(
        ((one, zero, zero), (zero, one, zero), (zero, zero, -two)),
        dtype=dtype,
    )
    return -0.5j * jnp.stack(
        (lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8),
        axis=0,
    )


def jax_su2_project_antihermitian_to_algebra(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project anti-Hermitian traceless matrices onto SU(2) coordinates.

    The generator convention is ``T_a = -i sigma_a / 2``.  For
    ``A = theta_a T_a`` the returned coordinates are ``theta_a``.  The
    projection uses the Hilbert-Schmidt inner product
    ``theta_a = 2 Re Tr(T_a^dagger A)`` and accepts arrays with shape
    ``(..., 2, 2)``.
    """

    if matrix.shape[-2:] != (2, 2):
        raise ValueError("SU(2) algebra matrices must have trailing shape (2, 2)")

    generators = jax_su2_generators(dtype=jnp.result_type(matrix, 1j))
    generator_daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    inner_products = jnp.einsum("aij,...ji->...a", generator_daggers, matrix)
    return 2 * jnp.real(inner_products)


def jax_su3_project_antihermitian_to_algebra(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project anti-Hermitian traceless matrices onto SU(3) coordinates."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("SU(3) algebra matrices must have trailing shape (3, 3)")

    generators = jax_su3_generators(dtype=jnp.result_type(matrix, 1j))
    generator_daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    inner_products = jnp.einsum("aij,...ji->...a", generator_daggers, matrix)
    return 2 * jnp.real(inner_products)


def jax_su2_link_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return compact SU(2) matrices from Lie-algebra coordinates.

    ``theta`` has shape ``(..., 3)`` and represents
    ``exp(theta_a T_a)`` with ``T_a = -i sigma_a / 2``.  The closed form uses
    ``exp(-i theta.sigma / 2)`` with a stable small-angle branch.
    """

    if theta.shape[-1:] != (3,):
        raise ValueError("SU(2) algebra coordinates must have trailing dimension 3")

    dtype = jnp.result_type(theta, 1j)
    coordinates = theta.astype(jnp.real(jnp.asarray(0, dtype=dtype)).dtype)
    generators = jax_su2_generators(dtype=dtype)
    algebra = jnp.einsum("...a,aij->...ij", coordinates, generators)
    radius_squared = jnp.sum(coordinates * coordinates, axis=-1)
    small = radius_squared < 1e-12
    safe_radius = jnp.sqrt(jnp.where(small, jnp.ones_like(radius_squared), radius_squared))
    exact_cos = jnp.cos(0.5 * safe_radius)
    exact_scale = 2 * jnp.sin(0.5 * safe_radius) / safe_radius
    series_cos = 1 - radius_squared / 8 + radius_squared**2 / 384
    series_scale = 1 - radius_squared / 24 + radius_squared**2 / 1920
    cos_factor = jnp.where(small, series_cos, exact_cos)
    scale = jnp.where(small, series_scale, exact_scale)
    eye = jnp.eye(2, dtype=dtype)
    return cos_factor[..., None, None] * eye + scale[..., None, None] * algebra


def jax_su2_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return a BCC JAX link field from SU(2) link-algebra coordinates."""

    _validate_su2_link_algebra(theta)
    return jax_su2_link_from_algebra(theta)


def jax_su2_site_field_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return site-local SU(2) gauge matrices from algebra coordinates."""

    _validate_su2_site_algebra(site_theta)
    return jax_su2_link_from_algebra(site_theta)


def jax_su3_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian SU(3) matrices from algebra coordinates."""

    if theta.shape[-1:] != (8,):
        raise ValueError("SU(3) algebra coordinates must have trailing dimension 8")

    dtype = jnp.result_type(theta, 1j)
    coordinates = theta.astype(jnp.real(jnp.asarray(0, dtype=dtype)).dtype)
    generators = jax_su3_generators(dtype=dtype)
    return jnp.einsum("...a,aij->...ij", coordinates, generators)


def jax_su3_link_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return compact SU(3) matrices from Lie-algebra coordinates."""

    return _jax_matrix_expm_batch(jax_su3_algebra_matrix(theta))


def jax_su3_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return a BCC JAX link field from SU(3) link-algebra coordinates."""

    _validate_su3_link_algebra(theta)
    return jax_su3_link_from_algebra(theta)


def jax_su3_site_field_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return site-local SU(3) gauge matrices from algebra coordinates."""

    _validate_su3_site_algebra(site_theta)
    return jax_su3_link_from_algebra(site_theta)


def jax_transform_link_field(links: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply finite site-local gauge transforms to BCC pull-convention links.

    The convention matches the exact backend:
    ``U[x,h] -> G[x] U[x,h] G[x+h]^dagger``.
    """

    return _generic_jax_transform_link_field(links, site_gauge, jax_bcc_displacements())


def jax_su2_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge SU(2) BCC links generated by site-local gauges."""

    site_gauge = jax_su2_site_field_from_algebra(site_theta)
    identity_links = jax_identity_link_field(site_gauge.shape[:3], 2, dtype=site_gauge.dtype)
    return jax_transform_link_field(identity_links, site_gauge)


def jax_su3_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge SU(3) BCC links generated by site-local gauges."""

    site_gauge = jax_su3_site_field_from_algebra(site_theta)
    identity_links = jax_identity_link_field(site_gauge.shape[:3], 3, dtype=site_gauge.dtype)
    return jax_transform_link_field(identity_links, site_gauge)


def jax_su2_wilson_action_density(
    theta: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return Wilson action density for SU(2)-parameterized BCC links."""

    return jax_average_wilson_action_density(
        jax_su2_link_field_from_algebra(theta),
        shapes,
    )


def jax_su2_wilson_action_gradient(
    theta: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return the Lie-coordinate gradient of the SU(2) Wilson action.

    This is the gradient with respect to the unconstrained coordinates
    ``theta_a`` used in ``exp(theta_a T_a)``.  It is not yet a
    left-trivialized HMC force.
    """

    return jax.grad(lambda coordinates: jax_su2_wilson_action_density(coordinates, shapes))(theta)


def jax_su3_wilson_action_density(
    theta: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return Wilson action density for SU(3)-parameterized BCC links."""

    return jax_average_wilson_action_density(
        jax_su3_link_field_from_algebra(theta),
        shapes,
    )


def jax_su2_left_force(
    links: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return the left-trivialized Wilson force.

    For each link ``U[x,h]`` and generator ``T_a`` this computes

    ``d/deps S(..., exp(eps T_a) U[x,h], ...)|_{eps=0}``

    by differentiating the action with respect to left-perturbation
    coordinates.  The returned array has shape ``(nx, ny, nz, 8, 3)``.
    ``epsilon`` is retained for Session 31 finite-difference callers that
    check the same convention; the force itself uses JAX autodiff.
    """

    _validate_su2_links(links)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    zero_perturbation = jnp.zeros((*links.shape[:4], 3), dtype=real_dtype)

    def action_from_left_perturbation(perturbation: jnp.ndarray) -> jnp.ndarray:
        return jax_average_wilson_action_density(
            _jax_su2_apply_left_coordinates(links, perturbation),
            shapes,
        )

    return jax.grad(action_from_left_perturbation)(zero_perturbation)


def jax_su2_left_force_from_algebra(
    theta: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return left-trivialized SU(2) force for coordinate-built links."""

    return jax_su2_left_force(
        jax_su2_link_field_from_algebra(theta),
        epsilon=epsilon,
        shapes=shapes,
    )


def jax_su3_left_force(
    links: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return the left-trivialized Wilson force for SU(3) links."""

    _validate_su3_links(links)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    zero_perturbation = jnp.zeros((*links.shape[:4], 8), dtype=real_dtype)

    def action_from_left_perturbation(perturbation: jnp.ndarray) -> jnp.ndarray:
        return jax_average_wilson_action_density(
            _jax_su3_apply_left_coordinates(links, perturbation),
            shapes,
        )

    return jax.grad(action_from_left_perturbation)(zero_perturbation)


def jax_su3_left_force_from_algebra(
    theta: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return left-trivialized SU(3) force for coordinate-built links."""

    return jax_su3_left_force(
        jax_su3_link_field_from_algebra(theta),
        epsilon=epsilon,
        shapes=shapes,
    )


def jax_su2_apply_left_update(
    links: jnp.ndarray,
    force: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply a compact left gradient-descent update to SU(2) links.

    The update convention is
    ``U[x,h] -> exp(-step_size * force[x,h,a] T_a) U[x,h]``.  Since the
    update is a group exponential multiplied into the link, compactness is
    preserved up to numerical precision.
    """

    _validate_su2_links(links)
    if force.shape != (*links.shape[:4], 3):
        raise ValueError("force must have shape (nx, ny, nz, 8, 3)")

    return _jax_su2_apply_left_coordinates(links, -jnp.asarray(step_size, dtype=force.dtype) * force)


def jax_su2_action_descent_step(
    links: jnp.ndarray,
    *,
    step_size: float = 0.05,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(updated_links, force)`` for one compact Wilson descent step."""

    force = jax_su2_left_force(links, epsilon=epsilon, shapes=shapes)
    return jax_su2_apply_left_update(links, force, step_size=step_size), force


def jax_su3_action_descent_step(
    links: jnp.ndarray,
    *,
    step_size: float = 0.05,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(updated_links, force)`` for one compact SU(3) Wilson descent step."""

    force = jax_su3_left_force(links, epsilon=epsilon, shapes=shapes)
    return jax_su3_apply_left_update(links, force, step_size=step_size), force


def jax_su3_apply_left_update(
    links: jnp.ndarray,
    force: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply a compact left gradient-descent update to SU(3) links."""

    _validate_su3_links(links)
    if force.shape != (*links.shape[:4], 8):
        raise ValueError("force must have shape (nx, ny, nz, 8, 8)")

    return _jax_su3_apply_left_coordinates(links, -jnp.asarray(step_size, dtype=force.dtype) * force)


def _jax_su2_apply_left_coordinates(links: jnp.ndarray, coordinates: jnp.ndarray) -> jnp.ndarray:
    """Apply ``exp(coordinates_a T_a)`` to each link from the left."""

    _validate_su2_links(links)
    if coordinates.shape != (*links.shape[:4], 3):
        raise ValueError("coordinates must have shape (nx, ny, nz, 8, 3)")

    updates = jax_su2_link_from_algebra(coordinates)
    return jnp.einsum("...ij,...jk->...ik", updates, links)


def _jax_su3_apply_left_coordinates(links: jnp.ndarray, coordinates: jnp.ndarray) -> jnp.ndarray:
    """Apply ``exp(coordinates_a T_a)`` to each SU(3) link from the left."""

    _validate_su3_links(links)
    if coordinates.shape != (*links.shape[:4], 8):
        raise ValueError("coordinates must have shape (nx, ny, nz, 8, 8)")

    updates = jax_su3_link_from_algebra(coordinates)
    return jnp.einsum("...ij,...jk->...ik", updates, links)
