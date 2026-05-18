"""JAX gauge-force helpers for compact SO(2) BCC link fields.

Session 29 intentionally uses the smallest compact gauge testbed: every BCC
link is an SO(2) rotation parameterized by one real angle.  This keeps the
Wilson action differentiable through ordinary JAX arrays while preserving the
important lattice-gauge controls:

* zero field has zero action and zero gradient;
* pure gauges have zero action and zero gradient;
* non-flat fields have non-zero curvature and a non-zero action gradient.

This is a force audit, not a dynamical gauge update rule.  Nonabelian SU(N)
force projection and leapfrog/heatbath dynamics remain future work.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_step import jax_bcc_displacements
from clifford_3plus2_d5.spacetime_qca.jax_wilson import jax_average_wilson_action_density
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape


def _validate_so2_angles(theta: jnp.ndarray) -> None:
    if theta.ndim != 4 or theta.shape[-1] != 8:
        raise ValueError("SO(2) link angles must have shape (nx, ny, nz, 8)")


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
        source_angles = jnp.roll(
            site_angles,
            shift=tuple(-component for component in displacement),
            axis=(0, 1, 2),
        )
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
