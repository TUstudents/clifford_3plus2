"""JAX SU(2) compact gauge-field dynamics prototypes.

Session 32 adds the first reversible gauge-field update on top of the Session
31 left-trivialized force.  This is a Hamiltonian/leapfrog simulation control
for compact SU(2) links, not yet a full QCA-native Yang-Mills rule.
"""

from __future__ import annotations

import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import (
    jax_su2_generators,
    jax_su2_left_force,
    jax_su2_link_from_algebra,
    jax_su2_project_antihermitian_to_algebra,
)
from clifford_3plus2_d5.spacetime_qca.jax_wilson import jax_average_wilson_action_density
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape


def _validate_su2_links(links: jnp.ndarray) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 2, 2):
        raise ValueError("SU(2) BCC links must have shape (nx, ny, nz, 8, 2, 2)")


def _validate_su2_momenta(momenta: jnp.ndarray) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, 3):
        raise ValueError("SU(2) momenta must have shape (nx, ny, nz, 8, 3)")


def _validate_site_gauge(site_gauge: jnp.ndarray) -> None:
    if site_gauge.ndim != 5 or site_gauge.shape[-2:] != (2, 2):
        raise ValueError("site gauge must have shape (nx, ny, nz, 2, 2)")


def jax_su2_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian SU(2) matrices from algebra coordinates."""

    if theta.shape[-1:] != (3,):
        raise ValueError("SU(2) algebra coordinates must have trailing dimension 3")

    generators = jax_su2_generators(dtype=jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def jax_su2_transform_momentum_field(momenta: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to SU(2) link momenta.

    Link fields use the pull convention ``U[x,h] -> G[x] U[x,h] G[x+h]^dag``.
    A left-trivialized momentum lives at the target site and therefore
    transforms as ``P[x,h] -> G[x] P[x,h] G[x]^dag``.
    """

    _validate_su2_momenta(momenta)
    _validate_site_gauge(site_gauge)
    if momenta.shape[:3] != site_gauge.shape[:3]:
        raise ValueError("momenta and site_gauge must share lattice shape")

    algebra = jax_su2_algebra_matrix(momenta)
    target_gauge = site_gauge[..., None, :, :]
    transformed = target_gauge @ algebra @ jnp.swapaxes(jnp.conj(target_gauge), -1, -2)
    return jax_su2_project_antihermitian_to_algebra(transformed)


def jax_su2_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return ``0.5 * mean_link sum_a P_a^2`` for SU(2) momenta."""

    _validate_su2_momenta(momenta)
    return 0.5 * jnp.mean(jnp.sum(momenta * momenta, axis=-1))


def jax_su2_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for SU(2) gauge fields."""

    _validate_su2_links(links)
    _validate_su2_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    return jax_su2_momentum_kinetic_energy_density(momenta) + beta * jax_average_wilson_action_density(
        links,
        shapes,
    )


def jax_su2_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply ``U[x,h] -> exp(step_size * P[x,h]) U[x,h]``."""

    _validate_su2_links(links)
    _validate_su2_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    updates = jax_su2_link_from_algebra(jnp.asarray(step_size, dtype=momenta.dtype) * momenta)
    return jnp.einsum("...ij,...jk->...ik", updates, links)


def jax_su2_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible leapfrog step for compact SU(2) gauge fields."""

    _validate_su2_links(links)
    _validate_su2_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    dt = jnp.asarray(step_size, dtype=momenta.dtype)
    force = beta * jax_su2_left_force(links, shapes=shapes)
    half_momenta = momenta - 0.5 * dt * force
    updated_links = jax_su2_apply_momentum_update(links, half_momenta, step_size=dt)
    updated_force = beta * jax_su2_left_force(updated_links, shapes=shapes)
    updated_momenta = half_momenta - 0.5 * dt * updated_force
    return updated_links, updated_momenta
