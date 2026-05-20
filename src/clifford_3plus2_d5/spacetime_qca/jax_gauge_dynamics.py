"""JAX compact gauge-field dynamics prototypes.

Session 32 adds the first reversible gauge-field update on top of the Session
31 left-trivialized force.  This is a Hamiltonian/leapfrog simulation control
for compact SU(2) links, not yet a full QCA-native Yang-Mills rule.  Session
33 mirrors that stack for compact SU(3), the first color-gauge dynamics
testbed.
"""

from __future__ import annotations

import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import (
    CompactLieForceMethod,
    jax_compact_lie_algebra_matrix,
    jax_compact_lie_left_force,
    jax_compact_lie_link_from_algebra,
    jax_compact_lie_project_to_coordinates,
    jax_su2_generators,
    jax_su2_left_force,
    jax_su2_link_from_algebra,
    jax_su2_project_antihermitian_to_algebra,
    jax_su3_generators,
    jax_su3_left_force,
    jax_su3_link_from_algebra,
    jax_su3_project_antihermitian_to_algebra,
)
from clifford_3plus2_d5.spacetime_qca.jax_wilson import jax_average_wilson_action_density
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape


def _validate_su2_links(links: jnp.ndarray) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 2, 2):
        raise ValueError("SU(2) BCC links must have shape (nx, ny, nz, 8, 2, 2)")


def _validate_su2_momenta(momenta: jnp.ndarray) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, 3):
        raise ValueError("SU(2) momenta must have shape (nx, ny, nz, 8, 3)")


def _validate_su2_site_gauge(site_gauge: jnp.ndarray) -> None:
    if site_gauge.ndim != 5 or site_gauge.shape[-2:] != (2, 2):
        raise ValueError("SU(2) site gauge must have shape (nx, ny, nz, 2, 2)")


def _validate_su3_links(links: jnp.ndarray) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, 3, 3):
        raise ValueError("SU(3) BCC links must have shape (nx, ny, nz, 8, 3, 3)")


def _validate_su3_momenta(momenta: jnp.ndarray) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, 8):
        raise ValueError("SU(3) momenta must have shape (nx, ny, nz, 8, 8)")


def _validate_su3_site_gauge(site_gauge: jnp.ndarray) -> None:
    if site_gauge.ndim != 5 or site_gauge.shape[-2:] != (3, 3):
        raise ValueError("SU(3) site gauge must have shape (nx, ny, nz, 3, 3)")


def _validate_compact_lie_generators(generators: jnp.ndarray) -> tuple[int, int]:
    if generators.ndim != 3:
        raise ValueError("generators must have shape (generator_count, matrix_dim, matrix_dim)")
    if generators.shape[-1] != generators.shape[-2]:
        raise ValueError("generators must be square matrices")
    return int(generators.shape[0]), int(generators.shape[-1])


def _validate_compact_lie_links(links: jnp.ndarray, generators: jnp.ndarray) -> tuple[int, int]:
    generator_count, matrix_dim = _validate_compact_lie_generators(generators)
    if links.ndim != 6 or links.shape[3:] != (8, matrix_dim, matrix_dim):
        raise ValueError(
            "links must have shape (nx, ny, nz, 8, matrix_dim, matrix_dim) "
            "matching generators",
        )
    return generator_count, matrix_dim


def _validate_compact_lie_momenta(momenta: jnp.ndarray, generators: jnp.ndarray) -> tuple[int, int]:
    generator_count, matrix_dim = _validate_compact_lie_generators(generators)
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, generator_count):
        raise ValueError("momenta must have shape (nx, ny, nz, 8, generator_count)")
    return generator_count, matrix_dim


def _validate_compact_lie_site_gauge(site_gauge: jnp.ndarray, generators: jnp.ndarray) -> None:
    _, matrix_dim = _validate_compact_lie_generators(generators)
    if site_gauge.ndim != 5 or site_gauge.shape[-2:] != (matrix_dim, matrix_dim):
        raise ValueError("site_gauge must have shape (nx, ny, nz, matrix_dim, matrix_dim)")


def jax_su2_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian SU(2) matrices from algebra coordinates."""

    if theta.shape[-1:] != (3,):
        raise ValueError("SU(2) algebra coordinates must have trailing dimension 3")

    generators = jax_su2_generators(dtype=jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def jax_su3_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian SU(3) matrices from algebra coordinates."""

    if theta.shape[-1:] != (8,):
        raise ValueError("SU(3) algebra coordinates must have trailing dimension 8")

    generators = jax_su3_generators(dtype=jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def jax_compact_lie_transform_momentum_field(
    momenta: jnp.ndarray,
    site_gauge: jnp.ndarray,
    generators: jnp.ndarray,
) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to basis-coordinate momenta."""

    _validate_compact_lie_momenta(momenta, generators)
    _validate_compact_lie_site_gauge(site_gauge, generators)
    if momenta.shape[:3] != site_gauge.shape[:3]:
        raise ValueError("momenta and site_gauge must share lattice shape")

    algebra = jax_compact_lie_algebra_matrix(momenta, generators)
    target_gauge = site_gauge[..., None, :, :]
    transformed = target_gauge @ algebra @ jnp.swapaxes(jnp.conj(target_gauge), -1, -2)
    return jax_compact_lie_project_to_coordinates(transformed, generators)


def jax_compact_lie_momentum_kinetic_energy_density(
    momenta: jnp.ndarray,
    generators: jnp.ndarray,
) -> jnp.ndarray:
    """Return Gram-metric kinetic energy density for coordinate momenta."""

    _validate_compact_lie_momenta(momenta, generators)
    basis = generators.astype(jnp.result_type(generators, 1j))
    basis_daggers = jnp.swapaxes(jnp.conj(basis), -1, -2)
    gram = jnp.real(jnp.einsum("aij,bji->ab", basis_daggers, basis))
    return 0.5 * jnp.mean(jnp.einsum("...a,ab,...b->...", momenta, gram, momenta))


def jax_compact_lie_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    generators: jnp.ndarray,
    *,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for compact Lie links."""

    _validate_compact_lie_links(links, generators)
    _validate_compact_lie_momenta(momenta, generators)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    return jax_compact_lie_momentum_kinetic_energy_density(
        momenta,
        generators,
    ) + beta * jax_average_wilson_action_density(links, shapes)


def jax_compact_lie_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    generators: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply ``U[x,h] -> exp(step_size * P[x,h]) U[x,h]``."""

    _validate_compact_lie_links(links, generators)
    _validate_compact_lie_momenta(momenta, generators)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    updates = jax_compact_lie_link_from_algebra(jnp.asarray(step_size, dtype=momenta.dtype) * momenta, generators)
    return jnp.einsum("...ij,...jk->...ik", updates, links)


def jax_compact_lie_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    generators: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: CompactLieForceMethod = "autodiff",
    force_epsilon: float = 1e-3,
    force_chunk_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible leapfrog step for compact basis-coordinate links."""

    _validate_compact_lie_links(links, generators)
    _validate_compact_lie_momenta(momenta, generators)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    dt = jnp.asarray(step_size, dtype=momenta.dtype)
    force = beta * jax_compact_lie_left_force(
        links,
        generators,
        epsilon=force_epsilon,
        shapes=shapes,
        method=force_method,
        chunk_size=force_chunk_size,
    )
    half_momenta = momenta - 0.5 * dt * force
    updated_links = jax_compact_lie_apply_momentum_update(
        links,
        half_momenta,
        generators,
        step_size=dt,
    )
    updated_force = beta * jax_compact_lie_left_force(
        updated_links,
        generators,
        epsilon=force_epsilon,
        shapes=shapes,
        method=force_method,
        chunk_size=force_chunk_size,
    )
    updated_momenta = half_momenta - 0.5 * dt * updated_force
    return updated_links, updated_momenta


def jax_su2_transform_momentum_field(momenta: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to SU(2) link momenta.

    Link fields use the pull convention ``U[x,h] -> G[x] U[x,h] G[x+h]^dag``.
    A left-trivialized momentum lives at the target site and therefore
    transforms as ``P[x,h] -> G[x] P[x,h] G[x]^dag``.
    """

    _validate_su2_momenta(momenta)
    _validate_su2_site_gauge(site_gauge)
    if momenta.shape[:3] != site_gauge.shape[:3]:
        raise ValueError("momenta and site_gauge must share lattice shape")

    algebra = jax_su2_algebra_matrix(momenta)
    target_gauge = site_gauge[..., None, :, :]
    transformed = target_gauge @ algebra @ jnp.swapaxes(jnp.conj(target_gauge), -1, -2)
    return jax_su2_project_antihermitian_to_algebra(transformed)


def jax_su3_transform_momentum_field(momenta: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to SU(3) link momenta."""

    _validate_su3_momenta(momenta)
    _validate_su3_site_gauge(site_gauge)
    if momenta.shape[:3] != site_gauge.shape[:3]:
        raise ValueError("momenta and site_gauge must share lattice shape")

    algebra = jax_su3_algebra_matrix(momenta)
    target_gauge = site_gauge[..., None, :, :]
    transformed = target_gauge @ algebra @ jnp.swapaxes(jnp.conj(target_gauge), -1, -2)
    return jax_su3_project_antihermitian_to_algebra(transformed)


def jax_su2_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return ``0.5 * mean_link sum_a P_a^2`` for SU(2) momenta."""

    _validate_su2_momenta(momenta)
    return 0.5 * jnp.mean(jnp.sum(momenta * momenta, axis=-1))


def jax_su3_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return ``0.5 * mean_link sum_a P_a^2`` for SU(3) momenta."""

    _validate_su3_momenta(momenta)
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


def jax_su3_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for SU(3) gauge fields."""

    _validate_su3_links(links)
    _validate_su3_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    return jax_su3_momentum_kinetic_energy_density(momenta) + beta * jax_average_wilson_action_density(
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


def jax_su3_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply ``U[x,h] -> exp(step_size * P[x,h]) U[x,h]`` for SU(3)."""

    _validate_su3_links(links)
    _validate_su3_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    updates = jax_su3_link_from_algebra(jnp.asarray(step_size, dtype=momenta.dtype) * momenta)
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


def jax_su3_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible leapfrog step for compact SU(3) gauge fields."""

    _validate_su3_links(links)
    _validate_su3_momenta(momenta)
    if momenta.shape[:4] != links.shape[:4]:
        raise ValueError("links and momenta must share shape (nx, ny, nz, 8)")

    dt = jnp.asarray(step_size, dtype=momenta.dtype)
    force = beta * jax_su3_left_force(links, shapes=shapes)
    half_momenta = momenta - 0.5 * dt * force
    updated_links = jax_su3_apply_momentum_update(links, half_momenta, step_size=dt)
    updated_force = beta * jax_su3_left_force(updated_links, shapes=shapes)
    updated_momenta = half_momenta - 0.5 * dt * updated_force
    return updated_links, updated_momenta
