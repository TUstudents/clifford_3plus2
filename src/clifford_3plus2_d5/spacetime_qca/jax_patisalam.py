"""JAX adapters for Pati-Salam compact gauge dynamics.

Session 34 keeps the physics-specific Pati-Salam representation in
``lepton`` and exposes only thin numerical adapters here.  The first target is
the chiral16 Spin(0,6) ~= SU(4) basis: 15 exact real-skew generators acting on
the 32-real-dimensional internal carrier.  This is not fundamental SU(4);
generic compact Lie helpers must be basis-based because the link matrices are
``32 x 32`` while the algebra has dimension 15.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.lepton.clifford_patisalam import su4_generators_from_spin06
from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.jax_gauge_dynamics import (
    jax_compact_lie_apply_momentum_update,
    jax_compact_lie_gauge_hamiltonian_density,
    jax_compact_lie_leapfrog_step,
    jax_compact_lie_momentum_kinetic_energy_density,
    jax_compact_lie_transform_momentum_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import (
    jax_compact_lie_action_descent_step,
    jax_compact_lie_algebra_matrix,
    jax_compact_lie_apply_left_update,
    jax_compact_lie_left_force,
    jax_compact_lie_left_force_from_algebra,
    jax_compact_lie_link_field_from_algebra,
    jax_compact_lie_link_from_algebra,
    jax_compact_lie_project_to_coordinates,
    jax_compact_lie_pure_gauge_links_from_site_algebra,
    jax_compact_lie_site_field_from_algebra,
)
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape


@lru_cache(maxsize=2)
def _patisalam_su4_generators_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return np.stack(
        [sympy_matrix_to_numpy(generator, dtype=dtype) for generator in su4_generators_from_spin06()],
        axis=0,
    )


def jax_patisalam_su4_generators_chiral16(dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the chiral16 SU(4) basis as JAX matrices.

    The basis comes from ``lepton.clifford_patisalam.su4_generators_from_spin06``.
    It has shape ``(15, 32, 32)`` and consists of anti-Hermitian real-skew
    matrices.  ``complex64`` is the default because Session 34 deliberately
    avoids the memory cost of reverse-mode differentiation through many
    ``32 x 32`` matrix exponentials.
    """

    dtype = np.dtype(dtype)
    generators = _patisalam_su4_generators_numpy(dtype.name)
    return jnp.asarray(generators, dtype=dtype)


def jax_patisalam_su4_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return chiral16 SU(4) algebra matrices from 15 coordinates."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(theta, 1j))
    return jax_compact_lie_algebra_matrix(theta, generators)


def jax_patisalam_su4_project_to_coordinates(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project chiral16 SU(4) algebra matrices onto the 15-generator basis."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(matrix, 1j))
    return jax_compact_lie_project_to_coordinates(matrix, generators)


def jax_patisalam_su4_link_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return compact chiral16 SU(4) representation links."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(theta, 1j))
    return jax_compact_lie_link_from_algebra(theta, generators)


def jax_patisalam_su4_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return BCC chiral16 SU(4) links from coordinates."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(theta, 1j))
    return jax_compact_lie_link_field_from_algebra(theta, generators)


def jax_patisalam_su4_site_field_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return site-local chiral16 SU(4) gauges from coordinates."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(site_theta, 1j))
    return jax_compact_lie_site_field_from_algebra(site_theta, generators)


def jax_patisalam_su4_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge BCC links in the chiral16 SU(4) representation."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(site_theta, 1j))
    return jax_compact_lie_pure_gauge_links_from_site_algebra(site_theta, generators)


def jax_patisalam_su4_left_force(
    links: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> jnp.ndarray:
    """Return the chiral16 SU(4) left force.

    The default is coordinate-wise finite difference.  It is slower than
    autodiff, but it keeps Session 34 away from the memory failure mode caused
    by reverse-mode differentiation through many ``32 x 32`` exponentials.
    """

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_left_force(
        links,
        generators,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_su4_left_force_from_algebra(
    theta: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> jnp.ndarray:
    """Return the chiral16 SU(4) left force for coordinate-built links."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(theta, 1j))
    return jax_compact_lie_left_force_from_algebra(
        theta,
        generators,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_su4_apply_left_update(
    links: jnp.ndarray,
    force: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply compact left descent updates to chiral16 SU(4) links."""

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_apply_left_update(links, force, generators, step_size=step_size)


def jax_patisalam_su4_action_descent_step(
    links: jnp.ndarray,
    *,
    step_size: float = 0.05,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(updated_links, force)`` for one SU(4) Wilson descent step."""

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_action_descent_step(
        links,
        generators,
        step_size=step_size,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_su4_transform_momentum_field(
    momenta: jnp.ndarray,
    site_gauge: jnp.ndarray,
) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to SU(4) momenta."""

    generators = jax_patisalam_su4_generators_chiral16(site_gauge.dtype)
    return jax_compact_lie_transform_momentum_field(momenta, site_gauge, generators)


def jax_patisalam_su4_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return basis-coordinate kinetic energy density for chiral16 SU(4)."""

    generators = jax_patisalam_su4_generators_chiral16(jnp.result_type(momenta, 1j))
    return jax_compact_lie_momentum_kinetic_energy_density(momenta, generators)


def jax_patisalam_su4_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for chiral16 SU(4)."""

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_gauge_hamiltonian_density(
        links,
        momenta,
        generators,
        beta=beta,
        shapes=shapes,
    )


def jax_patisalam_su4_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply compact momentum updates to chiral16 SU(4) links."""

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_apply_momentum_update(
        links,
        momenta,
        generators,
        step_size=step_size,
    )


def jax_patisalam_su4_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: Literal["autodiff", "finite_difference"] = "finite_difference",
    force_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible leapfrog step for chiral16 SU(4) gauge fields."""

    generators = jax_patisalam_su4_generators_chiral16(links.dtype)
    return jax_compact_lie_leapfrog_step(
        links,
        momenta,
        generators,
        step_size=step_size,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
    )
