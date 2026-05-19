"""JAX adapters for Pati-Salam and SM compact gauge dynamics.

Session 34 introduced the first chiral16 Pati-Salam adapter for the
``Spin(0,6) ~= SU(4)`` basis.  Session 35 generalizes the adapter to all
Pati-Salam and Standard Model sectors extracted in ``lepton``.

All sectors act on the same 32-real-dimensional chiral16 internal carrier.
This is not a fundamental SU(N) representation: the matrix size is always
``32 x 32`` while the algebra dimension depends on the sector.  The adapters
therefore route through the basis-based compact Lie helpers from Session 34.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.lepton.clifford_patisalam import (
    su2_l_generators_from_spin04,
    su2_r_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.patisalam_sm import (
    hypercharge_generator,
    sm_gauge_generators,
    su3_c_generators_from_su4,
)
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

PatiSalamGaugeSector: TypeAlias = Literal[
    "su4",
    "su2_l",
    "su2_r",
    "pati_salam",
    "su3_c",
    "u1_y",
    "sm",
]


def _sector_generators(sector: PatiSalamGaugeSector):
    if sector == "su4":
        return su4_generators_from_spin06()
    if sector == "su2_l":
        return su2_l_generators_from_spin04()
    if sector == "su2_r":
        return su2_r_generators_from_spin04()
    if sector == "pati_salam":
        return (
            *su4_generators_from_spin06(),
            *su2_l_generators_from_spin04(),
            *su2_r_generators_from_spin04(),
        )
    if sector == "su3_c":
        return su3_c_generators_from_su4()
    if sector == "u1_y":
        return (hypercharge_generator(),)
    if sector == "sm":
        return sm_gauge_generators()
    raise ValueError(f"unknown Pati-Salam gauge sector: {sector}")


@lru_cache(maxsize=16)
def _patisalam_generators_numpy(sector: PatiSalamGaugeSector, dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return np.stack(
        [sympy_matrix_to_numpy(generator, dtype=dtype) for generator in _sector_generators(sector)],
        axis=0,
    )


def jax_patisalam_generators_chiral16(
    sector: PatiSalamGaugeSector = "su4",
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """Return a chiral16 Pati-Salam/SM generator basis for ``sector``.

    Sector dimensions are ``su4=15``, ``su2_l=3``, ``su2_r=3``,
    ``pati_salam=21``, ``su3_c=8``, ``u1_y=1``, and ``sm=12``.
    """

    dtype = np.dtype(dtype)
    generators = _patisalam_generators_numpy(sector, dtype.name)
    return jnp.asarray(generators, dtype=dtype)


def jax_patisalam_su4_generators_chiral16(dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the chiral16 SU(4) basis as JAX matrices."""

    return jax_patisalam_generators_chiral16("su4", dtype)


def jax_patisalam_algebra_matrix(
    theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return chiral16 algebra matrices from sector coordinates."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(theta, 1j))
    return jax_compact_lie_algebra_matrix(theta, generators)


def jax_patisalam_su4_algebra_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Return chiral16 SU(4) algebra matrices from 15 coordinates."""

    return jax_patisalam_algebra_matrix(theta, sector="su4")


def jax_patisalam_project_to_coordinates(
    matrix: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Project chiral16 sector algebra matrices onto basis coordinates."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(matrix, 1j))
    return jax_compact_lie_project_to_coordinates(matrix, generators)


def jax_patisalam_su4_project_to_coordinates(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project chiral16 SU(4) algebra matrices onto the 15-generator basis."""

    return jax_patisalam_project_to_coordinates(matrix, sector="su4")


def jax_patisalam_link_from_algebra(
    theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return compact chiral16 representation links for a sector."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(theta, 1j))
    return jax_compact_lie_link_from_algebra(theta, generators)


def jax_patisalam_su4_link_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return compact chiral16 SU(4) representation links."""

    return jax_patisalam_link_from_algebra(theta, sector="su4")


def jax_patisalam_link_field_from_algebra(
    theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return BCC chiral16 links from sector coordinates."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(theta, 1j))
    return jax_compact_lie_link_field_from_algebra(theta, generators)


def jax_patisalam_su4_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return BCC chiral16 SU(4) links from coordinates."""

    return jax_patisalam_link_field_from_algebra(theta, sector="su4")


def jax_patisalam_site_field_from_algebra(
    site_theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return site-local chiral16 gauges from sector coordinates."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(site_theta, 1j))
    return jax_compact_lie_site_field_from_algebra(site_theta, generators)


def jax_patisalam_su4_site_field_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return site-local chiral16 SU(4) gauges from coordinates."""

    return jax_patisalam_site_field_from_algebra(site_theta, sector="su4")


def jax_patisalam_pure_gauge_links_from_site_algebra(
    site_theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return pure-gauge BCC links in a chiral16 sector."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(site_theta, 1j))
    return jax_compact_lie_pure_gauge_links_from_site_algebra(site_theta, generators)


def jax_patisalam_su4_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge BCC links in the chiral16 SU(4) representation."""

    return jax_patisalam_pure_gauge_links_from_site_algebra(site_theta, sector="su4")


def jax_patisalam_left_force(
    links: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> jnp.ndarray:
    """Return the chiral16 sector left force."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
    return jax_compact_lie_left_force(
        links,
        generators,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_su4_left_force(
    links: jnp.ndarray,
    *,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> jnp.ndarray:
    """Return the chiral16 SU(4) left force."""

    return jax_patisalam_left_force(
        links,
        sector="su4",
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_left_force_from_algebra(
    theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> jnp.ndarray:
    """Return the chiral16 sector left force for coordinate-built links."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(theta, 1j))
    return jax_compact_lie_left_force_from_algebra(
        theta,
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

    return jax_patisalam_left_force_from_algebra(
        theta,
        sector="su4",
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_apply_left_update(
    links: jnp.ndarray,
    force: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float,
) -> jnp.ndarray:
    """Apply compact left descent updates to chiral16 sector links."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
    return jax_compact_lie_apply_left_update(links, force, generators, step_size=step_size)


def jax_patisalam_su4_apply_left_update(
    links: jnp.ndarray,
    force: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply compact left descent updates to chiral16 SU(4) links."""

    return jax_patisalam_apply_left_update(links, force, sector="su4", step_size=step_size)


def jax_patisalam_action_descent_step(
    links: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float = 0.05,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(updated_links, force)`` for one sector Wilson descent step."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
    return jax_compact_lie_action_descent_step(
        links,
        generators,
        step_size=step_size,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_su4_action_descent_step(
    links: jnp.ndarray,
    *,
    step_size: float = 0.05,
    epsilon: float = 1e-3,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    method: Literal["autodiff", "finite_difference"] = "finite_difference",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(updated_links, force)`` for one SU(4) Wilson descent step."""

    return jax_patisalam_action_descent_step(
        links,
        sector="su4",
        step_size=step_size,
        epsilon=epsilon,
        shapes=shapes,
        method=method,
    )


def jax_patisalam_transform_momentum_field(
    momenta: jnp.ndarray,
    site_gauge: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to sector momenta."""

    generators = jax_patisalam_generators_chiral16(sector, site_gauge.dtype)
    return jax_compact_lie_transform_momentum_field(momenta, site_gauge, generators)


def jax_patisalam_su4_transform_momentum_field(
    momenta: jnp.ndarray,
    site_gauge: jnp.ndarray,
) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to SU(4) momenta."""

    return jax_patisalam_transform_momentum_field(momenta, site_gauge, sector="su4")


def jax_patisalam_momentum_kinetic_energy_density(
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return basis-coordinate kinetic energy density for a sector."""

    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(momenta, 1j))
    return jax_compact_lie_momentum_kinetic_energy_density(momenta, generators)


def jax_patisalam_su4_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return basis-coordinate kinetic energy density for chiral16 SU(4)."""

    return jax_patisalam_momentum_kinetic_energy_density(momenta, sector="su4")


def jax_patisalam_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for a chiral16 sector."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
    return jax_compact_lie_gauge_hamiltonian_density(
        links,
        momenta,
        generators,
        beta=beta,
        shapes=shapes,
    )


def jax_patisalam_su4_gauge_hamiltonian_density(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return kinetic plus Wilson-action density for chiral16 SU(4)."""

    return jax_patisalam_gauge_hamiltonian_density(
        links,
        momenta,
        sector="su4",
        beta=beta,
        shapes=shapes,
    )


def jax_patisalam_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float,
) -> jnp.ndarray:
    """Apply compact momentum updates to chiral16 sector links."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
    return jax_compact_lie_apply_momentum_update(
        links,
        momenta,
        generators,
        step_size=step_size,
    )


def jax_patisalam_su4_apply_momentum_update(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
) -> jnp.ndarray:
    """Apply compact momentum updates to chiral16 SU(4) links."""

    return jax_patisalam_apply_momentum_update(
        links,
        momenta,
        sector="su4",
        step_size=step_size,
    )


def jax_patisalam_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: Literal["autodiff", "finite_difference"] = "finite_difference",
    force_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible leapfrog step for a chiral16 gauge sector."""

    generators = jax_patisalam_generators_chiral16(sector, links.dtype)
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

    return jax_patisalam_leapfrog_step(
        links,
        momenta,
        sector="su4",
        step_size=step_size,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
    )
