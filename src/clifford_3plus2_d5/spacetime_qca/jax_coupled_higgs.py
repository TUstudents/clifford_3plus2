"""JAX coupled fermion/gauge/Higgs prototype.

Session 40 is the first ``spacetime_qca`` layer with all field types present:

* BCC Dirac fermions on the chiral16 internal carrier;
* compact Pati-Salam/SM gauge links and momenta;
* a site-local electroweak Higgs doublet with a conjugate momentum;
* a first-order site-local Yukawa kick sourced by ``Phi(x)``.

The implementation is intentionally conservative.  Higgs links are explicit
``2 x 2`` electroweak links, not inferred from the ``32 x 32`` chiral16 gauge
links.  The Yukawa kick is first-order explicit, not an exact local unitary
exponential.  Higgs current backreaction into gauge momenta is future work.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.state import norm_drift, state_norm_squared, sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.jax_fermion_gauge import PATISALAM_INTERNAL_DIM
from clifford_3plus2_d5.spacetime_qca.jax_gauss import (
    jax_patisalam_fermion_gauge_step_with_backreaction,
    jax_patisalam_gauss_residual,
)
from clifford_3plus2_d5.spacetime_qca.jax_higgs import (
    jax_higgs_energy_density,
    jax_higgs_kinetic_energy_density,
    jax_higgs_link_field_from_algebra,
    jax_higgs_potential_density,
    jax_higgs_site_gauge_from_algebra,
    jax_higgs_yukawa_internal_control_field,
    jax_transform_higgs_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import (
    PatiSalamGaugeSector,
    jax_patisalam_gauge_hamiltonian_density,
)
from clifford_3plus2_d5.spacetime_qca.mass import beta_matrix
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape

HiggsCoupledSector = Literal["u1_y", "su2_l", "sm"]


class PatiSalamFermionGaugeHiggsDiagnostics(TypedDict):
    """Diagnostics for the small coupled prototype."""

    fermion_norm: jnp.ndarray
    gauge_hamiltonian_density: jnp.ndarray
    higgs_norm: jnp.ndarray
    higgs_momentum_energy_density: jnp.ndarray
    higgs_kinetic_energy_density: jnp.ndarray
    higgs_potential_density: jnp.ndarray
    higgs_energy_density: jnp.ndarray
    gauss_residual_norm: jnp.ndarray
    yukawa_norm_drift: jnp.ndarray


def _validate_patisalam_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 5 or state.shape[-2:] != (4, PATISALAM_INTERNAL_DIM):
        raise ValueError("state must have shape (nx, ny, nz, 4, 32)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_patisalam_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int]) -> None:
    if links.ndim != 6 or links.shape[:3] != lattice_shape or links.shape[3:] != (
        8,
        PATISALAM_INTERNAL_DIM,
        PATISALAM_INTERNAL_DIM,
    ):
        raise ValueError("links must have shape (nx, ny, nz, 8, 32, 32)")


def _sector_dimension(sector: PatiSalamGaugeSector) -> int:
    if sector == "u1_y":
        return 1
    if sector == "su2_l" or sector == "su2_r":
        return 3
    if sector == "su3_c":
        return 8
    if sector == "su4":
        return 15
    if sector == "sm":
        return 12
    if sector == "pati_salam":
        return 21
    raise ValueError(f"unknown Pati-Salam gauge sector: {sector}")


def _validate_higgs_coupled_sector(sector: PatiSalamGaugeSector) -> None:
    if sector not in ("u1_y", "su2_l", "sm"):
        raise ValueError(f"sector {sector!r} is not supported by the v1 Higgs-coupled wrapper")


def _validate_patisalam_momenta(
    momenta: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    sector: PatiSalamGaugeSector,
) -> None:
    sector_dim = _sector_dimension(sector)
    if momenta.ndim != 5 or momenta.shape != (*lattice_shape, 8, sector_dim):
        raise ValueError(f"momenta must have shape (nx, ny, nz, 8, {sector_dim})")


def _validate_higgs_field(phi: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if phi.ndim != 4 or phi.shape[-1] != 2:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and phi.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the lattice shape")


def _validate_higgs_links(higgs_links: jnp.ndarray, lattice_shape: tuple[int, int, int]) -> None:
    if higgs_links.ndim != 6 or higgs_links.shape != (*lattice_shape, 8, 2, 2):
        raise ValueError("Higgs links must have shape (nx, ny, nz, 8, 2, 2)")


def _complex_field_to_real_coordinates(field: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack((jnp.real(field), jnp.imag(field)), axis=-1)


def _real_coordinates_to_complex_field(coordinates: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    return (coordinates[..., 0] + 1j * coordinates[..., 1]).astype(dtype)


@lru_cache(maxsize=8)
def _beta_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return sympy_matrix_to_numpy(beta_matrix(), dtype=dtype)


def _dirac_beta(dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(_beta_numpy(np.dtype(dtype).name), dtype=dtype)


def jax_transform_higgs_momentum(momentum: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply ``Pi[x] -> G[x] Pi[x]`` to the Higgs conjugate momentum."""

    return jax_transform_higgs_field(momentum, site_gauge)


def jax_higgs_momentum_energy_density(momentum: jnp.ndarray) -> jnp.ndarray:
    """Return sitewise ``0.5 * ||Pi||^2`` for Higgs conjugate momentum."""

    _validate_higgs_field(momentum)
    return 0.5 * jnp.real(jnp.sum(jnp.conj(momentum) * momentum, axis=-1))


def jax_higgs_total_energy(
    phi: jnp.ndarray,
    momentum: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
) -> jnp.ndarray:
    """Return total Higgs Hamiltonian ``sum_x (0.5|Pi|^2 + |D Phi|^2 + V)``."""

    lattice_shape = (int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2]))
    _validate_higgs_field(phi)
    _validate_higgs_field(momentum, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)
    return jnp.sum(
        jax_higgs_momentum_energy_density(momentum)
        + jax_higgs_energy_density(
            phi,
            higgs_links,
            vev_squared=vev_squared,
            quartic=quartic,
        ),
    )


def jax_higgs_force(
    phi: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
) -> jnp.ndarray:
    """Return canonical force ``-d E_H / d Phi`` in complex field layout."""

    lattice_shape = (int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2]))
    _validate_higgs_field(phi)
    _validate_higgs_links(higgs_links, lattice_shape)
    phi_dtype = phi.dtype
    real_coordinates = _complex_field_to_real_coordinates(phi)

    def potential_energy(coordinates: jnp.ndarray) -> jnp.ndarray:
        candidate = _real_coordinates_to_complex_field(coordinates, phi_dtype)
        return jnp.sum(
            jax_higgs_energy_density(
                candidate,
                higgs_links,
                vev_squared=vev_squared,
                quartic=quartic,
            ),
        )

    gradient = jax.grad(potential_energy)(real_coordinates)
    return _real_coordinates_to_complex_field(-gradient, phi_dtype)


def jax_higgs_leapfrog_step(
    phi: jnp.ndarray,
    momentum: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    step_size: float,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one leapfrog update for ``(Phi, Pi)`` with fixed Higgs links."""

    lattice_shape = (int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2]))
    _validate_higgs_field(phi)
    _validate_higgs_field(momentum, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)

    dt = jnp.asarray(step_size, dtype=jnp.real(phi).dtype)
    force = jax_higgs_force(phi, higgs_links, vev_squared=vev_squared, quartic=quartic)
    half_momentum = momentum + 0.5 * dt * force
    updated_phi = phi + dt * half_momentum
    updated_force = jax_higgs_force(updated_phi, higgs_links, vev_squared=vev_squared, quartic=quartic)
    updated_momentum = half_momentum + 0.5 * dt * updated_force
    return updated_phi, updated_momentum


def jax_higgs_coordinates_from_patisalam_sector(
    coordinates: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    """Map selected Pati-Salam/SM sector coordinates to Higgs coordinates.

    The returned trailing coordinate order is ``(su2_x, su2_y, su2_z, u1_y)``.
    Color-only and Pati-Salam-only sectors are rejected for this v1 coupled
    Higgs wrapper because the Higgs layer acts only through electroweak links.
    """

    if sector == "u1_y":
        if coordinates.shape[-1:] != (1,):
            raise ValueError("u1_y coordinates must have trailing dimension 1")
        zeros = jnp.zeros((*coordinates.shape[:-1], 3), dtype=coordinates.dtype)
        return jnp.concatenate((zeros, coordinates[..., :1]), axis=-1)
    if sector == "su2_l":
        if coordinates.shape[-1:] != (3,):
            raise ValueError("su2_l coordinates must have trailing dimension 3")
        zeros = jnp.zeros((*coordinates.shape[:-1], 1), dtype=coordinates.dtype)
        return jnp.concatenate((coordinates, zeros), axis=-1)
    if sector == "sm":
        if coordinates.shape[-1:] != (12,):
            raise ValueError("sm coordinates must have trailing dimension 12")
        return jnp.concatenate((coordinates[..., 8:11], coordinates[..., 11:12]), axis=-1)
    raise ValueError(f"sector {sector!r} does not act on the Higgs doublet in this v1 wrapper")


def jax_higgs_link_field_from_patisalam_sector(
    theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    """Return fundamental Higgs BCC links from supported sector coordinates."""

    return jax_higgs_link_field_from_algebra(
        jax_higgs_coordinates_from_patisalam_sector(theta, sector=sector),
    )


def jax_higgs_site_gauge_from_patisalam_sector(
    site_theta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    """Return fundamental Higgs site gauges from supported sector coordinates."""

    return jax_higgs_site_gauge_from_algebra(
        jax_higgs_coordinates_from_patisalam_sector(site_theta, sector=sector),
    )


def jax_apply_site_local_yukawa_kick(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    step_size: float,
    yukawa_coupling: float = 1.0,
) -> jnp.ndarray:
    """Apply first-order explicit local kick ``psi -> psi - i dt beta Y(Phi) psi``."""

    lattice_shape = _validate_patisalam_state(state)
    _validate_higgs_field(phi, lattice_shape)
    if step_size == 0 or yukawa_coupling == 0:
        return state

    internal = jax_higgs_yukawa_internal_control_field(phi, dtype=state.dtype)
    beta = _dirac_beta(state.dtype)
    action = jnp.einsum("ab,...ij,...bj->...ai", beta, internal, state)
    dt = jnp.asarray(step_size * yukawa_coupling, dtype=jnp.real(state).dtype)
    return state - 1j * dt * action


def jax_patisalam_fermion_gauge_higgs_step(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    sector: HiggsCoupledSector = "u1_y",
    step_size: float,
    matter_coupling: float = 1.0,
    yukawa_coupling: float = 1.0,
    beta: float = 1.0,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: Literal["autodiff", "finite_difference"] = "finite_difference",
    force_epsilon: float = 1e-3,
    current_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one small coupled ``(fermion, gauge, Higgs)`` prototype step."""

    _validate_higgs_coupled_sector(sector)
    lattice_shape = _validate_patisalam_state(state)
    _validate_patisalam_links(links, lattice_shape)
    _validate_patisalam_momenta(momenta, lattice_shape, sector=sector)
    _validate_higgs_field(phi, lattice_shape)
    _validate_higgs_field(higgs_momentum, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)

    half_kicked_state = jax_apply_site_local_yukawa_kick(
        state,
        phi,
        step_size=0.5 * step_size,
        yukawa_coupling=yukawa_coupling,
    )
    updated_state, updated_links, updated_momenta = jax_patisalam_fermion_gauge_step_with_backreaction(
        half_kicked_state,
        links,
        momenta,
        sector=sector,
        step_size=step_size,
        matter_coupling=matter_coupling,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
        current_epsilon=current_epsilon,
    )
    updated_phi, updated_higgs_momentum = jax_higgs_leapfrog_step(
        phi,
        higgs_momentum,
        higgs_links,
        step_size=step_size,
        vev_squared=vev_squared,
        quartic=quartic,
    )
    final_state = jax_apply_site_local_yukawa_kick(
        updated_state,
        updated_phi,
        step_size=0.5 * step_size,
        yukawa_coupling=yukawa_coupling,
    )
    return final_state, updated_links, updated_momenta, updated_phi, updated_higgs_momentum, higgs_links


def jax_patisalam_fermion_gauge_higgs_diagnostics(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    sector: HiggsCoupledSector = "u1_y",
    beta: float = 1.0,
    matter_coupling: float = 1.0,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    reference_state: jnp.ndarray | None = None,
) -> PatiSalamFermionGaugeHiggsDiagnostics:
    """Return finite diagnostics for the coupled prototype state."""

    _validate_higgs_coupled_sector(sector)
    lattice_shape = _validate_patisalam_state(state)
    _validate_patisalam_links(links, lattice_shape)
    _validate_patisalam_momenta(momenta, lattice_shape, sector=sector)
    _validate_higgs_field(phi, lattice_shape)
    _validate_higgs_field(higgs_momentum, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)
    residual = jax_patisalam_gauss_residual(
        state,
        links,
        momenta,
        sector=sector,
        matter_coupling=matter_coupling,
    )
    higgs_energy = jax_higgs_energy_density(phi, higgs_links, vev_squared=vev_squared, quartic=quartic)
    drift = jnp.asarray(0, dtype=jnp.real(state).dtype)
    if reference_state is not None:
        drift = norm_drift(reference_state, state)
    return {
        "fermion_norm": state_norm_squared(state),
        "gauge_hamiltonian_density": jax_patisalam_gauge_hamiltonian_density(
            links,
            momenta,
            sector=sector,
            beta=beta,
            shapes=shapes,
        ),
        "higgs_norm": jnp.real(jnp.vdot(phi, phi)),
        "higgs_momentum_energy_density": jax_higgs_momentum_energy_density(higgs_momentum),
        "higgs_kinetic_energy_density": jax_higgs_kinetic_energy_density(phi, higgs_links),
        "higgs_potential_density": jax_higgs_potential_density(phi, vev_squared=vev_squared, quartic=quartic),
        "higgs_energy_density": jax_higgs_momentum_energy_density(higgs_momentum) + higgs_energy,
        "gauss_residual_norm": jnp.real(jnp.vdot(residual, residual)),
        "yukawa_norm_drift": drift,
    }
