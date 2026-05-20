"""No-backreaction fermion/gauge coupling helpers for Pati-Salam sectors.

Session 36 couples the existing finite BCC Dirac step to the compact
Pati-Salam/SM gauge-link leapfrog stack.  The coupling is intentionally
one-way: gauge links and link momenta evolve under the pure Wilson
Hamiltonian, and the fermion field is then transported through the updated
background links.  There is no fermion current source, Gauss-law projection,
or backreaction term in this module.
"""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp

from clifford_3plus2_d5.sim.state import state_norm_squared
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import (
    PatiSalamGaugeSector,
    jax_patisalam_gauge_hamiltonian_density,
    jax_patisalam_leapfrog_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
from clifford_3plus2_d5.spacetime_qca.jax_step import jax_dirac_step_with_links
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape

PATISALAM_INTERNAL_DIM = 32


class PatiSalamFermionGaugeDiagnostics(TypedDict):
    """Small diagnostics payload for no-backreaction coupled runs."""

    fermion_norm: jnp.ndarray
    gauge_hamiltonian_density: jnp.ndarray


def _validate_patisalam_state(state: jnp.ndarray) -> None:
    if state.ndim != 5 or state.shape[-2:] != (4, PATISALAM_INTERNAL_DIM):
        raise ValueError(
            "Pati-Salam Dirac state must have shape "
            "(nx, ny, nz, 4, 32)",
        )


def _validate_patisalam_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int]) -> None:
    if links.ndim != 6 or links.shape[:3] != lattice_shape or links.shape[3:] != (
        8,
        PATISALAM_INTERNAL_DIM,
        PATISALAM_INTERNAL_DIM,
    ):
        raise ValueError(
            "Pati-Salam links must have shape "
            "(nx, ny, nz, 8, 32, 32) matching the state lattice",
        )


def _validate_patisalam_momenta(momenta: jnp.ndarray, links: jnp.ndarray) -> None:
    if momenta.ndim != 5 or momenta.shape[:4] != links.shape[:4]:
        raise ValueError(
            "Pati-Salam momenta must have shape "
            "(nx, ny, nz, 8, sector_dim) matching the link field",
        )


def jax_transform_patisalam_dirac_state(
    state: jnp.ndarray,
    site_gauge: jnp.ndarray,
) -> jnp.ndarray:
    """Apply a site-local internal gauge transform to a Dirac/internal state."""

    _validate_patisalam_state(state)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != state.shape[:3] or site_gauge.shape[-2:] != (
        PATISALAM_INTERNAL_DIM,
        PATISALAM_INTERNAL_DIM,
    ):
        raise ValueError(
            "site_gauge must have shape "
            "(nx, ny, nz, 32, 32) matching the state lattice",
        )
    return jnp.einsum("...ab,...sb->...sa", site_gauge, state)


def jax_patisalam_dirac_step(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Apply one BCC Dirac step through Pati-Salam/SM chiral16 links."""

    _validate_patisalam_state(state)
    _validate_patisalam_links(links, state.shape[:3])
    return jax_dirac_step_with_links(state, links)


def jax_patisalam_fermion_gauge_step(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: CompactLieForceMethod = "finite_difference",
    force_epsilon: float = 1e-3,
    force_chunk_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one no-backreaction ``(fermion, links, momenta)`` update.

    Ordering convention: first evolve ``(links, momenta)`` with the pure
    gauge leapfrog step, then transport ``state`` through the updated links.
    The fermion state is a spectator for gauge dynamics in this v1 coupling.
    """

    _validate_patisalam_state(state)
    _validate_patisalam_links(links, state.shape[:3])
    _validate_patisalam_momenta(momenta, links)

    updated_links, updated_momenta = jax_patisalam_leapfrog_step(
        links,
        momenta,
        sector=sector,
        step_size=step_size,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
        force_chunk_size=force_chunk_size,
    )
    updated_state = jax_patisalam_dirac_step(state, updated_links)
    return updated_state, updated_links, updated_momenta


def jax_patisalam_fermion_gauge_energy_density(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> PatiSalamFermionGaugeDiagnostics:
    """Return fermion norm plus pure-gauge Hamiltonian-density diagnostics."""

    _validate_patisalam_state(state)
    _validate_patisalam_links(links, state.shape[:3])
    _validate_patisalam_momenta(momenta, links)
    return {
        "fermion_norm": state_norm_squared(state),
        "gauge_hamiltonian_density": jax_patisalam_gauge_hamiltonian_density(
            links,
            momenta,
            sector=sector,
            beta=beta,
            shapes=shapes,
        ),
    }
