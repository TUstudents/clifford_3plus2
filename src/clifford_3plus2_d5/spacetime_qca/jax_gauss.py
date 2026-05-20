"""Gauss-law and first matter-backreaction helpers for Pati-Salam sectors.

Session 37 adds the first constrained matter/gauge prototype on top of the
Session 36 no-backreaction wrapper.  The conventions match the existing BCC
pull-link layout:

* ``U[x,h]`` maps source ``x+h`` to target ``x``;
* left-trivialized momenta ``P[x,h]`` transform at the target site ``x``;
* site-local fermions transform as ``psi[x] -> G[x] psi[x]``.

The backreaction update here is intentionally explicit and small: a
finite-difference matter current kicks the link momenta before the existing
pure-gauge leapfrog is applied.  This is not a full constraint projection or a
production HMC integrator.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import jax.numpy as jnp

from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.spacetime_qca.jax_fermion_gauge import (
    PATISALAM_INTERNAL_DIM,
    jax_patisalam_dirac_step,
    jax_patisalam_fermion_gauge_step,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import (
    PatiSalamGaugeSector,
    jax_patisalam_algebra_matrix,
    jax_patisalam_generators_chiral16,
    jax_patisalam_link_from_algebra,
    jax_patisalam_project_to_coordinates,
)
from clifford_3plus2_d5.spacetime_qca.jax_step import jax_bcc_displacements
from clifford_3plus2_d5.spacetime_qca.plaquette import PlaquetteShape

ChargeCoordinateMode = Literal["raw", "gram_dual"]


def _validate_patisalam_state(state: jnp.ndarray) -> None:
    if state.ndim != 5 or state.shape[-2:] != (4, PATISALAM_INTERNAL_DIM):
        raise ValueError("state must have shape (nx, ny, nz, 4, 32)")


def _validate_patisalam_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, PATISALAM_INTERNAL_DIM, PATISALAM_INTERNAL_DIM):
        raise ValueError("links must have shape (nx, ny, nz, 8, 32, 32)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("links must match the state lattice shape")


def _sector_dimension(sector: PatiSalamGaugeSector, dtype: jnp.dtype = jnp.complex64) -> int:
    return int(jax_patisalam_generators_chiral16(sector, dtype).shape[0])


def _validate_momenta(
    momenta: jnp.ndarray,
    links: jnp.ndarray | None = None,
    *,
    sector: PatiSalamGaugeSector,
) -> None:
    sector_dim = _sector_dimension(sector, jnp.result_type(momenta, 1j))
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, sector_dim):
        raise ValueError(f"momenta must have shape (nx, ny, nz, 8, {sector_dim})")
    if links is not None and momenta.shape[:4] != links.shape[:4]:
        raise ValueError("momenta and links must share shape (nx, ny, nz, 8)")


def _generator_gram(generators: jnp.ndarray) -> jnp.ndarray:
    basis_daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    return jnp.real(jnp.einsum("aij,bji->ab", basis_daggers, generators))


def _gram_dual_coordinates(raw_moments: jnp.ndarray, generators: jnp.ndarray) -> jnp.ndarray:
    generator_count = int(generators.shape[0])
    gram = _generator_gram(generators)
    flat_raw = jnp.reshape(raw_moments, (-1, generator_count))
    flat_coordinates = jnp.linalg.solve(gram, flat_raw.T).T
    return jnp.reshape(flat_coordinates, raw_moments.shape)


@lru_cache(maxsize=1)
def _minus_displacement_indices() -> tuple[int, ...]:
    displacements = jax_bcc_displacements()
    return tuple(
        displacements.index(tuple(-component for component in displacement))
        for displacement in displacements
    )


def _fermion_overlap(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.vdot(state, jax_patisalam_dirac_step(state, links)))


def jax_patisalam_momentum_algebra(
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return anti-Hermitian momentum matrices with shape ``(..., 8, 32, 32)``."""

    _validate_momenta(momenta, sector=sector)
    return jax_patisalam_algebra_matrix(momenta, sector=sector)


def jax_patisalam_electric_divergence(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
) -> jnp.ndarray:
    """Return target-site electric divergence in sector coordinates.

    The matrix convention is
    ``sum_h (P[x,h] - U[x+h,-h]^dagger P[x+h,-h] U[x+h,-h])``.
    """

    _validate_patisalam_links(links)
    _validate_momenta(momenta, links, sector=sector)

    momentum_matrices = jax_patisalam_momentum_algebra(momenta, sector=sector)
    divergence = jnp.zeros((*links.shape[:3], PATISALAM_INTERNAL_DIM, PATISALAM_INTERNAL_DIM), dtype=links.dtype)
    minus_indices = _minus_displacement_indices()
    for index, displacement in enumerate(jax_bcc_displacements()):
        outgoing_index = minus_indices[index]
        incoming = momentum_matrices[..., index, :, :]
        outgoing_momentum = source_roll(momentum_matrices[..., outgoing_index, :, :], displacement)
        outgoing_link = source_roll(links[..., outgoing_index, :, :], displacement)
        outgoing_link_dagger = jnp.swapaxes(jnp.conj(outgoing_link), -1, -2)
        transported_outgoing = outgoing_link_dagger @ outgoing_momentum @ outgoing_link
        divergence = divergence + incoming - transported_outgoing
    return jax_patisalam_project_to_coordinates(divergence, sector=sector)


def jax_patisalam_fermion_charge_density(
    state: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    coordinate_mode: ChargeCoordinateMode = "gram_dual",
) -> jnp.ndarray:
    """Return site charge density coordinates for the chosen gauge sector.

    ``raw`` returns the moments ``<psi| iT_a |psi>``.  ``gram_dual`` converts
    those moments into the coordinate convention used for link momenta.
    """

    _validate_patisalam_state(state)
    generators = jax_patisalam_generators_chiral16(sector, jnp.result_type(state, 1j))
    charge_observables = 1j * generators
    raw = jnp.real(jnp.einsum("...si,aij,...sj->...a", jnp.conj(state), charge_observables, state))
    if coordinate_mode == "raw":
        return raw
    if coordinate_mode == "gram_dual":
        return _gram_dual_coordinates(raw, generators)
    raise ValueError(f"unknown charge coordinate mode: {coordinate_mode}")


def jax_patisalam_gauss_residual(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    matter_coupling: float = 1.0,
) -> jnp.ndarray:
    """Return ``divE - matter_coupling * rho`` in sector coordinates."""

    _validate_patisalam_state(state)
    _validate_patisalam_links(links, state.shape[:3])
    _validate_momenta(momenta, links, sector=sector)
    return jax_patisalam_electric_divergence(
        links,
        momenta,
        sector=sector,
    ) - jnp.asarray(matter_coupling, dtype=momenta.dtype) * jax_patisalam_fermion_charge_density(
        state,
        sector=sector,
    )


def jax_patisalam_fermion_link_current(
    state: jnp.ndarray,
    links: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    epsilon: float = 1e-3,
) -> jnp.ndarray:
    """Return finite-difference link-current coordinates per BCC edge.

    This is a small-lattice audit primitive. It loops over links and generator
    coordinates explicitly so the left-variation convention remains readable.
    A vectorized or analytic current is future performance work.
    """

    _validate_patisalam_state(state)
    _validate_patisalam_links(links, state.shape[:3])
    sector_dim = _sector_dimension(sector, links.dtype)
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    eps = jnp.asarray(epsilon, dtype=real_dtype)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    basis_coordinates = eps * jnp.eye(sector_dim, dtype=real_dtype)
    plus_updates = jax_patisalam_link_from_algebra(basis_coordinates, sector=sector)
    minus_updates = jax_patisalam_link_from_algebra(-basis_coordinates, sector=sector)

    current = jnp.zeros((*links.shape[:4], sector_dim), dtype=real_dtype)
    nx, ny, nz = (int(size) for size in links.shape[:3])
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for hop in range(8):
                    base_link = links[x, y, z, hop]
                    for generator_index in range(sector_dim):
                        plus_links = links.at[x, y, z, hop].set(plus_updates[generator_index] @ base_link)
                        minus_links = links.at[x, y, z, hop].set(minus_updates[generator_index] @ base_link)
                        derivative = (_fermion_overlap(state, plus_links) - _fermion_overlap(state, minus_links)) / (
                            2 * eps
                        )
                        current = current.at[x, y, z, hop, generator_index].set(derivative)
    return current


def jax_patisalam_apply_fermion_backreaction(
    momenta: jnp.ndarray,
    current: jnp.ndarray,
    *,
    step_size: float,
    matter_coupling: float = 1.0,
) -> jnp.ndarray:
    """Return ``momenta + step_size * matter_coupling * current``."""

    if current.shape != momenta.shape:
        raise ValueError("current and momenta must have the same shape")
    return momenta + jnp.asarray(step_size * matter_coupling, dtype=momenta.dtype) * current


def jax_patisalam_fermion_gauge_step_with_backreaction(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector = "su4",
    step_size: float,
    matter_coupling: float = 1.0,
    beta: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: CompactLieForceMethod = "finite_difference",
    force_epsilon: float = 1e-3,
    force_chunk_size: int | None = None,
    current_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one source-kick plus pure-gauge leapfrog plus fermion step."""

    if matter_coupling == 0 or step_size == 0:
        return jax_patisalam_fermion_gauge_step(
            state,
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

    current = jax_patisalam_fermion_link_current(
        state,
        links,
        sector=sector,
        epsilon=current_epsilon,
    )
    sourced_momenta = jax_patisalam_apply_fermion_backreaction(
        momenta,
        current,
        step_size=step_size,
        matter_coupling=matter_coupling,
    )
    return jax_patisalam_fermion_gauge_step(
        state,
        links,
        sourced_momenta,
        sector=sector,
        step_size=step_size,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
        force_chunk_size=force_chunk_size,
    )
