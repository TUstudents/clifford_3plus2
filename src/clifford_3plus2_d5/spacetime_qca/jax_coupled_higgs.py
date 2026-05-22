"""JAX coupled fermion/gauge/Higgs prototype.

Session 40 is the first ``spacetime_qca`` layer with all field types present:

* BCC Dirac fermions on the chiral16 internal carrier;
* compact Pati-Salam/SM gauge links and momenta;
* a site-local electroweak Higgs doublet with a conjugate momentum;
* a site-local Yukawa update sourced by ``Phi(x)``.

The implementation is intentionally conservative.  Higgs links are explicit
``2 x 2`` electroweak links, not inferred from the ``32 x 32`` chiral16 gauge
links.  The default Yukawa update remains the Session 40 first-order explicit
kick for compatibility, while Session 45 adds an exact local unitary option.
Session 57 keeps that exact unitary as the public ``unitary`` mode, but
evaluates it with the selected Higgs-map cubic polynomial instead of a local
eigensolve.  Session 59 adds an off-by-default finite-difference Higgs-current
kick into gauge momenta.  Session 61 adds an off-by-default bounded
Gauss-descent projection for gauge momenta.
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
    ChargeCoordinateMode,
    jax_patisalam_electric_divergence,
    jax_patisalam_fermion_charge_density,
    jax_patisalam_fermion_gauge_step_with_backreaction,
)
from clifford_3plus2_d5.spacetime_qca.jax_gauge_force import CompactLieForceMethod
from clifford_3plus2_d5.spacetime_qca.jax_higgs import (
    jax_higgs_energy_density,
    jax_higgs_generators,
    jax_higgs_kinetic_energy_density,
    jax_higgs_link_field_from_algebra,
    jax_higgs_link_current,
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
YukawaUpdateMode = Literal["first_order", "unitary", "unitary_eigh"]


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


def _higgs_generator_gram(generators: jnp.ndarray) -> jnp.ndarray:
    basis_daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    return jnp.real(jnp.einsum("aij,bji->ab", basis_daggers, generators))


def _higgs_gram_dual_coordinates(raw_moments: jnp.ndarray, generators: jnp.ndarray) -> jnp.ndarray:
    generator_count = int(generators.shape[0])
    gram = _higgs_generator_gram(generators)
    flat_raw = jnp.reshape(raw_moments, (-1, generator_count))
    flat_coordinates = jnp.linalg.solve(gram, flat_raw.T).T
    return jnp.reshape(flat_coordinates, raw_moments.shape)


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


def jax_higgs_charge_density(
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    *,
    coordinate_mode: ChargeCoordinateMode = "gram_dual",
) -> jnp.ndarray:
    """Return site-local Higgs charge density in electroweak coordinates.

    The raw moments use anti-Hermitian Higgs generators ``T_a`` and the
    canonical momentum-map convention ``Re(Pi^dagger T_a Phi)``.  ``gram_dual``
    converts those moments into the same generator-coordinate convention used
    by link momenta and currents.
    """

    lattice_shape = (int(phi.shape[0]), int(phi.shape[1]), int(phi.shape[2]))
    _validate_higgs_field(phi)
    _validate_higgs_field(higgs_momentum, lattice_shape)
    generators = jax_higgs_generators(dtype=jnp.result_type(phi, higgs_momentum, 1j))
    raw = jnp.real(jnp.einsum("...i,aij,...j->...a", jnp.conj(higgs_momentum), generators, phi))
    if coordinate_mode == "raw":
        return raw
    if coordinate_mode == "gram_dual":
        return _higgs_gram_dual_coordinates(raw, generators)
    raise ValueError(f"unknown charge coordinate mode: {coordinate_mode}")


def jax_higgs_charge_to_patisalam_sector(
    higgs_charge: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    """Embed Higgs site charge coordinates into a supported gauge sector."""

    if higgs_charge.ndim != 4 or higgs_charge.shape[-1:] != (4,):
        raise ValueError("Higgs charge must have shape (nx, ny, nz, 4)")
    if sector == "u1_y":
        return higgs_charge[..., 3:4]
    if sector == "su2_l":
        return higgs_charge[..., :3]
    if sector == "sm":
        zeros = jnp.zeros((*higgs_charge.shape[:-1], 12), dtype=higgs_charge.dtype)
        zeros = zeros.at[..., 8:11].set(higgs_charge[..., :3])
        return zeros.at[..., 11].set(higgs_charge[..., 3])
    raise ValueError(f"sector {sector!r} does not act on the Higgs doublet in this v1 wrapper")


def jax_higgs_charge_density_from_patisalam_sector(
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
    coordinate_mode: ChargeCoordinateMode = "gram_dual",
) -> jnp.ndarray:
    """Return Higgs charge density embedded in the selected gauge sector."""

    return jax_higgs_charge_to_patisalam_sector(
        jax_higgs_charge_density(phi, higgs_momentum, coordinate_mode=coordinate_mode),
        sector=sector,
    )


def jax_higgs_current_to_patisalam_sector(
    higgs_current: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    """Embed Higgs current coordinates into a supported Pati-Salam sector."""

    if higgs_current.ndim != 5 or higgs_current.shape[-2:] != (8, 4):
        raise ValueError("Higgs current must have shape (nx, ny, nz, 8, 4)")
    if sector == "u1_y":
        return higgs_current[..., 3:4]
    if sector == "su2_l":
        return higgs_current[..., :3]
    if sector == "sm":
        zeros = jnp.zeros((*higgs_current.shape[:-1], 12), dtype=higgs_current.dtype)
        zeros = zeros.at[..., 8:11].set(higgs_current[..., :3])
        return zeros.at[..., 11].set(higgs_current[..., 3])
    raise ValueError(f"sector {sector!r} does not act on the Higgs doublet in this v1 wrapper")


def jax_higgs_link_current_from_patisalam_sector(
    phi: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
    epsilon: float = 1e-3,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
) -> jnp.ndarray:
    """Return Higgs link current embedded in the selected gauge sector."""

    return jax_higgs_current_to_patisalam_sector(
        jax_higgs_link_current(
            phi,
            higgs_links,
            epsilon=epsilon,
            vev_squared=vev_squared,
            quartic=quartic,
        ),
        sector=sector,
    )


def jax_patisalam_apply_higgs_backreaction(
    momenta: jnp.ndarray,
    higgs_current: jnp.ndarray,
    *,
    step_size: float,
    higgs_coupling: float = 1.0,
) -> jnp.ndarray:
    """Return ``momenta + step_size * higgs_coupling * higgs_current``."""

    if higgs_current.shape != momenta.shape:
        raise ValueError("Higgs current and momenta must have the same shape")
    return momenta + jnp.asarray(step_size * higgs_coupling, dtype=momenta.dtype) * higgs_current


def jax_patisalam_fermion_higgs_gauss_residual(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    *,
    sector: HiggsCoupledSector = "u1_y",
    matter_coupling: float = 1.0,
    higgs_coupling: float = 1.0,
) -> jnp.ndarray:
    """Return ``divE - g_f rho_f - g_H rho_H`` in sector coordinates."""

    _validate_higgs_coupled_sector(sector)
    lattice_shape = _validate_patisalam_state(state)
    _validate_patisalam_links(links, lattice_shape)
    _validate_patisalam_momenta(momenta, lattice_shape, sector=sector)
    _validate_higgs_field(phi, lattice_shape)
    _validate_higgs_field(higgs_momentum, lattice_shape)
    divergence = jax_patisalam_electric_divergence(links, momenta, sector=sector)
    fermion_charge = jax_patisalam_fermion_charge_density(state, sector=sector)
    higgs_charge = jax_higgs_charge_density_from_patisalam_sector(phi, higgs_momentum, sector=sector)
    return (
        divergence
        - jnp.asarray(matter_coupling, dtype=momenta.dtype) * fermion_charge
        - jnp.asarray(higgs_coupling, dtype=momenta.dtype) * higgs_charge
    )


def jax_patisalam_fermion_higgs_gauss_descent_step(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    *,
    sector: HiggsCoupledSector = "u1_y",
    matter_coupling: float = 1.0,
    higgs_coupling: float = 1.0,
    descent_step_size: float = 0.05,
) -> jnp.ndarray:
    """Return a diagnostic gradient-descent step for Gauss residual norm."""

    if descent_step_size < 0:
        raise ValueError("descent_step_size must be nonnegative")

    def residual_objective(candidate_momenta: jnp.ndarray) -> jnp.ndarray:
        residual = jax_patisalam_fermion_higgs_gauss_residual(
            state,
            links,
            candidate_momenta,
            phi,
            higgs_momentum,
            sector=sector,
            matter_coupling=matter_coupling,
            higgs_coupling=higgs_coupling,
        )
        return 0.5 * jnp.real(jnp.vdot(residual, residual))

    gradient = jax.grad(residual_objective)(momenta)
    return momenta - jnp.asarray(descent_step_size, dtype=momenta.dtype) * gradient


def jax_patisalam_apply_gauss_descent_projection(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    phi: jnp.ndarray,
    higgs_momentum: jnp.ndarray,
    *,
    sector: HiggsCoupledSector = "u1_y",
    matter_coupling: float = 1.0,
    higgs_coupling: float = 1.0,
    projection_steps: int = 0,
    projection_step_size: float = 0.0,
) -> jnp.ndarray:
    """Return momenta after bounded diagnostic descent on the Gauss residual."""

    if projection_steps < 0:
        raise ValueError("projection_steps must be nonnegative")
    if projection_step_size < 0:
        raise ValueError("projection_step_size must be nonnegative")
    if projection_steps == 0 or projection_step_size == 0:
        return momenta

    updated = momenta
    for _ in range(projection_steps):
        updated = jax_patisalam_fermion_higgs_gauss_descent_step(
            state,
            links,
            updated,
            phi,
            higgs_momentum,
            sector=sector,
            matter_coupling=matter_coupling,
            higgs_coupling=higgs_coupling,
            descent_step_size=projection_step_size,
        )
    return updated


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


def _matrix_from_eigh(eigenvectors: jnp.ndarray, function_values: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ik,...k,...jk->...ij", eigenvectors, function_values, jnp.conj(eigenvectors))


def _cos_sin_from_eigh(hermitian: jnp.ndarray, scale: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(hermitian)
    return (
        _matrix_from_eigh(eigenvectors, jnp.cos(scale * eigenvalues)),
        _matrix_from_eigh(eigenvectors, jnp.sin(scale * eigenvalues)),
    )


def _selected_yukawa_lambda_squared(phi: jnp.ndarray) -> jnp.ndarray:
    """Return ``lambda(Phi)^2`` for the selected Session 38 Higgs map.

    For the deterministic two-complex Higgs slice, the internal Hermitian
    matrix satisfies ``Y(Phi)^3 = lambda(Phi)^2 Y(Phi)`` with
    ``lambda(Phi)^2 = 256 ||Phi||^2``.
    """

    norm_squared = jnp.real(jnp.sum(jnp.conj(phi) * phi, axis=-1))
    return jnp.asarray(256.0, dtype=norm_squared.dtype) * norm_squared


def _cos_sin_from_cubic_yukawa(
    hermitian: jnp.ndarray,
    phi: jnp.ndarray,
    scale: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``cos(scale Y)`` and ``sin(scale Y)`` from ``Y^3=lambda^2 Y``."""

    lambda_squared = _selected_yukawa_lambda_squared(phi)
    lambda_value = jnp.sqrt(jnp.maximum(lambda_squared, jnp.asarray(0, dtype=lambda_squared.dtype)))
    safe_lambda = jnp.where(lambda_squared > 0, lambda_value, jnp.ones_like(lambda_value))
    safe_lambda_squared = jnp.where(lambda_squared > 0, lambda_squared, jnp.ones_like(lambda_squared))
    sin_coefficient = jnp.where(
        lambda_squared > 0,
        jnp.sin(scale * lambda_value) / safe_lambda,
        jnp.broadcast_to(scale, lambda_squared.shape),
    )
    cos_coefficient = jnp.where(
        lambda_squared > 0,
        (jnp.cos(scale * lambda_value) - 1.0) / safe_lambda_squared,
        jnp.broadcast_to(-0.5 * scale * scale, lambda_squared.shape),
    )
    identity = jnp.eye(hermitian.shape[-1], dtype=hermitian.dtype)
    hermitian_squared = hermitian @ hermitian
    cos_internal = identity + cos_coefficient[..., None, None].astype(hermitian.dtype) * hermitian_squared
    sin_internal = sin_coefficient[..., None, None].astype(hermitian.dtype) * hermitian
    return cos_internal, sin_internal


def _apply_site_local_yukawa_cos_sin(
    state: jnp.ndarray,
    cos_internal: jnp.ndarray,
    sin_internal: jnp.ndarray,
) -> jnp.ndarray:
    cos_action = jnp.einsum("...ij,...aj->...ai", cos_internal, state)
    sin_internal_action = jnp.einsum("...ij,...aj->...ai", sin_internal, state)
    beta_action = jnp.einsum("ab,...bj->...aj", _dirac_beta(state.dtype), sin_internal_action)
    return cos_action - 1j * beta_action


def jax_apply_site_local_yukawa_unitary(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    step_size: float,
    yukawa_coupling: float = 1.0,
) -> jnp.ndarray:
    """Apply exact local unitary ``exp(-i dt y beta x Y(Phi[x]))``.

    Since ``beta^2 = I`` and ``beta`` commutes with the internal matrix,

    ``exp(-i a beta x Y) = I x cos(aY) - i beta x sin(aY)``.

    Session 57 evaluates ``cos(aY)`` and ``sin(aY)`` with the selected
    Higgs-map identity ``Y(Phi)^3 = lambda(Phi)^2 Y(Phi)``.  This keeps the
    exact local unitary semantics while avoiding the old local eigensolve.
    """

    lattice_shape = _validate_patisalam_state(state)
    _validate_higgs_field(phi, lattice_shape)
    if step_size == 0 or yukawa_coupling == 0:
        return state

    internal = jax_higgs_yukawa_internal_control_field(phi, dtype=state.dtype)
    dt = jnp.asarray(step_size * yukawa_coupling, dtype=jnp.real(state).dtype)
    cos_internal, sin_internal = _cos_sin_from_cubic_yukawa(internal, phi, dt)
    return _apply_site_local_yukawa_cos_sin(state, cos_internal, sin_internal)


def jax_apply_site_local_yukawa_unitary_eigh(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    step_size: float,
    yukawa_coupling: float = 1.0,
) -> jnp.ndarray:
    """Apply the old eigensolve-based exact local unitary oracle."""

    lattice_shape = _validate_patisalam_state(state)
    _validate_higgs_field(phi, lattice_shape)
    if step_size == 0 or yukawa_coupling == 0:
        return state

    internal = jax_higgs_yukawa_internal_control_field(phi, dtype=state.dtype)
    dt = jnp.asarray(step_size * yukawa_coupling, dtype=jnp.real(state).dtype)
    cos_internal, sin_internal = _cos_sin_from_eigh(internal, dt)
    return _apply_site_local_yukawa_cos_sin(state, cos_internal, sin_internal)


def jax_apply_site_local_yukawa_update(
    state: jnp.ndarray,
    phi: jnp.ndarray,
    *,
    step_size: float,
    yukawa_coupling: float = 1.0,
    mode: YukawaUpdateMode = "first_order",
) -> jnp.ndarray:
    """Apply the selected site-local Yukawa update."""

    if mode == "first_order":
        return jax_apply_site_local_yukawa_kick(
            state,
            phi,
            step_size=step_size,
            yukawa_coupling=yukawa_coupling,
        )
    if mode == "unitary":
        return jax_apply_site_local_yukawa_unitary(
            state,
            phi,
            step_size=step_size,
            yukawa_coupling=yukawa_coupling,
        )
    if mode == "unitary_eigh":
        return jax_apply_site_local_yukawa_unitary_eigh(
            state,
            phi,
            step_size=step_size,
            yukawa_coupling=yukawa_coupling,
        )
    raise ValueError(f"unknown Yukawa update mode: {mode!r}")


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
    higgs_coupling: float = 0.0,
    yukawa_coupling: float = 1.0,
    beta: float = 1.0,
    vev_squared: float = 1.0,
    quartic: float = 1.0,
    shapes: tuple[PlaquetteShape, ...] | None = None,
    force_method: CompactLieForceMethod = "finite_difference",
    force_epsilon: float = 1e-3,
    force_chunk_size: int | None = None,
    current_epsilon: float = 1e-3,
    higgs_current_epsilon: float = 1e-3,
    yukawa_mode: YukawaUpdateMode = "first_order",
    gauss_projection_steps: int = 0,
    gauss_projection_step_size: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return one small coupled ``(fermion, gauge, Higgs)`` prototype step."""

    if gauss_projection_steps < 0:
        raise ValueError("gauss_projection_steps must be nonnegative")
    if gauss_projection_step_size < 0:
        raise ValueError("gauss_projection_step_size must be nonnegative")

    _validate_higgs_coupled_sector(sector)
    lattice_shape = _validate_patisalam_state(state)
    _validate_patisalam_links(links, lattice_shape)
    _validate_patisalam_momenta(momenta, lattice_shape, sector=sector)
    _validate_higgs_field(phi, lattice_shape)
    _validate_higgs_field(higgs_momentum, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)

    half_kicked_state = jax_apply_site_local_yukawa_update(
        state,
        phi,
        step_size=0.5 * step_size,
        yukawa_coupling=yukawa_coupling,
        mode=yukawa_mode,
    )
    sourced_momenta = momenta
    if higgs_coupling != 0 and step_size != 0:
        higgs_current = jax_higgs_link_current_from_patisalam_sector(
            phi,
            higgs_links,
            sector=sector,
            epsilon=higgs_current_epsilon,
            vev_squared=vev_squared,
            quartic=quartic,
        )
        sourced_momenta = jax_patisalam_apply_higgs_backreaction(
            momenta,
            higgs_current,
            step_size=step_size,
            higgs_coupling=higgs_coupling,
        )
    updated_state, updated_links, updated_momenta = jax_patisalam_fermion_gauge_step_with_backreaction(
        half_kicked_state,
        links,
        sourced_momenta,
        sector=sector,
        step_size=step_size,
        matter_coupling=matter_coupling,
        beta=beta,
        shapes=shapes,
        force_method=force_method,
        force_epsilon=force_epsilon,
        force_chunk_size=force_chunk_size,
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
    final_state = jax_apply_site_local_yukawa_update(
        updated_state,
        updated_phi,
        step_size=0.5 * step_size,
        yukawa_coupling=yukawa_coupling,
        mode=yukawa_mode,
    )
    projected_momenta = updated_momenta
    if step_size != 0:
        projected_momenta = jax_patisalam_apply_gauss_descent_projection(
            final_state,
            updated_links,
            updated_momenta,
            updated_phi,
            updated_higgs_momentum,
            sector=sector,
            matter_coupling=matter_coupling,
            higgs_coupling=higgs_coupling,
            projection_steps=gauss_projection_steps,
            projection_step_size=gauss_projection_step_size,
        )
    return final_state, updated_links, projected_momenta, updated_phi, updated_higgs_momentum, higgs_links


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
    higgs_coupling: float = 0.0,
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
    residual = jax_patisalam_fermion_higgs_gauss_residual(
        state,
        links,
        momenta,
        phi,
        higgs_momentum,
        sector=sector,
        matter_coupling=matter_coupling,
        higgs_coupling=higgs_coupling,
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
