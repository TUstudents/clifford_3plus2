"""Coupled gauge-Higgs backreaction for QCA_SMv0.

Stage 9 couples the Stage 8 Higgs field back into the electroweak gauge
momenta.  The Higgs current is implemented as the left-trivialized derivative
of the gauge-covariant Higgs gradient energy with respect to the BCC link
field.  Color remains a spectator for this stage; the resulting Higgs current
can be embedded into the full 12-coordinate SM gauge-momentum layout as zeros
on the first eight color generators and electroweak components on indices
8..11.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_GENERATOR_COUNT
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM, sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    SM_HIGGS_GENERATOR_COUNT,
    HiggsDynamicsParameters,
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    deterministic_higgs_theta,
    sm_higgs_algebra_matrix_field,
    sm_higgs_force,
    sm_higgs_generators,
    sm_higgs_gradient_density,
    sm_higgs_hamiltonian_density,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
    sm_higgs_site_gauge_from_algebra,
    sm_identity_higgs_links,
    sm_pure_gauge_higgs_links_from_site_algebra,
    sm_transform_higgs_field,
    sm_transform_higgs_links,
)
from clifford_3plus2_d5.sim.lattice import source_roll


class GaugeHiggsBackreactionDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 9 coupled gauge-Higgs backreaction."""

    projection_roundtrip_residual: jnp.ndarray
    higgs_link_force_vacuum_norm: jnp.ndarray
    higgs_link_force_nonzero_norm: jnp.ndarray
    higgs_link_force_covariance_residual: jnp.ndarray
    charge_covariance_residual: jnp.ndarray
    gauss_covariance_residual: jnp.ndarray
    gauss_vacuum_residual: jnp.ndarray
    sm_embedding_roundtrip_residual: jnp.ndarray
    coupled_link_unitarity_residual: jnp.ndarray
    coupled_hamiltonian_drift: jnp.ndarray
    coupled_reversibility_residual: jnp.ndarray
    jit_delta_field: jnp.ndarray
    jit_delta_links: jnp.ndarray
    jit_delta_higgs_momenta: jnp.ndarray
    jit_delta_link_momenta: jnp.ndarray


def _validate_higgs_link_momenta(momenta: jnp.ndarray, links: jnp.ndarray | None = None) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, SM_HIGGS_GENERATOR_COUNT):
        raise ValueError("Higgs gauge momenta must have shape (nx, ny, nz, 8, 4)")
    if links is not None and momenta.shape[:4] != links.shape[:4]:
        raise ValueError("Higgs gauge momenta and links must share shape (nx, ny, nz, 8)")


def _validate_higgs_field_pair(field: jnp.ndarray, momenta: jnp.ndarray | None = None) -> tuple[int, int, int]:
    if field.ndim != 4 or field.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if momenta is not None and (momenta.ndim != 4 or momenta.shape != field.shape):
        raise ValueError("Higgs momenta must match the Higgs field shape")
    return int(field.shape[0]), int(field.shape[1]), int(field.shape[2])


@lru_cache(maxsize=4)
def _higgs_generator_gram_numpy(dtype_name: str) -> np.ndarray:
    generators = np.asarray(sm_higgs_generators().astype(np.dtype(dtype_name)))
    daggers = np.swapaxes(np.conj(generators), -1, -2)
    gram = np.real(np.einsum("aij,bji->ab", daggers, generators))
    return gram.astype(np.float32, copy=False)


def sm_higgs_generator_gram(*, dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the Higgs-generator Hilbert-Schmidt Gram matrix."""

    return jnp.asarray(_higgs_generator_gram_numpy(np.dtype(dtype).name), dtype=jnp.float32)


def sm_higgs_project_to_coordinates(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project Higgs electroweak algebra matrices onto 4 coordinates."""

    if matrix.shape[-2:] != (SM_HIGGS_DIM, SM_HIGGS_DIM):
        raise ValueError("matrix must have trailing shape (2, 2)")
    generators = sm_higgs_generators().astype(jnp.result_type(matrix, 1j))
    daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    gram = sm_higgs_generator_gram(dtype=generators.dtype)
    rhs = jnp.real(jnp.einsum("aij,...ji->...a", daggers, matrix.astype(generators.dtype)))
    flat_rhs = rhs.reshape((-1, SM_HIGGS_GENERATOR_COUNT))
    flat_coordinates = jnp.linalg.solve(gram, flat_rhs.T).T
    return flat_coordinates.reshape(rhs.shape)


def sm_higgs_link_momentum_algebra(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian Higgs link-momentum matrices."""

    _validate_higgs_link_momenta(momenta)
    return sm_higgs_algebra_matrix_field(momenta)


def sm_transform_higgs_link_momenta(momenta: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site adjoint transforms to Higgs link momenta."""

    _validate_higgs_link_momenta(momenta)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != momenta.shape[:3] or site_gauge.shape[-2:] != (
        SM_HIGGS_DIM,
        SM_HIGGS_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 2, 2)")
    momentum_matrices = sm_higgs_link_momentum_algebra(momenta)
    gauge_dagger = jnp.swapaxes(jnp.conj(site_gauge), -1, -2)
    transformed = jnp.einsum("...ab,...hbc,...cd->...had", site_gauge, momentum_matrices, gauge_dagger)
    return sm_higgs_project_to_coordinates(transformed)


def sm_higgs_link_momentum_kinetic_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return the Higgs electroweak link-momentum kinetic density."""

    momentum_matrices = sm_higgs_link_momentum_algebra(momenta)
    densities = jnp.real(
        jnp.einsum(
            "...ij,...ji->...",
            jnp.swapaxes(jnp.conj(momentum_matrices), -1, -2),
            momentum_matrices,
        ),
    )
    return 0.5 * jnp.mean(densities / jnp.asarray(SM_HIGGS_DIM, dtype=densities.dtype))


def sm_coupled_higgs_gauge_hamiltonian_density(
    field: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    links: jnp.ndarray,
    link_momenta: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> jnp.ndarray:
    """Return scalar Hamiltonian plus Higgs electroweak link kinetic density."""

    _validate_higgs_link_momenta(link_momenta, links)
    return sm_higgs_hamiltonian_density(field, higgs_momenta, links, parameters=parameters) + (
        sm_higgs_link_momentum_kinetic_density(link_momenta)
    )


def _generator_left_updates(links: jnp.ndarray, epsilon: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    generators = sm_higgs_generators().astype(links.dtype)
    plus = jax.vmap(jsp_linalg.expm)(step * generators)
    minus = jax.vmap(jsp_linalg.expm)(-step * generators)
    return plus, minus


def _force_coordinate_indices(links: jnp.ndarray) -> np.ndarray:
    return np.asarray(list(np.ndindex(*links.shape[:4], SM_HIGGS_GENERATOR_COUNT)), dtype=np.int32)


def sm_higgs_left_gauge_force(
    field: jnp.ndarray,
    links: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    epsilon: float = 1e-3,
) -> jnp.ndarray:
    """Return the Higgs-gradient derivative with respect to left link updates."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _validate_higgs_field_pair(field)
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    plus_updates, minus_updates = _generator_left_updates(links, epsilon)
    coordinate_indices = jnp.asarray(_force_coordinate_indices(links))

    def energy_for_index(index: jnp.ndarray, updates: jnp.ndarray) -> jnp.ndarray:
        update = updates[index[4]]
        updated_link = update @ links[index[0], index[1], index[2], index[3]]
        perturbed_links = links.at[index[0], index[1], index[2], index[3]].set(updated_link)
        return sm_higgs_gradient_density(field, perturbed_links, parameters=parameters)

    batched_energy = jax.vmap(energy_for_index, in_axes=(0, None))
    derivatives = (batched_energy(coordinate_indices, plus_updates) - batched_energy(coordinate_indices, minus_updates)) / (
        2 * step
    )
    force = jnp.zeros((*links.shape[:4], SM_HIGGS_GENERATOR_COUNT), dtype=real_dtype)
    force = force.at[
        coordinate_indices[:, 0],
        coordinate_indices[:, 1],
        coordinate_indices[:, 2],
        coordinate_indices[:, 3],
        coordinate_indices[:, 4],
    ].set(derivatives)
    return force


def sm_apply_higgs_link_momentum_update(links: jnp.ndarray, momenta: jnp.ndarray, *, step_size: float) -> jnp.ndarray:
    """Apply ``U -> exp(step_size P) U`` to Higgs electroweak links."""

    _validate_higgs_link_momenta(momenta, links)
    updates = sm_higgs_link_field_from_algebra(jnp.asarray(step_size, dtype=momenta.dtype) * momenta)
    return jnp.einsum("...ij,...jk->...ik", updates, links)


def sm_coupled_higgs_gauge_leapfrog_step(
    field: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    links: jnp.ndarray,
    link_momenta: jnp.ndarray,
    *,
    step_size: float,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    force_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance Higgs and electroweak links with mutual no-fermion backreaction."""

    lattice_shape = _validate_higgs_field_pair(field, higgs_momenta)
    if links.shape[:3] != lattice_shape:
        raise ValueError("links must match the Higgs lattice")
    _validate_higgs_link_momenta(link_momenta, links)
    dt = jnp.asarray(step_size, dtype=higgs_momenta.dtype)

    first_higgs_force = sm_higgs_force(field, links, parameters=parameters)
    first_link_force = sm_higgs_left_gauge_force(field, links, parameters=parameters, epsilon=force_epsilon)
    half_higgs_momenta = higgs_momenta + 0.5 * dt * first_higgs_force
    half_link_momenta = link_momenta - 0.5 * dt * first_link_force

    updated_links = sm_apply_higgs_link_momentum_update(links, half_link_momenta, step_size=step_size)
    updated_field = field + dt * half_higgs_momenta

    second_higgs_force = sm_higgs_force(updated_field, updated_links, parameters=parameters)
    second_link_force = sm_higgs_left_gauge_force(updated_field, updated_links, parameters=parameters, epsilon=force_epsilon)
    updated_higgs_momenta = half_higgs_momenta + 0.5 * dt * second_higgs_force
    updated_link_momenta = half_link_momenta - 0.5 * dt * second_link_force
    return updated_field, updated_higgs_momenta, updated_links, updated_link_momenta


@lru_cache(maxsize=1)
def _minus_displacement_indices() -> tuple[int, ...]:
    return tuple(
        BCC_DISPLACEMENTS.index(tuple(-component for component in displacement))
        for displacement in BCC_DISPLACEMENTS
    )


def sm_higgs_electric_divergence(links: jnp.ndarray, momenta: jnp.ndarray) -> jnp.ndarray:
    """Return target-site electric divergence in Higgs algebra coordinates."""

    _validate_higgs_link_momenta(momenta, links)
    momentum_matrices = sm_higgs_link_momentum_algebra(momenta)
    divergence = jnp.zeros((*links.shape[:3], SM_HIGGS_DIM, SM_HIGGS_DIM), dtype=links.dtype)
    minus_indices = _minus_displacement_indices()
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        outgoing_index = minus_indices[index]
        incoming = momentum_matrices[..., index, :, :]
        outgoing_momentum = source_roll(momentum_matrices[..., outgoing_index, :, :], displacement)
        outgoing_link = source_roll(links[..., outgoing_index, :, :], displacement)
        outgoing_link_dagger = jnp.swapaxes(jnp.conj(outgoing_link), -1, -2)
        transported_outgoing = outgoing_link_dagger @ outgoing_momentum @ outgoing_link
        divergence = divergence + incoming - transported_outgoing
    return sm_higgs_project_to_coordinates(divergence)


def sm_higgs_charge_density(field: jnp.ndarray, momenta: jnp.ndarray) -> jnp.ndarray:
    """Return the local Higgs gauge-charge density in generator coordinates."""

    _validate_higgs_field_pair(field, momenta)
    generators = sm_higgs_generators().astype(jnp.result_type(field, momenta, 1j))
    return -2.0 * jnp.real(jnp.einsum("...i,aij,...j->...a", jnp.conj(momenta), generators, field))


def sm_higgs_gauss_constraint(
    links: jnp.ndarray,
    link_momenta: jnp.ndarray,
    field: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
) -> jnp.ndarray:
    """Return electric divergence minus local Higgs charge density."""

    return sm_higgs_electric_divergence(links, link_momenta) - sm_higgs_charge_density(field, higgs_momenta)


def sm_embed_higgs_to_sm_momenta(higgs_momenta: jnp.ndarray) -> jnp.ndarray:
    """Embed Higgs electroweak momenta into full SM gauge-momentum coordinates."""

    _validate_higgs_link_momenta(higgs_momenta)
    embedded = jnp.zeros((*higgs_momenta.shape[:4], SM_GENERATOR_COUNT), dtype=higgs_momenta.dtype)
    return embedded.at[..., 8:12].set(higgs_momenta)


def sm_extract_higgs_from_sm_momenta(sm_momenta: jnp.ndarray) -> jnp.ndarray:
    """Extract electroweak Higgs-representation components from SM momenta."""

    if sm_momenta.ndim != 5 or sm_momenta.shape[-2:] != (8, SM_GENERATOR_COUNT):
        raise ValueError("SM momenta must have shape (nx, ny, nz, 8, 12)")
    return sm_momenta[..., 8:12]


def deterministic_higgs_link_momenta(
    lattice_shape: tuple[int, int, int] = (1, 1, 1),
    *,
    scale: float = 0.02,
) -> jnp.ndarray:
    """Return deterministic Higgs electroweak link momenta."""

    x, y, z, edge, gen = jnp.indices((*lattice_shape, 8, SM_HIGGS_GENERATOR_COUNT), dtype=jnp.float32)
    return scale * (
        0.021 * (x + 1.0)
        - 0.013 * (y + 1.0)
        + 0.017 * (z + 1.0)
        + 0.009 * (edge + 1.0)
        - 0.006 * (gen + 1.0)
    )


def sm_gauge_higgs_backreaction_diagnostics() -> GaugeHiggsBackreactionDiagnostics:
    """Return focused Stage 9 gauge-Higgs backreaction diagnostics."""

    lattice_shape = (1, 1, 1)
    parameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS
    field = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    link_momenta = deterministic_higgs_link_momenta(lattice_shape)
    site_gauge = sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape, scale=0.17))
    transformed_field = sm_transform_higgs_field(field, site_gauge)
    transformed_higgs_momenta = sm_transform_higgs_field(higgs_momenta, site_gauge)
    transformed_links = sm_transform_higgs_links(links, site_gauge)
    transformed_link_momenta = sm_transform_higgs_link_momenta(link_momenta, site_gauge)

    momentum_matrices = sm_higgs_link_momentum_algebra(link_momenta)
    projection_roundtrip = sm_higgs_project_to_coordinates(momentum_matrices)
    force = sm_higgs_left_gauge_force(field, links, parameters=parameters)
    transformed_force = sm_higgs_left_gauge_force(transformed_field, transformed_links, parameters=parameters)
    expected_force = sm_transform_higgs_link_momenta(force, site_gauge)
    charge = sm_higgs_charge_density(field, higgs_momenta)
    transformed_charge = sm_higgs_charge_density(transformed_field, transformed_higgs_momenta)
    expected_charge = sm_higgs_project_to_coordinates(
        site_gauge
        @ sm_higgs_algebra_matrix_field(charge)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )
    gauss = sm_higgs_gauss_constraint(links, link_momenta, field, higgs_momenta)
    transformed_gauss = sm_higgs_gauss_constraint(
        transformed_links,
        transformed_link_momenta,
        transformed_field,
        transformed_higgs_momenta,
    )
    expected_gauss = sm_higgs_project_to_coordinates(
        site_gauge
        @ sm_higgs_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )

    vacuum = sm_constant_higgs(lattice_shape, vev=parameters.vev)
    zero_field_momenta = jnp.zeros_like(vacuum)
    zero_link_momenta = jnp.zeros((*lattice_shape, 8, SM_HIGGS_GENERATOR_COUNT), dtype=jnp.float32)
    identity_links = sm_identity_higgs_links(lattice_shape)
    pure_links = sm_pure_gauge_higgs_links_from_site_algebra(deterministic_higgs_site_theta(lattice_shape))
    pure_field = sm_transform_higgs_field(vacuum, sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape)))

    step_size = 0.005
    hamiltonian_before = sm_coupled_higgs_gauge_hamiltonian_density(
        field,
        higgs_momenta,
        links,
        link_momenta,
        parameters=parameters,
    )
    updated_field, updated_higgs_momenta, updated_links, updated_link_momenta = sm_coupled_higgs_gauge_leapfrog_step(
        field,
        higgs_momenta,
        links,
        link_momenta,
        step_size=step_size,
        parameters=parameters,
    )
    hamiltonian_after = sm_coupled_higgs_gauge_hamiltonian_density(
        updated_field,
        updated_higgs_momenta,
        updated_links,
        updated_link_momenta,
        parameters=parameters,
    )
    reversed_field, reversed_higgs_momenta, reversed_links, reversed_link_momenta = (
        sm_coupled_higgs_gauge_leapfrog_step(
            updated_field,
            -updated_higgs_momenta,
            updated_links,
            -updated_link_momenta,
            step_size=step_size,
            parameters=parameters,
        )
    )
    jitted_step = jax.jit(
        sm_coupled_higgs_gauge_leapfrog_step,
        static_argnames=("step_size", "parameters", "force_epsilon"),
    )
    jit_field, jit_higgs_momenta, jit_links, jit_link_momenta = jitted_step(
        field,
        higgs_momenta,
        links,
        link_momenta,
        step_size=step_size,
        parameters=parameters,
    )

    return GaugeHiggsBackreactionDiagnostics(
        projection_roundtrip_residual=jnp.max(jnp.abs(projection_roundtrip - link_momenta)),
        higgs_link_force_vacuum_norm=jnp.linalg.norm(
            sm_higgs_left_gauge_force(pure_field, pure_links, parameters=parameters),
        ),
        higgs_link_force_nonzero_norm=jnp.linalg.norm(force),
        higgs_link_force_covariance_residual=jnp.max(jnp.abs(transformed_force - expected_force)),
        charge_covariance_residual=jnp.max(jnp.abs(transformed_charge - expected_charge)),
        gauss_covariance_residual=jnp.max(jnp.abs(transformed_gauss - expected_gauss)),
        gauss_vacuum_residual=jnp.linalg.norm(
            sm_higgs_gauss_constraint(identity_links, zero_link_momenta, vacuum, zero_field_momenta),
        ),
        sm_embedding_roundtrip_residual=jnp.max(
            jnp.abs(sm_extract_higgs_from_sm_momenta(sm_embed_higgs_to_sm_momenta(link_momenta)) - link_momenta),
        ),
        coupled_link_unitarity_residual=sm_higgs_link_unitarity_residual(updated_links),
        coupled_hamiltonian_drift=jnp.abs(hamiltonian_after - hamiltonian_before),
        coupled_reversibility_residual=jnp.maximum(
            jnp.max(jnp.abs(reversed_field - field)),
            jnp.maximum(
                jnp.max(jnp.abs(reversed_higgs_momenta + higgs_momenta)),
                jnp.maximum(
                    jnp.max(jnp.abs(reversed_links - links)),
                    jnp.max(jnp.abs(reversed_link_momenta + link_momenta)),
                ),
            ),
        ),
        jit_delta_field=jnp.max(jnp.abs(jit_field - updated_field)),
        jit_delta_links=jnp.max(jnp.abs(jit_links - updated_links)),
        jit_delta_higgs_momenta=jnp.max(jnp.abs(jit_higgs_momenta - updated_higgs_momenta)),
        jit_delta_link_momenta=jnp.max(jnp.abs(jit_link_momenta - updated_link_momenta)),
    )
