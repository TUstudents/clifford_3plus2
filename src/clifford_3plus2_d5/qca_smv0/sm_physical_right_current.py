"""Physical-right fermion current for QCA_SMv0.

Stage 19 lifts the Stage 13 family streaming current to the Stage 18
physical-right bridged transport carrier.  The link coordinates remain the
same 12 transport-coordinate labels; the matter representation is changed by
bridging every finite link and generator to the physical-right carrier.

This stage builds current/charge/Gauss diagnostics and a reversible momentum
kick.  It does not yet replace the sourced or production ticks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS, bcc_dirac_hop_matrices
from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import (
    sm_antiunitary_bridge_gauge_from_transport,
    sm_physical_right_generators,
)
from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta, sm_apply_momentum_update
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import (
    sm_family_fermion_left_gauge_current,
    sm_transform_family_gauge_state,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import SM_FAMILY_DIM, deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    sm_generators,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import (
    sm_family_physical_right_gauged_dirac_step,
    sm_physical_right_links_from_transport,
)
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightCurrentDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 19 physical-right current."""

    streaming_energy_reality_residual: jnp.ndarray
    zero_state_current_norm: jnp.ndarray
    nonzero_current_norm: jnp.ndarray
    transport_current_difference_norm: jnp.ndarray
    streaming_energy_covariance_residual: jnp.ndarray
    current_covariance_residual: jnp.ndarray
    charge_covariance_residual: jnp.ndarray
    gauss_covariance_residual: jnp.ndarray
    momentum_kick_delta_norm: jnp.ndarray
    momentum_kick_reversibility_residual: jnp.ndarray
    kicked_link_unitarity_residual: jnp.ndarray
    spectator_norm_drift_after_kick: jnp.ndarray
    jit_delta_current: jnp.ndarray
    jit_delta_transport: jnp.ndarray


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, 3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_sm_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("SM links must have shape (nx, ny, nz, 8, 32, 32)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("SM links must match the lattice")


def _validate_sm_momenta(momenta: jnp.ndarray, links: jnp.ndarray | None = None) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, SM_GENERATOR_COUNT):
        raise ValueError("SM momenta must have shape (nx, ny, nz, 8, 12)")
    if links is not None and momenta.shape[:4] != links.shape[:4]:
        raise ValueError("SM momenta and links must share shape (nx, ny, nz, 8)")


def _minus_displacement_indices() -> tuple[int, ...]:
    lookup = {tuple(map(int, displacement)): index for index, displacement in enumerate(np.asarray(BCC_DISPLACEMENTS))}
    return tuple(lookup[tuple(map(int, -displacement))] for displacement in np.asarray(BCC_DISPLACEMENTS))


def _force_coordinate_indices(links: jnp.ndarray) -> np.ndarray:
    return np.asarray(list(np.ndindex(*links.shape[:4], SM_GENERATOR_COUNT)), dtype=np.int32)


def sm_physical_right_generator_gram() -> jnp.ndarray:
    """Return the real Hilbert-Schmidt Gram matrix of physical-right generators."""

    generators = sm_physical_right_generators()
    daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    return jnp.real(jnp.einsum("aij,bji->ab", daggers, generators))


def sm_physical_right_project_to_coordinates(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project physical-right algebra matrices onto the shared 12 coordinates."""

    if matrix.shape[-2:] != (SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("matrix must have trailing shape (32, 32)")
    generators = sm_physical_right_generators(dtype=jnp.result_type(matrix, 1j))
    daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    gram = sm_physical_right_generator_gram().astype(jnp.real(matrix).dtype)
    rhs = jnp.real(jnp.einsum("aij,...ji->...a", daggers, matrix.astype(generators.dtype)))
    flat_rhs = rhs.reshape((-1, SM_GENERATOR_COUNT))
    flat_coordinates = jnp.linalg.solve(gram, flat_rhs.T).T
    return flat_coordinates.reshape(rhs.shape)


def _sm_physical_right_algebra_matrix_field_any(theta: jnp.ndarray) -> jnp.ndarray:
    if theta.shape[-1] != SM_GENERATOR_COUNT:
        raise ValueError("theta must have trailing coordinate dimension 12")
    generators = sm_physical_right_generators(dtype=jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def sm_physical_right_transform_momenta(momenta: jnp.ndarray, transport_site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site physical-right adjoint transforms to momenta."""

    _validate_sm_momenta(momenta)
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_site_gauge)
    momentum_matrices = _sm_physical_right_algebra_matrix_field_any(momenta)
    gauge_dagger = jnp.swapaxes(jnp.conj(physical_site_gauge), -1, -2)
    transformed = jnp.einsum("...ab,...hbc,...cd->...had", physical_site_gauge, momentum_matrices, gauge_dagger)
    return sm_physical_right_project_to_coordinates(transformed)


def sm_physical_right_streaming_local_density(state: jnp.ndarray, transport_links: jnp.ndarray) -> jnp.ndarray:
    """Return local physical-right family streaming bilinear density."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(transport_links, lattice_shape)
    physical_links = sm_physical_right_links_from_transport(transport_links)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    local = jnp.zeros(lattice_shape, dtype=state.dtype)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        linked = jnp.einsum("...ab,...sbf->...saf", physical_links[..., index, :, :], source)
        hopped = jnp.einsum("rs,...sdf->...rdf", hops[index], linked)
        local = local + jnp.sum(jnp.conj(state) * hopped, axis=(-3, -2, -1))
    return jnp.real(local)


def sm_physical_right_streaming_energy_density(state: jnp.ndarray, transport_links: jnp.ndarray) -> jnp.ndarray:
    """Return mean physical-right family streaming bilinear density."""

    return jnp.mean(sm_physical_right_streaming_local_density(state, transport_links))


def _generator_left_updates(links: jnp.ndarray, epsilon: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    generators = sm_generators(dtype=links.dtype)
    plus = jax.vmap(jsp_linalg.expm)(step * generators)
    minus = jax.vmap(jsp_linalg.expm)(-step * generators)
    return plus, minus


def sm_physical_right_fermion_left_gauge_current(
    state: jnp.ndarray,
    transport_links: jnp.ndarray,
    *,
    epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Return physical-right family current in the shared 12 coordinates."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    lattice_shape = _validate_family_state(state)
    _validate_sm_links(transport_links, lattice_shape)
    real_dtype = jnp.real(jnp.asarray(0, dtype=transport_links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    plus_updates, minus_updates = _generator_left_updates(transport_links, epsilon)
    coordinate_indices = jnp.asarray(_force_coordinate_indices(transport_links))

    def energy_for_index(index: jnp.ndarray, updates: jnp.ndarray) -> jnp.ndarray:
        update = updates[index[4]]
        updated_link = update @ transport_links[index[0], index[1], index[2], index[3]]
        perturbed_links = transport_links.at[index[0], index[1], index[2], index[3]].set(updated_link)
        return sm_physical_right_streaming_energy_density(state, perturbed_links)

    batched_energy = jax.vmap(energy_for_index, in_axes=(0, None))
    derivatives = (batched_energy(coordinate_indices, plus_updates) - batched_energy(coordinate_indices, minus_updates)) / (
        2 * step
    )
    current = jnp.zeros((*transport_links.shape[:4], SM_GENERATOR_COUNT), dtype=real_dtype)
    current = current.at[
        coordinate_indices[:, 0],
        coordinate_indices[:, 1],
        coordinate_indices[:, 2],
        coordinate_indices[:, 3],
        coordinate_indices[:, 4],
    ].set(derivatives)
    return current


def sm_physical_right_streaming_fermion_charge_density(state: jnp.ndarray) -> jnp.ndarray:
    """Return local physical-right family gauge charge density."""

    _validate_family_state(state)
    generators = sm_physical_right_generators(dtype=state.dtype)
    return jnp.imag(jnp.einsum("...sif,aij,...sjf->...a", jnp.conj(state), generators, state))


def sm_physical_right_electric_divergence(transport_links: jnp.ndarray, momenta: jnp.ndarray) -> jnp.ndarray:
    """Return target-site physical-right electric divergence."""

    _validate_sm_momenta(momenta, transport_links)
    physical_links = sm_physical_right_links_from_transport(transport_links)
    momentum_matrices = _sm_physical_right_algebra_matrix_field_any(momenta)
    divergence = jnp.zeros((*transport_links.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=transport_links.dtype)
    minus_indices = _minus_displacement_indices()
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        outgoing_index = minus_indices[index]
        incoming = momentum_matrices[..., index, :, :]
        outgoing_momentum = source_roll(momentum_matrices[..., outgoing_index, :, :], displacement)
        outgoing_link = source_roll(physical_links[..., outgoing_index, :, :], displacement)
        outgoing_link_dagger = jnp.swapaxes(jnp.conj(outgoing_link), -1, -2)
        transported_outgoing = outgoing_link_dagger @ outgoing_momentum @ outgoing_link
        divergence = divergence + incoming - transported_outgoing
    return sm_physical_right_project_to_coordinates(divergence)


def sm_physical_right_fermion_gauss_constraint(
    transport_links: jnp.ndarray,
    momenta: jnp.ndarray,
    state: jnp.ndarray,
) -> jnp.ndarray:
    """Return physical-right electric divergence minus family charge."""

    return sm_physical_right_electric_divergence(transport_links, momenta) - sm_physical_right_streaming_fermion_charge_density(
        state,
    )


def sm_apply_physical_right_fermion_gauge_momentum_kick(
    momenta: jnp.ndarray,
    state: jnp.ndarray,
    transport_links: jnp.ndarray,
    *,
    step_size: float,
    epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Kick transport-coordinate momenta by physical-right family current."""

    _validate_sm_momenta(momenta, transport_links)
    current = sm_physical_right_fermion_left_gauge_current(state, transport_links, epsilon=epsilon)
    return momenta - jnp.asarray(step_size, dtype=momenta.dtype) * current


def sm_physical_right_fermion_gauge_kick_then_transport(
    state: jnp.ndarray,
    transport_links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    epsilon: float = 3e-2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Kick physical-right current, update transport links, then bridge-transport."""

    kicked_momenta = sm_apply_physical_right_fermion_gauge_momentum_kick(
        momenta,
        state,
        transport_links,
        step_size=step_size,
        epsilon=epsilon,
    )
    updated_transport_links = sm_apply_momentum_update(transport_links, kicked_momenta, step_size=step_size)
    updated_state = sm_family_physical_right_gauged_dirac_step(state, updated_transport_links)
    return updated_state, updated_transport_links, kicked_momenta


def sm_physical_right_current_diagnostics() -> PhysicalRightCurrentDiagnostics:
    """Return focused Stage 19 physical-right current diagnostics."""

    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    zero_state = jnp.zeros_like(state)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.24))
    momenta = deterministic_sm_momenta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape, scale=0.07))
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(site_gauge)
    transformed_state = sm_transform_family_gauge_state(state, physical_site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_momenta = sm_physical_right_transform_momenta(momenta, site_gauge)

    energy = sm_physical_right_streaming_energy_density(state, links)
    transformed_energy = sm_physical_right_streaming_energy_density(transformed_state, transformed_links)
    current = sm_physical_right_fermion_left_gauge_current(state, links)
    transformed_current = sm_physical_right_fermion_left_gauge_current(transformed_state, transformed_links)
    expected_current = sm_physical_right_transform_momenta(current, site_gauge)
    transport_current = sm_family_fermion_left_gauge_current(state, links)
    charge = sm_physical_right_streaming_fermion_charge_density(state)
    transformed_charge = sm_physical_right_streaming_fermion_charge_density(transformed_state)
    expected_charge = sm_physical_right_project_to_coordinates(
        physical_site_gauge
        @ _sm_physical_right_algebra_matrix_field_any(charge)
        @ jnp.swapaxes(jnp.conj(physical_site_gauge), -1, -2),
    )
    gauss = sm_physical_right_fermion_gauss_constraint(links, momenta, state)
    transformed_gauss = sm_physical_right_fermion_gauss_constraint(transformed_links, transformed_momenta, transformed_state)
    expected_gauss = sm_physical_right_project_to_coordinates(
        physical_site_gauge
        @ _sm_physical_right_algebra_matrix_field_any(gauss)
        @ jnp.swapaxes(jnp.conj(physical_site_gauge), -1, -2),
    )

    step_size = 0.004
    kicked = sm_apply_physical_right_fermion_gauge_momentum_kick(momenta, state, links, step_size=step_size)
    restored = sm_apply_physical_right_fermion_gauge_momentum_kick(kicked, state, links, step_size=-step_size)
    updated_state, updated_links, _ = sm_physical_right_fermion_gauge_kick_then_transport(
        state,
        links,
        momenta,
        step_size=step_size,
    )
    jitted_current = jax.jit(sm_physical_right_fermion_left_gauge_current)
    jit_current = jitted_current(state, links)
    jitted_transport = jax.jit(sm_family_physical_right_gauged_dirac_step)
    jit_transport = jitted_transport(state, links)
    transport = sm_family_physical_right_gauged_dirac_step(state, links)

    return PhysicalRightCurrentDiagnostics(
        streaming_energy_reality_residual=jnp.abs(jnp.imag(energy + 0j)),
        zero_state_current_norm=jnp.linalg.norm(sm_physical_right_fermion_left_gauge_current(zero_state, links)),
        nonzero_current_norm=jnp.linalg.norm(current),
        transport_current_difference_norm=jnp.linalg.norm(current - transport_current),
        streaming_energy_covariance_residual=jnp.abs(transformed_energy - energy),
        current_covariance_residual=jnp.max(jnp.abs(transformed_current - expected_current)),
        charge_covariance_residual=jnp.max(jnp.abs(transformed_charge - expected_charge)),
        gauss_covariance_residual=jnp.max(jnp.abs(transformed_gauss - expected_gauss)),
        momentum_kick_delta_norm=jnp.linalg.norm(kicked - momenta),
        momentum_kick_reversibility_residual=jnp.max(jnp.abs(restored - momenta)),
        kicked_link_unitarity_residual=sm_link_unitarity_residual(updated_links),
        spectator_norm_drift_after_kick=jnp.abs(state_norm_squared(updated_state) - state_norm_squared(state)),
        jit_delta_current=jnp.max(jnp.abs(jit_current - current)),
        jit_delta_transport=jnp.max(jnp.abs(jit_transport - transport)),
    )
