"""Family-summed BCC streaming fermion gauge current for QCA_SMv0.

Stage 13 lifts the Stage 11 one-carrier streaming current to the Stage 7
family state layout ``(nx, ny, nz, 4, 32, 3)``.  The gauge links remain
family-blind; the link current and local charge are sums over the family axis.
This is the missing bridge between the Stage 12 sourced gauge tick and the
three-family FN/Yukawa carrier.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS, bcc_dirac_hop_matrices
from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    deterministic_sm_momenta,
    sm_apply_momentum_update,
    sm_electric_divergence,
    sm_project_to_coordinates,
    sm_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    SM_FAMILY_DIM,
    deterministic_sm_family_state,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_gauge import (
    sm_dirac_streaming_energy_density,
    sm_fermion_left_gauge_current,
    sm_streaming_fermion_charge_density,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    deterministic_sm_state,
    sm_algebra_matrix_field,
    sm_generators,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.state import state_norm_squared


class FamilyFermionGaugeCurrentDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 13 family-summed fermion gauge current."""

    streaming_energy_reality_residual: jnp.ndarray
    single_family_energy_reduction_residual: jnp.ndarray
    single_family_current_reduction_residual: jnp.ndarray
    single_family_charge_reduction_residual: jnp.ndarray
    zero_state_current_norm: jnp.ndarray
    nonzero_current_norm: jnp.ndarray
    streaming_energy_gauge_invariance_residual: jnp.ndarray
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


def _validate_sm_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 5 or state.shape[-2:] != (4, SM_INTERNAL_DIM):
        raise ValueError("SM Dirac state must have shape (nx, ny, nz, 4, 32)")
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


def sm_single_family_state(family_state: jnp.ndarray, family: int = 0) -> jnp.ndarray:
    """Return one family slice as a Stage 11 state."""

    _validate_family_state(family_state)
    if family < 0 or family >= SM_FAMILY_DIM:
        raise ValueError(f"family must be in [0,{SM_FAMILY_DIM})")
    return family_state[..., family]


def sm_family_state_from_single(state: jnp.ndarray, family: int = 0) -> jnp.ndarray:
    """Embed one Stage 11 state into the family register."""

    lattice_shape = _validate_sm_state(state)
    if family < 0 or family >= SM_FAMILY_DIM:
        raise ValueError(f"family must be in [0,{SM_FAMILY_DIM})")
    family_state = jnp.zeros((*lattice_shape, 4, SM_INTERNAL_DIM, SM_FAMILY_DIM), dtype=state.dtype)
    return family_state.at[..., family].set(state)


def sm_transform_family_gauge_state(state: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply a family-blind SM site gauge to a family state."""

    lattice_shape = _validate_family_state(state)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != lattice_shape or site_gauge.shape[-2:] != (
        SM_INTERNAL_DIM,
        SM_INTERNAL_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 32, 32)")
    return jnp.einsum("...ab,...sbf->...saf", site_gauge, state)


def sm_family_gauged_dirac_step(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Apply the family-blind gauged BCC Dirac step to every family."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(links, lattice_shape)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        linked = jnp.einsum("...ab,...sbf->...saf", links[..., index, :, :], source)
        out = out + jnp.einsum("rs,...sdf->...rdf", hops[index], linked)
    return out


def sm_family_streaming_local_density(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return local family-summed BCC streaming bilinear density."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(links, lattice_shape)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    local = jnp.zeros(lattice_shape, dtype=state.dtype)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        linked = jnp.einsum("...ab,...sbf->...saf", links[..., index, :, :], source)
        hopped = jnp.einsum("rs,...sdf->...rdf", hops[index], linked)
        local = local + jnp.sum(jnp.conj(state) * hopped, axis=(-3, -2, -1))
    return jnp.real(local)


def sm_family_streaming_energy_density(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return mean family-summed BCC streaming bilinear density."""

    return jnp.mean(sm_family_streaming_local_density(state, links))


def _generator_left_updates(links: jnp.ndarray, epsilon: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    generators = sm_generators(dtype=links.dtype)
    plus = jax.vmap(jsp_linalg.expm)(step * generators)
    minus = jax.vmap(jsp_linalg.expm)(-step * generators)
    return plus, minus


def _force_coordinate_indices(links: jnp.ndarray) -> np.ndarray:
    return np.asarray(list(np.ndindex(*links.shape[:4], SM_GENERATOR_COUNT)), dtype=np.int32)


def sm_family_fermion_left_gauge_current(
    state: jnp.ndarray,
    links: jnp.ndarray,
    *,
    epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Return the family-summed fermion current on SM BCC links."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    lattice_shape = _validate_family_state(state)
    _validate_sm_links(links, lattice_shape)
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    plus_updates, minus_updates = _generator_left_updates(links, epsilon)
    coordinate_indices = jnp.asarray(_force_coordinate_indices(links))

    def energy_for_index(index: jnp.ndarray, updates: jnp.ndarray) -> jnp.ndarray:
        update = updates[index[4]]
        updated_link = update @ links[index[0], index[1], index[2], index[3]]
        perturbed_links = links.at[index[0], index[1], index[2], index[3]].set(updated_link)
        return sm_family_streaming_energy_density(state, perturbed_links)

    batched_energy = jax.vmap(energy_for_index, in_axes=(0, None))
    derivatives = (batched_energy(coordinate_indices, plus_updates) - batched_energy(coordinate_indices, minus_updates)) / (
        2 * step
    )
    current = jnp.zeros((*links.shape[:4], SM_GENERATOR_COUNT), dtype=real_dtype)
    current = current.at[
        coordinate_indices[:, 0],
        coordinate_indices[:, 1],
        coordinate_indices[:, 2],
        coordinate_indices[:, 3],
        coordinate_indices[:, 4],
    ].set(derivatives)
    return current


def sm_family_streaming_fermion_charge_density(state: jnp.ndarray) -> jnp.ndarray:
    """Return local family-summed gauge charge density."""

    _validate_family_state(state)
    generators = sm_generators(dtype=state.dtype)
    return jnp.imag(jnp.einsum("...sif,aij,...sjf->...a", jnp.conj(state), generators, state))


def sm_family_fermion_gauss_constraint(links: jnp.ndarray, momenta: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    """Return electric divergence minus family-summed fermion charge density."""

    return sm_electric_divergence(links, momenta) - sm_family_streaming_fermion_charge_density(state)


def sm_apply_family_fermion_gauge_momentum_kick(
    momenta: jnp.ndarray,
    state: jnp.ndarray,
    links: jnp.ndarray,
    *,
    step_size: float,
    epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Kick SM link momenta by the family-summed BCC streaming current."""

    _validate_sm_momenta(momenta, links)
    current = sm_family_fermion_left_gauge_current(state, links, epsilon=epsilon)
    return momenta - jnp.asarray(step_size, dtype=momenta.dtype) * current


def sm_family_fermion_gauge_kick_then_transport(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    epsilon: float = 3e-2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Kick momenta by the family current, update links, then transport state."""

    kicked_momenta = sm_apply_family_fermion_gauge_momentum_kick(
        momenta,
        state,
        links,
        step_size=step_size,
        epsilon=epsilon,
    )
    updated_links = sm_apply_momentum_update(links, kicked_momenta, step_size=step_size)
    updated_state = sm_family_gauged_dirac_step(state, updated_links)
    return updated_state, updated_links, kicked_momenta


def sm_family_fermion_gauge_current_diagnostics() -> FamilyFermionGaugeCurrentDiagnostics:
    """Return focused Stage 13 family-summed current diagnostics."""

    lattice_shape = (1, 1, 1)
    one_state = deterministic_sm_state(lattice_shape)
    single_family_state = sm_family_state_from_single(one_state, family=1)
    family_state = deterministic_sm_family_state(lattice_shape)
    zero_state = jnp.zeros_like(family_state)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    momenta = deterministic_sm_momenta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape, scale=0.06))
    transformed_state = sm_transform_family_gauge_state(family_state, site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_momenta = sm_transform_momenta(momenta, site_gauge)

    energy = sm_family_streaming_energy_density(family_state, links)
    transformed_energy = sm_family_streaming_energy_density(transformed_state, transformed_links)
    current = sm_family_fermion_left_gauge_current(family_state, links)
    transformed_current = sm_family_fermion_left_gauge_current(transformed_state, transformed_links)
    expected_current = sm_transform_momenta(current, site_gauge)
    charge = sm_family_streaming_fermion_charge_density(family_state)
    transformed_charge = sm_family_streaming_fermion_charge_density(transformed_state)
    expected_charge = sm_project_to_coordinates(
        site_gauge @ sm_algebra_matrix_field(charge) @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )
    gauss = sm_family_fermion_gauss_constraint(links, momenta, family_state)
    transformed_gauss = sm_family_fermion_gauss_constraint(transformed_links, transformed_momenta, transformed_state)
    expected_gauss = sm_project_to_coordinates(
        site_gauge @ sm_algebra_matrix_field(gauss) @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )
    single_energy = sm_family_streaming_energy_density(single_family_state, links)
    single_current = sm_family_fermion_left_gauge_current(single_family_state, links)
    single_charge = sm_family_streaming_fermion_charge_density(single_family_state)

    step_size = 0.004
    kicked = sm_apply_family_fermion_gauge_momentum_kick(momenta, family_state, links, step_size=step_size)
    restored = sm_apply_family_fermion_gauge_momentum_kick(kicked, family_state, links, step_size=-step_size)
    updated_state, updated_links, _ = sm_family_fermion_gauge_kick_then_transport(
        family_state,
        links,
        momenta,
        step_size=step_size,
    )
    jitted_current = jax.jit(sm_family_fermion_left_gauge_current)
    jit_current = jitted_current(family_state, links)
    jitted_transport = jax.jit(sm_family_gauged_dirac_step)
    jit_transport = jitted_transport(family_state, links)
    transport = sm_family_gauged_dirac_step(family_state, links)

    return FamilyFermionGaugeCurrentDiagnostics(
        streaming_energy_reality_residual=jnp.abs(jnp.imag(energy + 0j)),
        single_family_energy_reduction_residual=jnp.abs(single_energy - sm_dirac_streaming_energy_density(one_state, links)),
        single_family_current_reduction_residual=jnp.max(
            jnp.abs(single_current - sm_fermion_left_gauge_current(one_state, links)),
        ),
        single_family_charge_reduction_residual=jnp.max(
            jnp.abs(single_charge - sm_streaming_fermion_charge_density(one_state)),
        ),
        zero_state_current_norm=jnp.linalg.norm(sm_family_fermion_left_gauge_current(zero_state, links)),
        nonzero_current_norm=jnp.linalg.norm(current),
        streaming_energy_gauge_invariance_residual=jnp.abs(transformed_energy - energy),
        current_covariance_residual=jnp.max(jnp.abs(transformed_current - expected_current)),
        charge_covariance_residual=jnp.max(jnp.abs(transformed_charge - expected_charge)),
        gauss_covariance_residual=jnp.max(jnp.abs(transformed_gauss - expected_gauss)),
        momentum_kick_delta_norm=jnp.linalg.norm(kicked - momenta),
        momentum_kick_reversibility_residual=jnp.max(jnp.abs(restored - momenta)),
        kicked_link_unitarity_residual=sm_link_unitarity_residual(updated_links),
        spectator_norm_drift_after_kick=jnp.abs(state_norm_squared(updated_state) - state_norm_squared(family_state)),
        jit_delta_current=jnp.max(jnp.abs(jit_current - current)),
        jit_delta_transport=jnp.max(jnp.abs(jit_transport - transport)),
    )
