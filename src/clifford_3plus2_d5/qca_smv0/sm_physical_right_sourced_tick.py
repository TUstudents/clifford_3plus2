"""Physical-right sourced gauge tick for QCA_SMv0.

Stage 20 replaces the Stage 14 transport-convention family fermion source with
the Stage 19 physical-right current, charge, Gauss, and bridged transport.  The
pure Wilson force and Higgs gauge force remain the same transport-coordinate
objects; all sources are still accumulated in the shared 12 SM link-momentum
coordinates.

This is the bridged gauge-source tick.  It does not merge the local Yukawa
production collision into the physical-right carrier; that remains a later
production-tick rewrite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import (
    sm_antiunitary_bridge_gauge_from_transport,
    sm_physical_right_algebra_matrix_field,
)
from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta, sm_left_wilson_force
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import sm_transform_family_gauge_state
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import SM_FAMILY_DIM, deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_family_sourced_tick import (
    sm_family_sourced_link_force,
    sm_family_sourced_sm_tick,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_site_theta_from_higgs_site_theta
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge_higgs import (
    sm_embed_higgs_to_sm_momenta,
    sm_higgs_charge_density,
    sm_higgs_left_gauge_force,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM, sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    deterministic_higgs_theta,
    sm_higgs_force,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
    sm_higgs_site_gauge_from_algebra,
    sm_identity_higgs_links,
    sm_transform_higgs_field,
    sm_transform_higgs_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_physical_right_electric_divergence,
    sm_physical_right_fermion_left_gauge_current,
    sm_physical_right_project_to_coordinates,
    sm_physical_right_streaming_fermion_charge_density,
    sm_physical_right_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import sm_family_physical_right_gauged_dirac_step
from clifford_3plus2_d5.qca_smv0.sm_sourced_tick import (
    sm_apply_sourced_link_update,
    sm_embed_higgs_site_to_sm_coordinates,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightSourcedSMTickDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 20 physical-right sourced gauge tick."""

    transport_force_difference_norm: jnp.ndarray
    zero_source_force_norm: jnp.ndarray
    nonzero_source_force_norm: jnp.ndarray
    force_covariance_residual: jnp.ndarray
    gauss_covariance_residual: jnp.ndarray
    gauss_zero_residual: jnp.ndarray
    kick_delta_norm: jnp.ndarray
    kick_reversibility_residual: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    family_norm_drift: jnp.ndarray
    higgs_field_delta_norm: jnp.ndarray
    transport_tick_state_difference_norm: jnp.ndarray
    jit_delta_family_state: jnp.ndarray
    jit_delta_sm_links: jnp.ndarray
    jit_delta_higgs_links: jnp.ndarray
    jit_delta_sm_momenta: jnp.ndarray
    jit_delta_higgs_field: jnp.ndarray
    jit_delta_higgs_momenta: jnp.ndarray


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


def _validate_higgs_field(field: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if field.ndim != 4 or field.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and field.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the lattice")


def _validate_higgs_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, SM_HIGGS_DIM, SM_HIGGS_DIM):
        raise ValueError("Higgs links must have shape (nx, ny, nz, 8, 2, 2)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("Higgs links must match the lattice")


def _validate_higgs_momenta(momenta: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if momenta.ndim != 4 or momenta.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs momenta must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and momenta.shape[:3] != lattice_shape:
        raise ValueError("Higgs momenta must match the lattice")


def sm_physical_right_sourced_link_force(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    sm_links: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Return Wilson + Higgs + physical-right family fermion force."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)
    wilson = jnp.asarray(beta, dtype=jnp.real(sm_links).dtype) * sm_left_wilson_force(
        sm_links,
        epsilon=wilson_epsilon,
    )
    higgs_force = sm_embed_higgs_to_sm_momenta(
        sm_higgs_left_gauge_force(
            higgs,
            higgs_links,
            parameters=parameters,
            epsilon=higgs_force_epsilon,
        ),
    )
    fermion = sm_physical_right_fermion_left_gauge_current(
        state,
        sm_links,
        epsilon=fermion_current_epsilon,
    )
    return wilson + higgs_force + fermion


def sm_apply_physical_right_sourced_sm_momentum_kick(
    momenta: jnp.ndarray,
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    sm_links: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> jnp.ndarray:
    """Kick SM link momenta by the Stage 20 physical-right gauge sources."""

    _validate_sm_momenta(momenta, sm_links)
    force = sm_physical_right_sourced_link_force(
        state,
        higgs,
        sm_links,
        higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    return momenta - jnp.asarray(step_size, dtype=momenta.dtype) * force


def sm_physical_right_sourced_gauss_constraint(
    sm_links: jnp.ndarray,
    sm_momenta: jnp.ndarray,
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
) -> jnp.ndarray:
    """Return physical-right electric divergence minus fermion and Higgs charges."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_sm_momenta(sm_momenta, sm_links)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    return (
        sm_physical_right_electric_divergence(sm_links, sm_momenta)
        - sm_physical_right_streaming_fermion_charge_density(state)
        - sm_embed_higgs_site_to_sm_coordinates(sm_higgs_charge_density(higgs, higgs_momenta))
    )


def sm_physical_right_sourced_sm_tick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    sm_links: jnp.ndarray,
    sm_momenta: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance one physical-right sourced SM tick.

    Return ``(state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links)``.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_sm_momenta(sm_momenta, sm_links)
    _validate_higgs_links(higgs_links, lattice_shape)
    dt = jnp.asarray(step_size, dtype=sm_momenta.dtype)

    first_link_force = sm_physical_right_sourced_link_force(
        state,
        higgs,
        sm_links,
        higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    first_higgs_force = sm_higgs_force(higgs, higgs_links, parameters=parameters)
    half_sm_momenta = sm_momenta - 0.5 * dt * first_link_force
    half_higgs_momenta = higgs_momenta + 0.5 * dt * first_higgs_force

    updated_sm_links, updated_higgs_links = sm_apply_sourced_link_update(
        sm_links,
        higgs_links,
        half_sm_momenta,
        step_size=step_size,
    )
    updated_higgs = higgs + dt * half_higgs_momenta
    updated_state = sm_family_physical_right_gauged_dirac_step(state, updated_sm_links)

    second_link_force = sm_physical_right_sourced_link_force(
        updated_state,
        updated_higgs,
        updated_sm_links,
        updated_higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    second_higgs_force = sm_higgs_force(updated_higgs, updated_higgs_links, parameters=parameters)
    updated_sm_momenta = half_sm_momenta - 0.5 * dt * second_link_force
    updated_higgs_momenta = half_higgs_momenta + 0.5 * dt * second_higgs_force
    return (
        updated_state,
        updated_higgs,
        updated_higgs_momenta,
        updated_sm_links,
        updated_sm_momenta,
        updated_higgs_links,
    )


def sm_physical_right_sourced_tick_diagnostics() -> PhysicalRightSourcedSMTickDiagnostics:
    """Return focused Stage 20 physical-right sourced tick diagnostics."""

    lattice_shape = (1, 1, 1)
    family_state = deterministic_sm_family_state(lattice_shape)
    zero_state = jnp.zeros_like(family_state)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    vacuum = sm_constant_higgs(lattice_shape)
    zero_higgs_momenta = jnp.zeros_like(higgs_momenta)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    identity_sm_links = sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32))
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    identity_higgs_links = sm_identity_higgs_links(lattice_shape)
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    zero_sm_momenta = jnp.zeros_like(sm_momenta)

    higgs_site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.08)
    higgs_site_gauge = sm_higgs_site_gauge_from_algebra(higgs_site_theta)
    sm_site_gauge = sm_site_gauge_from_algebra(sm_site_theta_from_higgs_site_theta(higgs_site_theta))
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(sm_site_gauge)
    transformed_state = sm_transform_family_gauge_state(family_state, physical_site_gauge)
    transformed_higgs = sm_transform_higgs_field(higgs, higgs_site_gauge)
    transformed_higgs_momenta = sm_transform_higgs_field(higgs_momenta, higgs_site_gauge)
    transformed_sm_links = sm_transform_links(sm_links, sm_site_gauge)
    transformed_higgs_links = sm_transform_higgs_links(higgs_links, higgs_site_gauge)
    transformed_sm_momenta = sm_physical_right_transform_momenta(sm_momenta, sm_site_gauge)

    force = sm_physical_right_sourced_link_force(family_state, higgs, sm_links, higgs_links)
    transformed_force = sm_physical_right_sourced_link_force(
        transformed_state,
        transformed_higgs,
        transformed_sm_links,
        transformed_higgs_links,
    )
    expected_force = sm_physical_right_transform_momenta(force, sm_site_gauge)
    gauss = sm_physical_right_sourced_gauss_constraint(sm_links, sm_momenta, family_state, higgs, higgs_momenta)
    transformed_gauss = sm_physical_right_sourced_gauss_constraint(
        transformed_sm_links,
        transformed_sm_momenta,
        transformed_state,
        transformed_higgs,
        transformed_higgs_momenta,
    )
    expected_gauss = sm_physical_right_project_to_coordinates(
        physical_site_gauge
        @ sm_physical_right_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(physical_site_gauge), -1, -2),
    )
    transport_force = sm_family_sourced_link_force(family_state, higgs, sm_links, higgs_links)

    step_size = 0.003
    kicked = sm_apply_physical_right_sourced_sm_momentum_kick(
        sm_momenta,
        family_state,
        higgs,
        sm_links,
        higgs_links,
        step_size=step_size,
    )
    restored = sm_apply_physical_right_sourced_sm_momentum_kick(
        kicked,
        family_state,
        higgs,
        sm_links,
        higgs_links,
        step_size=-step_size,
    )
    updated = sm_physical_right_sourced_sm_tick(
        family_state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )
    transport_updated = sm_family_sourced_sm_tick(
        family_state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )
    jitted_tick = jax.jit(
        sm_physical_right_sourced_sm_tick,
        static_argnames=(
            "step_size",
            "beta",
            "parameters",
            "wilson_epsilon",
            "higgs_force_epsilon",
            "fermion_current_epsilon",
        ),
    )
    jit_updated = jitted_tick(
        family_state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )

    return PhysicalRightSourcedSMTickDiagnostics(
        transport_force_difference_norm=jnp.linalg.norm(force - transport_force),
        zero_source_force_norm=jnp.linalg.norm(
            sm_physical_right_sourced_link_force(zero_state, vacuum, identity_sm_links, identity_higgs_links),
        ),
        nonzero_source_force_norm=jnp.linalg.norm(force),
        force_covariance_residual=jnp.max(jnp.abs(transformed_force - expected_force)),
        gauss_covariance_residual=jnp.max(jnp.abs(transformed_gauss - expected_gauss)),
        gauss_zero_residual=jnp.linalg.norm(
            sm_physical_right_sourced_gauss_constraint(
                identity_sm_links,
                zero_sm_momenta,
                zero_state,
                vacuum,
                zero_higgs_momenta,
            ),
        ),
        kick_delta_norm=jnp.linalg.norm(kicked - sm_momenta),
        kick_reversibility_residual=jnp.max(jnp.abs(restored - sm_momenta)),
        sm_link_unitarity_residual=sm_link_unitarity_residual(updated[3]),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(updated[5]),
        family_norm_drift=jnp.abs(state_norm_squared(updated[0]) - state_norm_squared(family_state)),
        higgs_field_delta_norm=jnp.linalg.norm(updated[1] - higgs),
        transport_tick_state_difference_norm=jnp.linalg.norm(updated[0] - transport_updated[0]),
        jit_delta_family_state=jnp.max(jnp.abs(jit_updated[0] - updated[0])),
        jit_delta_higgs_field=jnp.max(jnp.abs(jit_updated[1] - updated[1])),
        jit_delta_higgs_momenta=jnp.max(jnp.abs(jit_updated[2] - updated[2])),
        jit_delta_sm_links=jnp.max(jnp.abs(jit_updated[3] - updated[3])),
        jit_delta_sm_momenta=jnp.max(jnp.abs(jit_updated[4] - updated[4])),
        jit_delta_higgs_links=jnp.max(jnp.abs(jit_updated[5] - updated[5])),
    )
