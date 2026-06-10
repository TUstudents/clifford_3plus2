"""Adjoint audit for the physical-right production tick.

Stage 27 separates a positive unitary fact from a production-integrator
limitation.  The fermion substage of the production tick is an exact unitary
map when the background fields are frozen: local half-collision, physical-right
BCC transport, local half-collision.  This module implements the adjoint
physical-right BCC transport and audits that frozen fermion substage directly.

The full production tick is still not given an inverse by simply using a
negative timestep.  That limitation is recorded explicitly; a later stage must
build a genuine adjoint/full reversible integrator if the production dynamics
is to claim that property.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS, bcc_dirac_hop_matrices
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyLeptonYukawas,
    sm_apply_family_yukawa_collision,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_INTERNAL_DIM, sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import (
    sm_family_physical_right_gauged_dirac_step,
    sm_physical_right_links_from_transport,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionAdjointDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 27 adjoint/inverse boundaries."""

    transport_adjoint_residual: jnp.ndarray
    frozen_fermion_stage_adjoint_residual: jnp.ndarray
    frozen_fermion_norm_drift: jnp.ndarray
    local_collision_inverse_residual: jnp.ndarray
    naive_negative_tick_residual: jnp.ndarray
    naive_negative_tick_limitation_detected: jnp.ndarray
    forward_sm_link_unitarity_residual: jnp.ndarray
    naive_negative_sm_link_unitarity_residual: jnp.ndarray
    forward_higgs_link_unitarity_residual: jnp.ndarray
    naive_negative_higgs_link_unitarity_residual: jnp.ndarray


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:-1] != (4, SM_INTERNAL_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, families)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_sm_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("SM links must have shape (nx, ny, nz, 8, 32, 32)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("SM links must match the state lattice")


def _validate_higgs(higgs: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if higgs.ndim != 4 or higgs.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and higgs.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the state lattice")


def sm_family_physical_right_gauged_dirac_adjoint_step(
    state: jnp.ndarray,
    transport_links: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the adjoint of the physical-right family BCC transport."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(transport_links, lattice_shape)
    physical_links = sm_physical_right_links_from_transport(transport_links)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, tuple(-component for component in displacement))
        link = source_roll(physical_links[..., index, :, :], tuple(-component for component in displacement))
        internal_adjoint = jnp.einsum("...ab,...raf->...rbf", jnp.conj(link), source)
        out = out + jnp.einsum("sr,...rdf->...sdf", jnp.conj(hops[index].T), internal_adjoint)
    return out


def sm_physical_right_production_frozen_fermion_stage(
    state: jnp.ndarray,
    old_higgs: jnp.ndarray,
    updated_higgs: jnp.ndarray,
    updated_sm_links: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return the fermion substage used inside the production tick."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs(old_higgs, lattice_shape)
    _validate_higgs(updated_higgs, lattice_shape)
    _validate_sm_links(updated_sm_links, lattice_shape)
    half_collided = sm_apply_family_yukawa_collision(
        state,
        old_higgs,
        step_size=0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    transported = sm_family_physical_right_gauged_dirac_step(half_collided, updated_sm_links)
    return sm_apply_family_yukawa_collision(
        transported,
        updated_higgs,
        step_size=0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )


def sm_physical_right_production_frozen_fermion_stage_adjoint(
    state: jnp.ndarray,
    old_higgs: jnp.ndarray,
    updated_higgs: jnp.ndarray,
    updated_sm_links: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return the adjoint of the frozen production fermion substage."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs(old_higgs, lattice_shape)
    _validate_higgs(updated_higgs, lattice_shape)
    _validate_sm_links(updated_sm_links, lattice_shape)
    uncollided_new = sm_apply_family_yukawa_collision(
        state,
        updated_higgs,
        step_size=-0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    untransported = sm_family_physical_right_gauged_dirac_adjoint_step(uncollided_new, updated_sm_links)
    return sm_apply_family_yukawa_collision(
        untransported,
        old_higgs,
        step_size=-0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )


def _rollout_delta(left: PhysicalRightProductionRolloutState, right: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.max(
        jnp.asarray(
            [
                jnp.max(jnp.abs(left.family_state - right.family_state)),
                jnp.max(jnp.abs(left.higgs - right.higgs)),
                jnp.max(jnp.abs(left.higgs_momenta - right.higgs_momenta)),
                jnp.max(jnp.abs(left.sm_links - right.sm_links)),
                jnp.max(jnp.abs(left.sm_momenta - right.sm_momenta)),
                jnp.max(jnp.abs(left.higgs_links - right.higgs_links)),
            ],
        ),
    )


def sm_physical_right_production_adjoint_diagnostics() -> PhysicalRightProductionAdjointDiagnostics:
    """Return focused Stage 27 adjoint/reversibility diagnostics."""

    initial = sm_physical_right_production_initial_state()
    step_size = 1e-3
    transported = sm_family_physical_right_gauged_dirac_step(initial.family_state, initial.sm_links)
    transport_restored = sm_family_physical_right_gauged_dirac_adjoint_step(transported, initial.sm_links)
    locally_collided = sm_apply_family_yukawa_collision(
        initial.family_state,
        initial.higgs,
        step_size=0.5 * step_size,
    )
    local_restored = sm_apply_family_yukawa_collision(
        locally_collided,
        initial.higgs,
        step_size=-0.5 * step_size,
    )
    forward = sm_physical_right_production_step(initial, step_size=step_size)
    frozen_forward = sm_physical_right_production_frozen_fermion_stage(
        initial.family_state,
        initial.higgs,
        forward.higgs,
        forward.sm_links,
        step_size=step_size,
    )
    frozen_restored = sm_physical_right_production_frozen_fermion_stage_adjoint(
        frozen_forward,
        initial.higgs,
        forward.higgs,
        forward.sm_links,
        step_size=step_size,
    )
    naive_negative = sm_physical_right_production_step(forward, step_size=-step_size)
    naive_residual = _rollout_delta(initial, naive_negative)

    return PhysicalRightProductionAdjointDiagnostics(
        transport_adjoint_residual=jnp.max(jnp.abs(transport_restored - initial.family_state)),
        frozen_fermion_stage_adjoint_residual=jnp.max(jnp.abs(frozen_restored - initial.family_state)),
        frozen_fermion_norm_drift=jnp.abs(state_norm_squared(frozen_forward) - state_norm_squared(initial.family_state)),
        local_collision_inverse_residual=jnp.max(jnp.abs(local_restored - initial.family_state)),
        naive_negative_tick_residual=naive_residual,
        naive_negative_tick_limitation_detected=naive_residual > jnp.asarray(1e-2, dtype=naive_residual.dtype),
        forward_sm_link_unitarity_residual=sm_link_unitarity_residual(forward.sm_links),
        naive_negative_sm_link_unitarity_residual=sm_link_unitarity_residual(naive_negative.sm_links),
        forward_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(forward.higgs_links),
        naive_negative_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(naive_negative.higgs_links),
    )
