"""Explicit inverse for the physical-right production tick.

Stage 28 turns the Stage 27 adjoint pieces into a genuine inverse map for the
current explicit production tick.  The inverse is not a negative-timestep
forward tick.  It reverses the actual update order:

1. undo the second source half-kicks using final fields;
2. undo the link and Higgs-field drift using the recovered half momenta;
3. undo the frozen fermion substage using its adjoint;
4. undo the first source half-kicks using the recovered initial fields.

This is an inverse/audit layer for the existing tick, not a new dynamics rule
or an energy-convergent integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_GENERATOR_COUNT, SM_INTERNAL_DIM, sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_adjoint import (
    sm_physical_right_production_frozen_fermion_stage_adjoint,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tick import (
    sm_physical_right_production_higgs_force,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_sourced_tick import (
    sm_physical_right_sourced_link_force,
)
from clifford_3plus2_d5.qca_smv0.sm_sourced_tick import sm_apply_sourced_link_update
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionInverseDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 28 explicit production inverse."""

    inverse_roundtrip_residual: jnp.ndarray
    forward_inverse_forward_residual: jnp.ndarray
    inverse_family_norm_drift: jnp.ndarray
    forward_family_norm_drift: jnp.ndarray
    inverse_sm_link_unitarity_residual: jnp.ndarray
    inverse_higgs_link_unitarity_residual: jnp.ndarray
    naive_negative_residual: jnp.ndarray
    explicit_inverse_improvement_ratio: jnp.ndarray
    jit_delta_family_state: jnp.ndarray
    jit_delta_higgs: jnp.ndarray
    jit_delta_higgs_momenta: jnp.ndarray
    jit_delta_sm_links: jnp.ndarray
    jit_delta_sm_momenta: jnp.ndarray
    jit_delta_higgs_links: jnp.ndarray


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:-1] != (4, SM_INTERNAL_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, families)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_higgs(field: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if field.ndim != 4 or field.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and field.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the lattice")


def _validate_higgs_momenta(momenta: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if momenta.ndim != 4 or momenta.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs momenta must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and momenta.shape[:3] != lattice_shape:
        raise ValueError("Higgs momenta must match the lattice")


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


def _validate_higgs_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, SM_HIGGS_DIM, SM_HIGGS_DIM):
        raise ValueError("Higgs links must have shape (nx, ny, nz, 8, 2, 2)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("Higgs links must match the lattice")


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


def sm_physical_right_production_inverse_step(
    final_state: PhysicalRightProductionRolloutState,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> PhysicalRightProductionRolloutState:
    """Invert one Stage 21 physical-right production tick."""

    lattice_shape = _validate_family_state(final_state.family_state)
    _validate_higgs(final_state.higgs, lattice_shape)
    _validate_higgs_momenta(final_state.higgs_momenta, lattice_shape)
    _validate_sm_links(final_state.sm_links, lattice_shape)
    _validate_sm_momenta(final_state.sm_momenta, final_state.sm_links)
    _validate_higgs_links(final_state.higgs_links, lattice_shape)
    dt = jnp.asarray(step_size, dtype=final_state.sm_momenta.dtype)

    second_link_force = sm_physical_right_sourced_link_force(
        final_state.family_state,
        final_state.higgs,
        final_state.sm_links,
        final_state.higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    second_higgs_force = sm_physical_right_production_higgs_force(
        final_state.family_state,
        final_state.higgs,
        final_state.higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    half_sm_momenta = final_state.sm_momenta + 0.5 * dt * second_link_force
    half_higgs_momenta = final_state.higgs_momenta - 0.5 * dt * second_higgs_force

    initial_sm_links, initial_higgs_links = sm_apply_sourced_link_update(
        final_state.sm_links,
        final_state.higgs_links,
        half_sm_momenta,
        step_size=-step_size,
    )
    initial_higgs = final_state.higgs - dt * half_higgs_momenta
    initial_family_state = sm_physical_right_production_frozen_fermion_stage_adjoint(
        final_state.family_state,
        initial_higgs,
        final_state.higgs,
        final_state.sm_links,
        step_size=step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )

    first_link_force = sm_physical_right_sourced_link_force(
        initial_family_state,
        initial_higgs,
        initial_sm_links,
        initial_higgs_links,
        beta=beta,
        parameters=parameters,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    first_higgs_force = sm_physical_right_production_higgs_force(
        initial_family_state,
        initial_higgs,
        initial_higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    initial_sm_momenta = half_sm_momenta + 0.5 * dt * first_link_force
    initial_higgs_momenta = half_higgs_momenta - 0.5 * dt * first_higgs_force

    return PhysicalRightProductionRolloutState(
        family_state=initial_family_state,
        higgs=initial_higgs,
        higgs_momenta=initial_higgs_momenta,
        sm_links=initial_sm_links,
        sm_momenta=initial_sm_momenta,
        higgs_links=initial_higgs_links,
    )


def sm_physical_right_production_inverse_diagnostics() -> PhysicalRightProductionInverseDiagnostics:
    """Return focused Stage 28 explicit inverse diagnostics."""

    initial = sm_physical_right_production_initial_state()
    step_size = 1e-3
    forward = sm_physical_right_production_step(initial, step_size=step_size)
    restored = sm_physical_right_production_inverse_step(forward, step_size=step_size)
    replayed = sm_physical_right_production_step(restored, step_size=step_size)
    naive_negative = sm_physical_right_production_step(forward, step_size=-step_size)
    jitted_inverse = jax.jit(sm_physical_right_production_inverse_step, static_argnames=("step_size",))
    jit_restored = jitted_inverse(forward, step_size=step_size)
    inverse_residual = _rollout_delta(initial, restored)
    naive_residual = _rollout_delta(initial, naive_negative)

    return PhysicalRightProductionInverseDiagnostics(
        inverse_roundtrip_residual=inverse_residual,
        forward_inverse_forward_residual=_rollout_delta(forward, replayed),
        inverse_family_norm_drift=jnp.abs(state_norm_squared(restored.family_state) - state_norm_squared(initial.family_state)),
        forward_family_norm_drift=jnp.abs(state_norm_squared(forward.family_state) - state_norm_squared(initial.family_state)),
        inverse_sm_link_unitarity_residual=sm_link_unitarity_residual(restored.sm_links),
        inverse_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(restored.higgs_links),
        naive_negative_residual=naive_residual,
        explicit_inverse_improvement_ratio=naive_residual / jnp.maximum(
            inverse_residual,
            jnp.asarray(1e-12, dtype=naive_residual.dtype),
        ),
        jit_delta_family_state=jnp.max(jnp.abs(jit_restored.family_state - restored.family_state)),
        jit_delta_higgs=jnp.max(jnp.abs(jit_restored.higgs - restored.higgs)),
        jit_delta_higgs_momenta=jnp.max(jnp.abs(jit_restored.higgs_momenta - restored.higgs_momenta)),
        jit_delta_sm_links=jnp.max(jnp.abs(jit_restored.sm_links - restored.sm_links)),
        jit_delta_sm_momenta=jnp.max(jnp.abs(jit_restored.sm_momenta - restored.sm_momenta)),
        jit_delta_higgs_links=jnp.max(jnp.abs(jit_restored.higgs_links - restored.higgs_links)),
    )
