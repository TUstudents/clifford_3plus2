"""Full family production tick for QCA_SMv0.

Stage 15 merges the Stage 10 local three-family Yukawa/Higgs source and exact
Yukawa collision into the Stage 14 family-sourced SM gauge tick.  This is the
first classical-field production tick: BCC family transport, gauge-link
backreaction, Higgs force, Yukawa Higgs source, and local family Yukawa
collision are advanced in one explicit local rule.

The stage keeps a live convention boundary.  Stage 14 transport/current
covariance uses the SM transport carrier, while the Stage 10 Yukawa source is
covariant in the physical local Higgs-door convention.  The tick composes both
local rules, but it does not claim that those two gauge conventions have been
microscopically unified.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import sm_family_gauged_dirac_step
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyFNQuarkPathAuxState,
    FamilyFNQuarkPathReadouts,
    FamilyLeptonYukawas,
    SM_FAMILY_DIM,
    sm_apply_family_fn_quark_path_source_kick,
    sm_apply_family_yukawa_collision,
    sm_family_fn_quark_path_higgs_force,
    sm_family_quark_path_readouts_from_masses_ckm,
    sm_family_quark_yukawas_from_path_readouts,
    sm_family_recirculated_quark_path_readouts,
    sm_zero_family_fn_quark_path_aux_state,
    sm_zero_family_fn_quark_path_state_aux,
)
from clifford_3plus2_d5.qca_smv0.sm_family_sourced_tick import (
    sm_family_sourced_link_force,
    sm_family_sourced_sm_tick,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import deterministic_yukawa_source_state, sm_yukawa_higgs_force
from clifford_3plus2_d5.qca_smv0.sm_fn import DEFAULT_FN_QUARK_CHARGES, FN_LAMBDA_WOLFENSTEIN, FNQuarkCharges, FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM, sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_theta,
    sm_higgs_force,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_sourced_tick import sm_apply_sourced_link_update
from clifford_3plus2_d5.sim.state import state_norm_squared


class FamilyProductionSMTickDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 15 full family production tick."""

    zero_yukawa_state_reduction_residual: jnp.ndarray
    zero_yukawa_higgs_reduction_residual: jnp.ndarray
    zero_yukawa_higgs_momentum_reduction_residual: jnp.ndarray
    zero_yukawa_sm_momentum_reduction_residual: jnp.ndarray
    zero_yukawa_sm_link_reduction_residual: jnp.ndarray
    zero_yukawa_higgs_link_reduction_residual: jnp.ndarray
    zero_yukawa_source_norm: jnp.ndarray
    nonzero_yukawa_source_norm: jnp.ndarray
    production_state_delta_norm: jnp.ndarray
    production_higgs_momentum_delta_norm: jnp.ndarray
    source_kick_reversibility_residual: jnp.ndarray
    production_family_norm_drift: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    jit_delta_family_state: jnp.ndarray
    jit_delta_higgs_field: jnp.ndarray
    jit_delta_higgs_momenta: jnp.ndarray
    jit_delta_sm_links: jnp.ndarray
    jit_delta_sm_momenta: jnp.ndarray
    jit_delta_higgs_links: jnp.ndarray


class FamilyFNProductionSMTickOutput(NamedTuple):
    """One production tick with persistent FN quark recirculation memory."""

    state: jnp.ndarray
    higgs: jnp.ndarray
    higgs_momenta: jnp.ndarray
    sm_links: jnp.ndarray
    sm_momenta: jnp.ndarray
    higgs_links: jnp.ndarray
    fn_aux_state: FamilyFNQuarkPathAuxState


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, 3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_higgs_field(field: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
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


def sm_zero_quark_yukawas() -> FNQuarkYukawas:
    """Return zero quark Yukawa matrices for reduction controls."""

    zeros = jnp.zeros((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=jnp.complex64)
    return FNQuarkYukawas(up=zeros, down=zeros)


def sm_zero_family_lepton_yukawas() -> FamilyLeptonYukawas:
    """Return zero lepton Yukawa matrices for reduction controls."""

    zeros = jnp.zeros((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=jnp.complex64)
    return FamilyLeptonYukawas(neutrino=zeros, electron=zeros)


def sm_family_production_higgs_force(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return Higgs force plus local Yukawa source force."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)
    return sm_higgs_force(higgs, higgs_links, parameters=parameters) + sm_yukawa_higgs_force(
        state,
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )


def sm_family_fn_production_higgs_force(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_links: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return Higgs force with quark FN paths plus matrix lepton Yukawas."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_links(higgs_links, lattice_shape)
    zero_quarks = sm_zero_quark_yukawas()
    return sm_family_production_higgs_force(
        state,
        higgs,
        higgs_links,
        parameters=parameters,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    ) + sm_family_fn_quark_path_higgs_force(state, higgs, aux_state=aux_state, readouts=readouts)


def sm_apply_family_production_higgs_momentum_kick(
    higgs_momenta: jnp.ndarray,
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_links: jnp.ndarray,
    *,
    step_size: float,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Kick Higgs momenta by Higgs dynamics plus local Yukawa source."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    source = sm_family_production_higgs_force(
        state,
        higgs,
        higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    return higgs_momenta + jnp.asarray(step_size, dtype=higgs_momenta.dtype) * source


def sm_family_production_sm_tick(
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
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance one full family production tick.

    Return ``(state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links)``.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_sm_momenta(sm_momenta, sm_links)
    _validate_higgs_links(higgs_links, lattice_shape)
    dt = jnp.asarray(step_size, dtype=sm_momenta.dtype)

    first_link_force = sm_family_sourced_link_force(
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
    first_higgs_force = sm_family_production_higgs_force(
        state,
        higgs,
        higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    half_sm_momenta = sm_momenta - 0.5 * dt * first_link_force
    half_higgs_momenta = higgs_momenta + 0.5 * dt * first_higgs_force

    updated_sm_links, updated_higgs_links = sm_apply_sourced_link_update(
        sm_links,
        higgs_links,
        half_sm_momenta,
        step_size=step_size,
    )
    updated_higgs = higgs + dt * half_higgs_momenta
    half_collided = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    transported = sm_family_gauged_dirac_step(half_collided, updated_sm_links)
    updated_state = sm_apply_family_yukawa_collision(
        transported,
        updated_higgs,
        step_size=0.5 * step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )

    second_link_force = sm_family_sourced_link_force(
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
    second_higgs_force = sm_family_production_higgs_force(
        updated_state,
        updated_higgs,
        updated_higgs_links,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
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


def sm_family_fn_production_sm_tick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    sm_links: jnp.ndarray,
    sm_momenta: jnp.ndarray,
    higgs_links: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one family production tick with persistent FN quark recirculation.

    Gauge transport and classical fields follow ``sm_family_production_sm_tick``.
    The local quark Yukawa half-steps are replaced by hidden-FN source kicks that
    carry an auxiliary memory through the tick.  Leptons still use the local
    matrix collision, with quark matrices set to zero there to avoid double
    counting the quark sector.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_sm_momenta(sm_momenta, sm_links)
    _validate_higgs_links(higgs_links, lattice_shape)
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()

    dt = jnp.asarray(step_size, dtype=sm_momenta.dtype)
    zero_quarks = sm_zero_quark_yukawas()
    first_link_force = sm_family_sourced_link_force(
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
    first_higgs_force = sm_family_production_higgs_force(
        state,
        higgs,
        higgs_links,
        parameters=parameters,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    ) + sm_family_fn_quark_path_higgs_force(state, higgs, aux_state=aux_state, readouts=readouts)
    half_sm_momenta = sm_momenta - 0.5 * dt * first_link_force
    half_higgs_momenta = higgs_momenta + 0.5 * dt * first_higgs_force

    updated_sm_links, updated_higgs_links = sm_apply_sourced_link_update(
        sm_links,
        higgs_links,
        half_sm_momenta,
        step_size=step_size,
    )
    updated_higgs = higgs + dt * half_higgs_momenta

    first_quark = sm_apply_family_fn_quark_path_source_kick(
        state,
        higgs,
        aux_state=aux_state,
        readouts=readouts,
        step_size=0.5 * step_size,
    )
    half_collided = sm_apply_family_yukawa_collision(
        first_quark.state,
        higgs,
        step_size=0.5 * step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    )
    transported = sm_family_gauged_dirac_step(half_collided, updated_sm_links)
    second_quark = sm_apply_family_fn_quark_path_source_kick(
        transported,
        updated_higgs,
        aux_state=first_quark.aux_state,
        readouts=readouts,
        step_size=0.5 * step_size,
    )
    updated_state = sm_apply_family_yukawa_collision(
        second_quark.state,
        updated_higgs,
        step_size=0.5 * step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    )

    second_link_force = sm_family_sourced_link_force(
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
    second_higgs_force = sm_family_production_higgs_force(
        updated_state,
        updated_higgs,
        updated_higgs_links,
        parameters=parameters,
        quark_yukawas=zero_quarks,
        lepton_yukawas=lepton_yukawas,
    ) + sm_family_fn_quark_path_higgs_force(
        updated_state,
        updated_higgs,
        aux_state=second_quark.aux_state,
        readouts=readouts,
    )
    updated_sm_momenta = half_sm_momenta - 0.5 * dt * second_link_force
    updated_higgs_momenta = half_higgs_momenta + 0.5 * dt * second_higgs_force
    return FamilyFNProductionSMTickOutput(
        state=updated_state,
        higgs=updated_higgs,
        higgs_momenta=updated_higgs_momenta,
        sm_links=updated_sm_links,
        sm_momenta=updated_sm_momenta,
        higgs_links=updated_higgs_links,
        fn_aux_state=second_quark.aux_state,
    )


def sm_family_fn_unitary_production_sm_tick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    higgs_momenta: jnp.ndarray,
    sm_links: jnp.ndarray,
    sm_momenta: jnp.ndarray,
    higgs_links: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one production tick with FN readouts folded into a unitary collision.

    This mode treats the explicit FN path network as the origin of the quark
    Yukawa matrices, then applies the existing exact local Yukawa exponential.
    The path auxiliary state is preserved only for output-shape compatibility.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    _validate_higgs_momenta(higgs_momenta, lattice_shape)
    _validate_sm_links(sm_links, lattice_shape)
    _validate_sm_momenta(sm_momenta, sm_links)
    _validate_higgs_links(higgs_links, lattice_shape)
    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_path_aux_state((*lattice_shape, 4, 3, 2), readouts=readouts)

    quark_yukawas = sm_family_quark_yukawas_from_path_readouts(readouts)
    updated = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    return FamilyFNProductionSMTickOutput(
        state=updated[0],
        higgs=updated[1],
        higgs_momenta=updated[2],
        sm_links=updated[3],
        sm_momenta=updated[4],
        higgs_links=updated[5],
        fn_aux_state=aux_state,
    )


def sm_family_fn_production_initial_state(
    lattice_shape: tuple[int, int, int] = (1, 1, 1),
) -> FamilyFNProductionSMTickOutput:
    """Return a deterministic FN production rollout state with empty aux."""

    return FamilyFNProductionSMTickOutput(
        state=deterministic_yukawa_source_state(lattice_shape),
        higgs=deterministic_higgs_field(lattice_shape),
        higgs_momenta=deterministic_higgs_momenta(lattice_shape),
        sm_links=sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25)),
        sm_momenta=deterministic_sm_momenta(lattice_shape),
        higgs_links=sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08)),
        fn_aux_state=sm_zero_family_fn_quark_path_state_aux(lattice_shape),
    )


def sm_family_fn_production_step(
    state: FamilyFNProductionSMTickOutput,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one FN production rollout state."""

    return sm_family_fn_production_sm_tick(
        state.state,
        state.higgs,
        state.higgs_momenta,
        state.sm_links,
        state.sm_momenta,
        state.higgs_links,
        state.fn_aux_state,
        readouts=readouts,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_fn_production_step_from_masses_ckm(
    state: FamilyFNProductionSMTickOutput,
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one FN production step calibrated from masses and CKM."""

    readouts = sm_family_quark_path_readouts_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
    )
    return sm_family_fn_production_step(
        state,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        readouts=readouts,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_fn_unitary_production_step(
    state: FamilyFNProductionSMTickOutput,
    *,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one FN production state using exact local Yukawa collisions."""

    return sm_family_fn_unitary_production_sm_tick(
        state.state,
        state.higgs,
        state.higgs_momenta,
        state.sm_links,
        state.sm_momenta,
        state.higgs_links,
        state.fn_aux_state,
        readouts=readouts,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_fn_unitary_production_step_from_masses_ckm(
    state: FamilyFNProductionSMTickOutput,
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Advance one exact-unitary FN step calibrated from masses and CKM."""

    readouts = sm_family_quark_path_readouts_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
    )
    return sm_family_fn_unitary_production_step(
        state,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        readouts=readouts,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_fn_production_rollout(
    initial_state: FamilyFNProductionSMTickOutput,
    *,
    steps: int,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Return the final state after ``steps`` FN-memory production ticks."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = initial_state
    for _ in range(steps):
        state = sm_family_fn_production_step(
            state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            readouts=readouts,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
    return state


def sm_family_fn_unitary_production_rollout(
    initial_state: FamilyFNProductionSMTickOutput,
    *,
    steps: int,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Return the final state after exact-unitary FN production ticks."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = initial_state
    for _ in range(steps):
        state = sm_family_fn_unitary_production_step(
            state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            readouts=readouts,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
    return state


def sm_family_fn_production_rollout_from_masses_ckm(
    initial_state: FamilyFNProductionSMTickOutput,
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    steps: int,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Run FN production calibrated directly from masses and CKM."""

    readouts = sm_family_quark_path_readouts_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
    )
    return sm_family_fn_production_rollout(
        initial_state,
        steps=steps,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        readouts=readouts,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_fn_unitary_production_rollout_from_masses_ckm(
    initial_state: FamilyFNProductionSMTickOutput,
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    steps: int,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> FamilyFNProductionSMTickOutput:
    """Run exact-unitary FN production calibrated from masses and CKM."""

    readouts = sm_family_quark_path_readouts_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
    )
    return sm_family_fn_unitary_production_rollout(
        initial_state,
        steps=steps,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        readouts=readouts,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )


def sm_family_production_tick_diagnostics() -> FamilyProductionSMTickDiagnostics:
    """Return focused Stage 15 full family production diagnostics."""

    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    vacuum = sm_constant_higgs(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()
    zero_state = jnp.zeros_like(state)
    step_size = 0.003

    stage14_zero = sm_family_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )
    production_zero = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )
    production = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )
    stage14_default = sm_family_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )
    source = sm_yukawa_higgs_force(state, higgs)
    zero_source = sm_yukawa_higgs_force(zero_state, vacuum)
    kicked = sm_apply_family_production_higgs_momentum_kick(
        higgs_momenta,
        state,
        higgs,
        higgs_links,
        step_size=step_size,
    )
    restored = sm_apply_family_production_higgs_momentum_kick(
        kicked,
        state,
        higgs,
        higgs_links,
        step_size=-step_size,
    )
    jitted_tick = jax.jit(
        sm_family_production_sm_tick,
        static_argnames=(
            "step_size",
            "beta",
            "parameters",
            "wilson_epsilon",
            "higgs_force_epsilon",
            "fermion_current_epsilon",
        ),
    )
    jit_production = jitted_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=step_size,
    )

    return FamilyProductionSMTickDiagnostics(
        zero_yukawa_state_reduction_residual=jnp.max(jnp.abs(production_zero[0] - stage14_zero[0])),
        zero_yukawa_higgs_reduction_residual=jnp.max(jnp.abs(production_zero[1] - stage14_zero[1])),
        zero_yukawa_higgs_momentum_reduction_residual=jnp.max(jnp.abs(production_zero[2] - stage14_zero[2])),
        zero_yukawa_sm_link_reduction_residual=jnp.max(jnp.abs(production_zero[3] - stage14_zero[3])),
        zero_yukawa_sm_momentum_reduction_residual=jnp.max(jnp.abs(production_zero[4] - stage14_zero[4])),
        zero_yukawa_higgs_link_reduction_residual=jnp.max(jnp.abs(production_zero[5] - stage14_zero[5])),
        zero_yukawa_source_norm=jnp.linalg.norm(zero_source),
        nonzero_yukawa_source_norm=jnp.linalg.norm(source),
        production_state_delta_norm=jnp.linalg.norm(production[0] - stage14_default[0]),
        production_higgs_momentum_delta_norm=jnp.linalg.norm(production[2] - stage14_default[2]),
        source_kick_reversibility_residual=jnp.max(jnp.abs(restored - higgs_momenta)),
        production_family_norm_drift=jnp.abs(state_norm_squared(production[0]) - state_norm_squared(state)),
        sm_link_unitarity_residual=sm_link_unitarity_residual(production[3]),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(production[5]),
        jit_delta_family_state=jnp.max(jnp.abs(jit_production[0] - production[0])),
        jit_delta_higgs_field=jnp.max(jnp.abs(jit_production[1] - production[1])),
        jit_delta_higgs_momenta=jnp.max(jnp.abs(jit_production[2] - production[2])),
        jit_delta_sm_links=jnp.max(jnp.abs(jit_production[3] - production[3])),
        jit_delta_sm_momenta=jnp.max(jnp.abs(jit_production[4] - production[4])),
        jit_delta_higgs_links=jnp.max(jnp.abs(jit_production[5] - production[5])),
    )
