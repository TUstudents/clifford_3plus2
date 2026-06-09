"""Recorded rollout for the QCA_SMv0 physical-right production tick.

Stage 22 wraps the Stage 21 physical-right production tick in the shared
``sim`` recorded-runner interface.  This is a simulator stability layer: it
checks that the single-tick update can be iterated and observed sparsely
without changing the tick semantics.

No boundary rule, quantized register, performance claim, or derivation of
Yukawa/FN inputs is introduced here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import deterministic_yukawa_source_state
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_theta,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tick import (
    sm_physical_right_production_sm_tick,
)
from clifford_3plus2_d5.sim.observables import ObservableMap
from clifford_3plus2_d5.sim.runner import GenericRunConfig, GenericRunResult, run_recorded_loop, run_recorded_scan
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionRolloutState(NamedTuple):
    """State carried by a Stage 22 physical-right production rollout."""

    family_state: jnp.ndarray
    higgs: jnp.ndarray
    higgs_momenta: jnp.ndarray
    sm_links: jnp.ndarray
    sm_momenta: jnp.ndarray
    higgs_links: jnp.ndarray


class PhysicalRightProductionRolloutDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 22 recorded rollout."""

    one_step_direct_residual: jnp.ndarray
    loop_scan_final_residual: jnp.ndarray
    loop_scan_observation_residual: jnp.ndarray
    rollout_family_norm_drift: jnp.ndarray
    rollout_max_family_norm_drift: jnp.ndarray
    max_sm_link_unitarity_residual: jnp.ndarray
    max_higgs_link_unitarity_residual: jnp.ndarray
    higgs_field_total_delta_norm: jnp.ndarray
    higgs_momentum_total_delta_norm: jnp.ndarray
    sm_momentum_total_delta_norm: jnp.ndarray
    production_zero_yukawa_family_difference_norm: jnp.ndarray
    production_zero_yukawa_higgs_momentum_difference_norm: jnp.ndarray
    record_count: jnp.ndarray
    scan_all_finite: jnp.ndarray


def sm_physical_right_production_initial_state(
    lattice_shape: tuple[int, int, int] = (1, 1, 1),
) -> PhysicalRightProductionRolloutState:
    """Return the deterministic Stage 22 rollout initial state."""

    return PhysicalRightProductionRolloutState(
        family_state=deterministic_yukawa_source_state(lattice_shape),
        higgs=deterministic_higgs_field(lattice_shape),
        higgs_momenta=deterministic_higgs_momenta(lattice_shape),
        sm_links=sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25)),
        sm_momenta=deterministic_sm_momenta(lattice_shape),
        higgs_links=sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08)),
    )


def _max_state_delta(
    left: PhysicalRightProductionRolloutState,
    right: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
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


def _max_observation_delta(
    left: Mapping[str, jnp.ndarray],
    right: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    if tuple(left.keys()) != tuple(right.keys()):
        raise ValueError("observation keys must match")
    return jnp.max(jnp.asarray([jnp.max(jnp.abs(left[key] - right[key])) for key in left]))


def sm_physical_right_production_step(
    state: PhysicalRightProductionRolloutState,
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
    """Advance one Stage 22 rollout state."""

    next_parts = sm_physical_right_production_sm_tick(
        state.family_state,
        state.higgs,
        state.higgs_momenta,
        state.sm_links,
        state.sm_momenta,
        state.higgs_links,
        step_size=step_size,
        beta=beta,
        parameters=parameters,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
        wilson_epsilon=wilson_epsilon,
        higgs_force_epsilon=higgs_force_epsilon,
        fermion_current_epsilon=fermion_current_epsilon,
    )
    return PhysicalRightProductionRolloutState(*next_parts)


def sm_physical_right_production_rollout(
    initial_state: PhysicalRightProductionRolloutState,
    *,
    steps: int,
    step_size: float,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> PhysicalRightProductionRolloutState:
    """Return the final state after ``steps`` physical-right production ticks."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = initial_state
    for _ in range(steps):
        state = sm_physical_right_production_step(
            state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
    return state


def sm_physical_right_production_observables(
    state: PhysicalRightProductionRolloutState,
) -> ObservableMap:
    """Return scalar observables for sparse Stage 22 recording."""

    return {
        "family_norm": state_norm_squared(state.family_state),
        "higgs_norm": state_norm_squared(state.higgs),
        "higgs_momentum_norm": state_norm_squared(state.higgs_momenta),
        "sm_momentum_norm": state_norm_squared(state.sm_momenta),
        "sm_link_unitarity_residual": sm_link_unitarity_residual(state.sm_links),
        "higgs_link_unitarity_residual": sm_higgs_link_unitarity_residual(state.higgs_links),
    }


def sm_physical_right_production_recorded_rollout(
    initial_state: PhysicalRightProductionRolloutState | None = None,
    *,
    steps: int = 3,
    record_every: int = 1,
    step_size: float = 0.001,
    use_scan: bool = True,
    use_jit: bool = False,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> GenericRunResult:
    """Run a sparse recorded physical-right production rollout."""

    state = initial_state or sm_physical_right_production_initial_state()
    config = GenericRunConfig(steps=steps, record_every=record_every, use_jit=use_jit)

    def step_fn(current: PhysicalRightProductionRolloutState) -> PhysicalRightProductionRolloutState:
        return sm_physical_right_production_step(
            current,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )

    runner = run_recorded_scan if use_scan else run_recorded_loop
    return runner(state, step_fn, sm_physical_right_production_observables, config)


def sm_physical_right_production_rollout_diagnostics() -> PhysicalRightProductionRolloutDiagnostics:
    """Return focused Stage 22 physical-right production rollout diagnostics."""

    initial = sm_physical_right_production_initial_state()
    steps = 2
    step_size = 0.001
    direct_one_step = sm_physical_right_production_step(initial, step_size=step_size)
    rollout_one_step = sm_physical_right_production_rollout(initial, steps=1, step_size=step_size)
    loop_result = sm_physical_right_production_recorded_rollout(
        initial,
        steps=steps,
        record_every=1,
        step_size=step_size,
        use_scan=False,
    )
    scan_result = sm_physical_right_production_recorded_rollout(
        initial,
        steps=steps,
        record_every=1,
        step_size=step_size,
        use_scan=True,
        use_jit=False,
    )
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()
    zero_yukawa_final = sm_physical_right_production_rollout(
        initial,
        steps=steps,
        step_size=step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )
    family_norms = scan_result.observations["family_norm"]
    initial_family_norm = family_norms[0]
    final_state = scan_result.final_state

    return PhysicalRightProductionRolloutDiagnostics(
        one_step_direct_residual=_max_state_delta(rollout_one_step, direct_one_step),
        loop_scan_final_residual=_max_state_delta(scan_result.final_state, loop_result.final_state),
        loop_scan_observation_residual=_max_observation_delta(scan_result.observations, loop_result.observations),
        rollout_family_norm_drift=jnp.abs(family_norms[-1] - initial_family_norm),
        rollout_max_family_norm_drift=jnp.max(jnp.abs(family_norms - initial_family_norm)),
        max_sm_link_unitarity_residual=jnp.max(scan_result.observations["sm_link_unitarity_residual"]),
        max_higgs_link_unitarity_residual=jnp.max(scan_result.observations["higgs_link_unitarity_residual"]),
        higgs_field_total_delta_norm=jnp.linalg.norm(final_state.higgs - initial.higgs),
        higgs_momentum_total_delta_norm=jnp.linalg.norm(final_state.higgs_momenta - initial.higgs_momenta),
        sm_momentum_total_delta_norm=jnp.linalg.norm(final_state.sm_momenta - initial.sm_momenta),
        production_zero_yukawa_family_difference_norm=jnp.linalg.norm(
            final_state.family_state - zero_yukawa_final.family_state,
        ),
        production_zero_yukawa_higgs_momentum_difference_norm=jnp.linalg.norm(
            final_state.higgs_momenta - zero_yukawa_final.higgs_momenta,
        ),
        record_count=jnp.asarray(scan_result.step_indices.shape[0], dtype=jnp.int32),
        scan_all_finite=scan_result.all_finite,
    )
