"""Gauss-monitor rollout for the QCA_SMv0 physical-right production tick.

Stage 23 adds a physical-right sourced Gauss monitor to the Stage 22
production rollout.  The monitor reads the existing Stage 20 Gauss diagnostic

    electric divergence - physical-right family charge - embedded Higgs charge

on the repeated Stage 21 production dynamics.  This is a physics observable
layer, not a projection step: it does not enforce Gauss, add boundary rules, or
change the tick.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import deterministic_yukawa_source_state
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_GENERATOR_COUNT, sm_identity_links, sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_link_unitarity_residual,
    sm_identity_higgs_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_sourced_tick import (
    sm_physical_right_sourced_gauss_constraint,
)
from clifford_3plus2_d5.sim.observables import ObservableMap, observations_all_finite, stack_observations
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionGaussDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 23 physical-right production Gauss monitor."""

    vacuum_initial_gauss_norm: jnp.ndarray
    vacuum_final_gauss_norm: jnp.ndarray
    vacuum_family_norm: jnp.ndarray
    deterministic_initial_gauss_norm: jnp.ndarray
    deterministic_final_gauss_norm: jnp.ndarray
    deterministic_max_gauss_norm: jnp.ndarray
    deterministic_gauss_delta_norm: jnp.ndarray
    zero_yukawa_final_gauss_difference_norm: jnp.ndarray
    rollout_family_norm_drift: jnp.ndarray
    max_sm_link_unitarity_residual: jnp.ndarray
    max_higgs_link_unitarity_residual: jnp.ndarray
    history_count: jnp.ndarray
    history_all_finite: jnp.ndarray


def sm_physical_right_production_vacuum_state(
    lattice_shape: tuple[int, int, int] = (1, 1, 1),
) -> PhysicalRightProductionRolloutState:
    """Return the exact zero-source vacuum control state for Stage 23."""

    family_state = jnp.zeros_like(deterministic_yukawa_source_state(lattice_shape))
    higgs = sm_constant_higgs(lattice_shape)
    return PhysicalRightProductionRolloutState(
        family_state=family_state,
        higgs=higgs,
        higgs_momenta=jnp.zeros_like(higgs),
        sm_links=sm_identity_links(lattice_shape),
        sm_momenta=jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32),
        higgs_links=sm_identity_higgs_links(lattice_shape),
    )


def sm_physical_right_production_gauss(
    state: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    """Return the existing physical-right sourced Gauss diagnostic on a rollout state."""

    return sm_physical_right_sourced_gauss_constraint(
        state.sm_links,
        state.sm_momenta,
        state.family_state,
        state.higgs,
        state.higgs_momenta,
    )


def sm_physical_right_production_gauss_observables(
    state: PhysicalRightProductionRolloutState,
) -> ObservableMap:
    """Return Stage 23 scalar Gauss-monitor observables."""

    gauss = sm_physical_right_production_gauss(state)
    return {
        "family_norm": state_norm_squared(state.family_state),
        "gauss_norm": jnp.linalg.norm(gauss),
        "gauss_max_abs": jnp.max(jnp.abs(gauss)),
        "sm_link_unitarity_residual": sm_link_unitarity_residual(state.sm_links),
        "higgs_link_unitarity_residual": sm_higgs_link_unitarity_residual(state.higgs_links),
    }


def sm_physical_right_production_gauss_history(
    initial_state: PhysicalRightProductionRolloutState | None = None,
    *,
    steps: int = 2,
    step_size: float = 0.001,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> tuple[PhysicalRightProductionRolloutState, dict[str, jnp.ndarray]]:
    """Run a short Python-loop rollout and record Gauss-monitor observables."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = initial_state or sm_physical_right_production_initial_state()
    observations: list[ObservableMap] = [sm_physical_right_production_gauss_observables(state)]
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
        observations.append(sm_physical_right_production_gauss_observables(state))
    return state, stack_observations(tuple(observations))


def sm_physical_right_production_gauss_diagnostics() -> PhysicalRightProductionGaussDiagnostics:
    """Return focused Stage 23 physical-right production Gauss diagnostics."""

    steps = 2
    step_size = 0.001
    vacuum_initial = sm_physical_right_production_vacuum_state()
    vacuum_final, vacuum_history = sm_physical_right_production_gauss_history(
        vacuum_initial,
        steps=steps,
        step_size=step_size,
    )
    deterministic_initial = sm_physical_right_production_initial_state()
    deterministic_final, deterministic_history = sm_physical_right_production_gauss_history(
        deterministic_initial,
        steps=steps,
        step_size=step_size,
    )
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()
    default_compare_final, _ = sm_physical_right_production_gauss_history(
        deterministic_initial,
        steps=3,
        step_size=step_size,
    )
    zero_yukawa_final, _ = sm_physical_right_production_gauss_history(
        deterministic_initial,
        steps=3,
        step_size=step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )
    initial_gauss = sm_physical_right_production_gauss(deterministic_initial)
    final_gauss = sm_physical_right_production_gauss(deterministic_final)
    default_compare_final_gauss = sm_physical_right_production_gauss(default_compare_final)
    zero_yukawa_final_gauss = sm_physical_right_production_gauss(zero_yukawa_final)
    family_norms = deterministic_history["family_norm"]

    return PhysicalRightProductionGaussDiagnostics(
        vacuum_initial_gauss_norm=vacuum_history["gauss_norm"][0],
        vacuum_final_gauss_norm=vacuum_history["gauss_norm"][-1],
        vacuum_family_norm=state_norm_squared(vacuum_final.family_state),
        deterministic_initial_gauss_norm=deterministic_history["gauss_norm"][0],
        deterministic_final_gauss_norm=deterministic_history["gauss_norm"][-1],
        deterministic_max_gauss_norm=jnp.max(deterministic_history["gauss_norm"]),
        deterministic_gauss_delta_norm=jnp.linalg.norm(final_gauss - initial_gauss),
        zero_yukawa_final_gauss_difference_norm=jnp.linalg.norm(
            default_compare_final_gauss - zero_yukawa_final_gauss,
        ),
        rollout_family_norm_drift=jnp.abs(family_norms[-1] - family_norms[0]),
        max_sm_link_unitarity_residual=jnp.max(deterministic_history["sm_link_unitarity_residual"]),
        max_higgs_link_unitarity_residual=jnp.max(deterministic_history["higgs_link_unitarity_residual"]),
        history_count=jnp.asarray(deterministic_history["gauss_norm"].shape[0], dtype=jnp.int32),
        history_all_finite=observations_all_finite(vacuum_history) & observations_all_finite(deterministic_history),
    )
