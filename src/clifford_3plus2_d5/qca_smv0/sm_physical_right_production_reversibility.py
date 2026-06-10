"""Trajectory-level reversibility audit for the production tick.

Stage 29 composes the Stage 28 explicit inverse over a short trajectory.  This
checks that the inverse is not merely a one-step algebraic certificate but can
rewind an iterated physical-right production rollout.

The audit still belongs to the existing discrete map.  It does not upgrade the
hybrid production tick into an energy-convergent Hamiltonian integrator or add
new dynamics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_inverse import (
    sm_physical_right_production_inverse_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionReversibilityDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 29 trajectory reversibility."""

    steps: jnp.ndarray
    trajectory_roundtrip_residual: jnp.ndarray
    trajectory_replay_residual: jnp.ndarray
    max_path_restore_residual: jnp.ndarray
    inverse_family_norm_drift: jnp.ndarray
    forward_family_norm_drift: jnp.ndarray
    max_forward_sm_link_unitarity_residual: jnp.ndarray
    max_forward_higgs_link_unitarity_residual: jnp.ndarray
    max_inverse_sm_link_unitarity_residual: jnp.ndarray
    max_inverse_higgs_link_unitarity_residual: jnp.ndarray
    naive_negative_trajectory_residual: jnp.ndarray
    trajectory_inverse_improvement_ratio: jnp.ndarray
    jit_inverse_rollout_delta: jnp.ndarray


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


def _max_link_unitarity(state: PhysicalRightProductionRolloutState) -> tuple[jnp.ndarray, jnp.ndarray]:
    return sm_link_unitarity_residual(state.sm_links), sm_higgs_link_unitarity_residual(state.higgs_links)


def sm_physical_right_production_inverse_rollout(
    final_state: PhysicalRightProductionRolloutState,
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
    """Rewind ``steps`` applications of the physical-right production tick."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = final_state
    for _ in range(steps):
        state = sm_physical_right_production_inverse_step(
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


def sm_physical_right_production_reversibility_diagnostics(
    *,
    steps: int = 2,
    step_size: float = 1e-3,
) -> PhysicalRightProductionReversibilityDiagnostics:
    """Return focused Stage 29 trajectory-reversibility diagnostics."""

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    initial = sm_physical_right_production_initial_state()
    forward_states = [initial]
    state = initial
    max_forward_sm_unitarity, max_forward_higgs_unitarity = _max_link_unitarity(initial)
    for _ in range(steps):
        state = sm_physical_right_production_step(state, step_size=step_size)
        forward_states.append(state)
        sm_unitarity, higgs_unitarity = _max_link_unitarity(state)
        max_forward_sm_unitarity = jnp.maximum(max_forward_sm_unitarity, sm_unitarity)
        max_forward_higgs_unitarity = jnp.maximum(max_forward_higgs_unitarity, higgs_unitarity)

    final = forward_states[-1]
    restored = final
    max_path_restore = jnp.asarray(0.0, dtype=jnp.float32)
    max_inverse_sm_unitarity, max_inverse_higgs_unitarity = _max_link_unitarity(final)
    for index in range(steps - 1, -1, -1):
        restored = sm_physical_right_production_inverse_step(restored, step_size=step_size)
        max_path_restore = jnp.maximum(max_path_restore, _rollout_delta(restored, forward_states[index]))
        sm_unitarity, higgs_unitarity = _max_link_unitarity(restored)
        max_inverse_sm_unitarity = jnp.maximum(max_inverse_sm_unitarity, sm_unitarity)
        max_inverse_higgs_unitarity = jnp.maximum(max_inverse_higgs_unitarity, higgs_unitarity)

    replayed = sm_physical_right_production_rollout(restored, steps=steps, step_size=step_size)
    naive_negative = sm_physical_right_production_rollout(final, steps=steps, step_size=-step_size)
    jitted_inverse_rollout = jax.jit(
        sm_physical_right_production_inverse_rollout,
        static_argnames=("steps", "step_size"),
    )
    jit_restored = jitted_inverse_rollout(final, steps=steps, step_size=step_size)
    trajectory_residual = _rollout_delta(initial, restored)
    naive_residual = _rollout_delta(initial, naive_negative)

    return PhysicalRightProductionReversibilityDiagnostics(
        steps=jnp.asarray(steps, dtype=jnp.int32),
        trajectory_roundtrip_residual=trajectory_residual,
        trajectory_replay_residual=_rollout_delta(final, replayed),
        max_path_restore_residual=max_path_restore,
        inverse_family_norm_drift=jnp.abs(state_norm_squared(restored.family_state) - state_norm_squared(initial.family_state)),
        forward_family_norm_drift=jnp.abs(state_norm_squared(final.family_state) - state_norm_squared(initial.family_state)),
        max_forward_sm_link_unitarity_residual=max_forward_sm_unitarity,
        max_forward_higgs_link_unitarity_residual=max_forward_higgs_unitarity,
        max_inverse_sm_link_unitarity_residual=max_inverse_sm_unitarity,
        max_inverse_higgs_link_unitarity_residual=max_inverse_higgs_unitarity,
        naive_negative_trajectory_residual=naive_residual,
        trajectory_inverse_improvement_ratio=naive_residual / jnp.maximum(
            trajectory_residual,
            jnp.asarray(1e-12, dtype=naive_residual.dtype),
        ),
        jit_inverse_rollout_delta=_rollout_delta(restored, jit_restored),
    )
