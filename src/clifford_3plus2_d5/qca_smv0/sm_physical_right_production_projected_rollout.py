"""Finite rollout of the Gauss-projected production step.

Stage 49 iterates the Stage 48 projected tick over a short finite horizon and
records raw-vs-projected Gauss histories.  The projected trajectory advances by

    raw production tick -> fixed-link SM-momentum Gauss relaxation

at every step.  A raw production trajectory from the same initial state is
recorded alongside it as the comparison baseline.

This is still a wrapper-level constraint treatment.  It does not turn the
production tick into a Gauss-preserving integrator; it only checks that the
explicit post-step projection can be iterated and keeps the finite-horizon
Gauss residual below the raw rollout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_solver import (
    sm_physical_right_production_gauss_relaxation_solve,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)


class PhysicalRightProductionProjectedRolloutHistory(NamedTuple):
    """Recorded Gauss data for raw and projected finite rollouts."""

    raw_gauss_norms: jnp.ndarray
    projected_gauss_norms: jnp.ndarray
    step_post_tick_gauss_norms: jnp.ndarray
    step_projected_gauss_norms: jnp.ndarray
    step_projection_reduction_fractions: jnp.ndarray
    step_history_monotonicity_violations: jnp.ndarray
    step_history_final_residuals: jnp.ndarray
    step_projection_momentum_delta_norms: jnp.ndarray
    projected_sm_link_unitarity_residuals: jnp.ndarray
    projected_higgs_link_unitarity_residuals: jnp.ndarray


class PhysicalRightProductionProjectedRolloutDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 49 projected production rollout."""

    step_count: jnp.ndarray
    projection_iteration_count: jnp.ndarray
    vacuum_initial_gauss_norm: jnp.ndarray
    vacuum_final_gauss_norm: jnp.ndarray
    vacuum_momentum_delta_norm: jnp.ndarray
    initial_gauss_norm: jnp.ndarray
    raw_final_gauss_norm: jnp.ndarray
    projected_final_gauss_norm: jnp.ndarray
    final_gauss_reduction_vs_raw_norm: jnp.ndarray
    final_gauss_reduction_vs_raw_fraction: jnp.ndarray
    min_step_projection_reduction_fraction: jnp.ndarray
    min_step_projection_momentum_delta_norm: jnp.ndarray
    max_step_history_monotonicity_violation: jnp.ndarray
    max_step_history_final_residual: jnp.ndarray
    max_projected_sm_link_unitarity_residual: jnp.ndarray
    max_projected_higgs_link_unitarity_residual: jnp.ndarray
    history_all_finite: jnp.ndarray


def _gauss_norm(state: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.linalg.norm(sm_physical_right_production_gauss(state))


def _arrays_all_finite(history: PhysicalRightProductionProjectedRolloutHistory) -> jnp.ndarray:
    checks = [jnp.all(jnp.isfinite(array)) for array in history]
    return jnp.all(jnp.asarray(checks))


def sm_physical_right_production_projected_rollout(
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
    projection_iterations: int = 10,
    projection_relaxation_scale: float = 1.0,
) -> tuple[PhysicalRightProductionRolloutState, PhysicalRightProductionProjectedRolloutHistory]:
    """Return a finite rollout built from the Stage 48 projected step."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")

    raw_state = initial_state
    projected_state = initial_state
    raw_gauss_norms = [_gauss_norm(raw_state)]
    projected_gauss_norms = [_gauss_norm(projected_state)]
    step_post_tick_gauss_norms = []
    step_projected_gauss_norms = []
    step_projection_reduction_fractions = []
    step_history_monotonicity_violations = []
    step_history_final_residuals = []
    step_projection_momentum_delta_norms = []
    projected_sm_link_unitarity_residuals = [sm_link_unitarity_residual(projected_state.sm_links)]
    projected_higgs_link_unitarity_residuals = [sm_higgs_link_unitarity_residual(projected_state.higgs_links)]

    for _ in range(steps):
        raw_state = sm_physical_right_production_step(
            raw_state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
        raw_gauss_norms.append(_gauss_norm(raw_state))

        unprojected = sm_physical_right_production_step(
            projected_state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
        projected_state, relaxation_history = sm_physical_right_production_gauss_relaxation_solve(
            unprojected,
            iterations=projection_iterations,
            relaxation_scale=projection_relaxation_scale,
        )
        post_tick_norm = relaxation_history.gauss_norms[0]
        projected_norm = relaxation_history.gauss_norms[-1]
        reduction = post_tick_norm - projected_norm
        step_post_tick_gauss_norms.append(post_tick_norm)
        step_projected_gauss_norms.append(projected_norm)
        step_projection_reduction_fractions.append(
            jnp.where(post_tick_norm > 0, reduction / post_tick_norm, 0.0),
        )
        step_history_monotonicity_violations.append(
            jnp.max(
                jnp.maximum(
                    relaxation_history.gauss_norms[1:] - relaxation_history.gauss_norms[:-1],
                    0.0,
                ),
            ),
        )
        step_history_final_residuals.append(jnp.abs(projected_norm - _gauss_norm(projected_state)))
        step_projection_momentum_delta_norms.append(
            jnp.linalg.norm(projected_state.sm_momenta - unprojected.sm_momenta),
        )
        projected_gauss_norms.append(_gauss_norm(projected_state))
        projected_sm_link_unitarity_residuals.append(sm_link_unitarity_residual(projected_state.sm_links))
        projected_higgs_link_unitarity_residuals.append(sm_higgs_link_unitarity_residual(projected_state.higgs_links))

    return projected_state, PhysicalRightProductionProjectedRolloutHistory(
        raw_gauss_norms=jnp.asarray(raw_gauss_norms),
        projected_gauss_norms=jnp.asarray(projected_gauss_norms),
        step_post_tick_gauss_norms=jnp.asarray(step_post_tick_gauss_norms),
        step_projected_gauss_norms=jnp.asarray(step_projected_gauss_norms),
        step_projection_reduction_fractions=jnp.asarray(step_projection_reduction_fractions),
        step_history_monotonicity_violations=jnp.asarray(step_history_monotonicity_violations),
        step_history_final_residuals=jnp.asarray(step_history_final_residuals),
        step_projection_momentum_delta_norms=jnp.asarray(step_projection_momentum_delta_norms),
        projected_sm_link_unitarity_residuals=jnp.asarray(projected_sm_link_unitarity_residuals),
        projected_higgs_link_unitarity_residuals=jnp.asarray(projected_higgs_link_unitarity_residuals),
    )


def sm_physical_right_production_projected_rollout_diagnostics() -> (
    PhysicalRightProductionProjectedRolloutDiagnostics
):
    """Return focused Stage 49 projected-rollout diagnostics."""

    steps = 2
    projection_iterations = 10
    step_size = 0.001

    vacuum = sm_physical_right_production_vacuum_state()
    vacuum_projected, vacuum_history = sm_physical_right_production_projected_rollout(
        vacuum,
        steps=steps,
        step_size=step_size,
        projection_iterations=projection_iterations,
    )
    initial = sm_physical_right_production_initial_state()
    _, history = sm_physical_right_production_projected_rollout(
        initial,
        steps=steps,
        step_size=step_size,
        projection_iterations=projection_iterations,
    )

    raw_final = history.raw_gauss_norms[-1]
    projected_final = history.projected_gauss_norms[-1]
    final_reduction = raw_final - projected_final
    return PhysicalRightProductionProjectedRolloutDiagnostics(
        step_count=jnp.asarray(steps, dtype=jnp.int32),
        projection_iteration_count=jnp.asarray(projection_iterations, dtype=jnp.int32),
        vacuum_initial_gauss_norm=vacuum_history.projected_gauss_norms[0],
        vacuum_final_gauss_norm=vacuum_history.projected_gauss_norms[-1],
        vacuum_momentum_delta_norm=jnp.linalg.norm(vacuum_projected.sm_momenta - vacuum.sm_momenta),
        initial_gauss_norm=history.projected_gauss_norms[0],
        raw_final_gauss_norm=raw_final,
        projected_final_gauss_norm=projected_final,
        final_gauss_reduction_vs_raw_norm=final_reduction,
        final_gauss_reduction_vs_raw_fraction=jnp.where(raw_final > 0, final_reduction / raw_final, 0.0),
        min_step_projection_reduction_fraction=jnp.min(history.step_projection_reduction_fractions),
        min_step_projection_momentum_delta_norm=jnp.min(history.step_projection_momentum_delta_norms),
        max_step_history_monotonicity_violation=jnp.max(history.step_history_monotonicity_violations),
        max_step_history_final_residual=jnp.max(history.step_history_final_residuals),
        max_projected_sm_link_unitarity_residual=jnp.max(history.projected_sm_link_unitarity_residuals),
        max_projected_higgs_link_unitarity_residual=jnp.max(history.projected_higgs_link_unitarity_residuals),
        history_all_finite=_arrays_all_finite(vacuum_history) & _arrays_all_finite(history),
    )
