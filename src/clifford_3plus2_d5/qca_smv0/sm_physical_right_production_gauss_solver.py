"""Iterated Gauss relaxation for the production rollout state.

Stage 47 turns the Stage 46 one-step Gauss relaxation into a tiny solver with
an explicit residual history.  It repeatedly applies the fixed-link,
momentum-only exact-line relaxation and records the Gauss norm after each
iteration.

This is still a projection precursor: it only adjusts SM link momenta for the
current frozen production state.  It does not alter links, matter fields, Higgs
fields, or the production tick, and it does not claim a Gauss-preserving
integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_projection import (
    sm_physical_right_production_gauss_relaxation_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
)


class PhysicalRightProductionGaussRelaxationHistory(NamedTuple):
    """Recorded residual data for iterated Gauss relaxation."""

    gauss_norms: jnp.ndarray
    line_steps: jnp.ndarray
    gradient_norms: jnp.ndarray


class PhysicalRightProductionGaussSolverDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 47 iterated Gauss relaxation."""

    iteration_count: jnp.ndarray
    vacuum_initial_gauss_norm: jnp.ndarray
    vacuum_final_gauss_norm: jnp.ndarray
    vacuum_momentum_delta_norm: jnp.ndarray
    initial_gauss_norm: jnp.ndarray
    final_gauss_norm: jnp.ndarray
    total_gauss_reduction_norm: jnp.ndarray
    total_gauss_reduction_fraction: jnp.ndarray
    monotonicity_violation: jnp.ndarray
    min_step_reduction_norm: jnp.ndarray
    min_line_step: jnp.ndarray
    max_line_step: jnp.ndarray
    min_gradient_norm: jnp.ndarray
    max_gradient_norm: jnp.ndarray
    total_momentum_delta_norm: jnp.ndarray
    family_state_delta_norm: jnp.ndarray
    higgs_delta_norm: jnp.ndarray
    higgs_momentum_delta_norm: jnp.ndarray
    sm_link_delta_norm: jnp.ndarray
    higgs_link_delta_norm: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    jit_delta_momenta: jnp.ndarray
    jit_delta_history: jnp.ndarray


def _gauss_norm(state: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.linalg.norm(sm_physical_right_production_gauss(state))


def sm_physical_right_production_gauss_relaxation_solve(
    state: PhysicalRightProductionRolloutState,
    *,
    iterations: int = 4,
    relaxation_scale: float = 1.0,
) -> tuple[PhysicalRightProductionRolloutState, PhysicalRightProductionGaussRelaxationHistory]:
    """Apply repeated Stage 46 Gauss relaxation steps and record history."""

    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if relaxation_scale <= 0:
        raise ValueError(f"relaxation_scale must be positive, got {relaxation_scale}")

    current = state
    norms = [_gauss_norm(current)]
    line_steps = []
    gradient_norms = []
    for _ in range(iterations):
        current, line_step, gradient = sm_physical_right_production_gauss_relaxation_step(
            current,
            relaxation_scale=relaxation_scale,
        )
        norms.append(_gauss_norm(current))
        line_steps.append(line_step)
        gradient_norms.append(jnp.linalg.norm(gradient))
    return current, PhysicalRightProductionGaussRelaxationHistory(
        gauss_norms=jnp.asarray(norms),
        line_steps=jnp.asarray(line_steps),
        gradient_norms=jnp.asarray(gradient_norms),
    )


def sm_physical_right_production_gauss_solver_diagnostics() -> PhysicalRightProductionGaussSolverDiagnostics:
    """Return focused Stage 47 iterated Gauss relaxation diagnostics."""

    iterations = 10
    vacuum = sm_physical_right_production_vacuum_state()
    relaxed_vacuum, vacuum_history = sm_physical_right_production_gauss_relaxation_solve(
        vacuum,
        iterations=iterations,
    )
    initial = sm_physical_right_production_initial_state()
    relaxed, history = sm_physical_right_production_gauss_relaxation_solve(
        initial,
        iterations=iterations,
    )
    jitted_solve = jax.jit(
        sm_physical_right_production_gauss_relaxation_solve,
        static_argnames=("iterations",),
    )
    jit_relaxed, jit_history = jitted_solve(initial, iterations=iterations)

    reductions = history.gauss_norms[:-1] - history.gauss_norms[1:]
    initial_norm = history.gauss_norms[0]
    final_norm = history.gauss_norms[-1]
    total_reduction = initial_norm - final_norm
    total_fraction = jnp.where(initial_norm > 0, total_reduction / initial_norm, 0.0)

    return PhysicalRightProductionGaussSolverDiagnostics(
        iteration_count=jnp.asarray(iterations, dtype=jnp.int32),
        vacuum_initial_gauss_norm=vacuum_history.gauss_norms[0],
        vacuum_final_gauss_norm=vacuum_history.gauss_norms[-1],
        vacuum_momentum_delta_norm=jnp.linalg.norm(relaxed_vacuum.sm_momenta - vacuum.sm_momenta),
        initial_gauss_norm=initial_norm,
        final_gauss_norm=final_norm,
        total_gauss_reduction_norm=total_reduction,
        total_gauss_reduction_fraction=total_fraction,
        monotonicity_violation=jnp.max(jnp.maximum(history.gauss_norms[1:] - history.gauss_norms[:-1], 0.0)),
        min_step_reduction_norm=jnp.min(reductions),
        min_line_step=jnp.min(history.line_steps),
        max_line_step=jnp.max(history.line_steps),
        min_gradient_norm=jnp.min(history.gradient_norms),
        max_gradient_norm=jnp.max(history.gradient_norms),
        total_momentum_delta_norm=jnp.linalg.norm(relaxed.sm_momenta - initial.sm_momenta),
        family_state_delta_norm=jnp.linalg.norm(relaxed.family_state - initial.family_state),
        higgs_delta_norm=jnp.linalg.norm(relaxed.higgs - initial.higgs),
        higgs_momentum_delta_norm=jnp.linalg.norm(relaxed.higgs_momenta - initial.higgs_momenta),
        sm_link_delta_norm=jnp.linalg.norm(relaxed.sm_links - initial.sm_links),
        higgs_link_delta_norm=jnp.linalg.norm(relaxed.higgs_links - initial.higgs_links),
        sm_link_unitarity_residual=sm_link_unitarity_residual(relaxed.sm_links),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(relaxed.higgs_links),
        jit_delta_momenta=jnp.max(jnp.abs(jit_relaxed.sm_momenta - relaxed.sm_momenta)),
        jit_delta_history=jnp.max(jnp.abs(jit_history.gauss_norms - history.gauss_norms)),
    )
