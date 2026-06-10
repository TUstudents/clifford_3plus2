"""Gauss-projected wrapper for the physical-right production tick.

Stage 48 composes the current production step with the finite Stage 47
momentum-only Gauss relaxation solver.  This gives an explicit projected tick:

    raw production step -> fixed-link SM-momentum Gauss relaxation.

It is deliberately a wrapper, not a rewrite of the production integrator.  The
projection only adjusts SM link momenta after the tick; matter fields, Higgs
fields, Higgs momenta, and links are left as produced by the unprojected tick.
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
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_gauss,
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss_solver import (
    PhysicalRightProductionGaussRelaxationHistory,
    sm_physical_right_production_gauss_relaxation_solve,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)


class PhysicalRightProductionProjectedStepDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 48 projected production step."""

    projection_iteration_count: jnp.ndarray
    vacuum_unprojected_gauss_norm: jnp.ndarray
    vacuum_projected_gauss_norm: jnp.ndarray
    vacuum_projection_momentum_delta_norm: jnp.ndarray
    unprojected_gauss_norm: jnp.ndarray
    projected_gauss_norm: jnp.ndarray
    gauss_reduction_norm: jnp.ndarray
    gauss_reduction_fraction: jnp.ndarray
    history_monotonicity_violation: jnp.ndarray
    history_final_gauss_residual: jnp.ndarray
    min_history_step_reduction_norm: jnp.ndarray
    projection_momentum_delta_norm: jnp.ndarray
    family_state_delta_norm: jnp.ndarray
    higgs_delta_norm: jnp.ndarray
    higgs_momentum_delta_norm: jnp.ndarray
    sm_link_delta_norm: jnp.ndarray
    higgs_link_delta_norm: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    jit_delta_projected_sm_momenta: jnp.ndarray
    jit_delta_projected_history: jnp.ndarray


def _gauss_norm(state: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.linalg.norm(sm_physical_right_production_gauss(state))


def sm_physical_right_production_projected_step(
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
    projection_iterations: int = 10,
    projection_relaxation_scale: float = 1.0,
) -> tuple[PhysicalRightProductionRolloutState, PhysicalRightProductionGaussRelaxationHistory]:
    """Advance one production tick and relax the result toward Gauss."""

    unprojected = sm_physical_right_production_step(
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
    return sm_physical_right_production_gauss_relaxation_solve(
        unprojected,
        iterations=projection_iterations,
        relaxation_scale=projection_relaxation_scale,
    )


def sm_physical_right_production_projected_step_diagnostics() -> PhysicalRightProductionProjectedStepDiagnostics:
    """Return focused Stage 48 projected-step diagnostics."""

    step_size = 0.001
    projection_iterations = 10

    vacuum = sm_physical_right_production_vacuum_state()
    vacuum_unprojected = sm_physical_right_production_step(vacuum, step_size=step_size)
    vacuum_projected, _ = sm_physical_right_production_projected_step(
        vacuum,
        step_size=step_size,
        projection_iterations=projection_iterations,
    )

    initial = sm_physical_right_production_initial_state()
    unprojected = sm_physical_right_production_step(initial, step_size=step_size)
    projected, history = sm_physical_right_production_projected_step(
        initial,
        step_size=step_size,
        projection_iterations=projection_iterations,
    )
    jitted_projected_step = jax.jit(
        sm_physical_right_production_projected_step,
        static_argnames=("step_size", "projection_iterations"),
    )
    jit_projected, jit_history = jitted_projected_step(
        initial,
        step_size=step_size,
        projection_iterations=projection_iterations,
    )

    unprojected_norm = _gauss_norm(unprojected)
    projected_norm = _gauss_norm(projected)
    gauss_reduction = unprojected_norm - projected_norm
    history_reductions = history.gauss_norms[:-1] - history.gauss_norms[1:]

    return PhysicalRightProductionProjectedStepDiagnostics(
        projection_iteration_count=jnp.asarray(projection_iterations, dtype=jnp.int32),
        vacuum_unprojected_gauss_norm=_gauss_norm(vacuum_unprojected),
        vacuum_projected_gauss_norm=_gauss_norm(vacuum_projected),
        vacuum_projection_momentum_delta_norm=jnp.linalg.norm(
            vacuum_projected.sm_momenta - vacuum_unprojected.sm_momenta,
        ),
        unprojected_gauss_norm=unprojected_norm,
        projected_gauss_norm=projected_norm,
        gauss_reduction_norm=gauss_reduction,
        gauss_reduction_fraction=jnp.where(unprojected_norm > 0, gauss_reduction / unprojected_norm, 0.0),
        history_monotonicity_violation=jnp.max(
            jnp.maximum(history.gauss_norms[1:] - history.gauss_norms[:-1], 0.0),
        ),
        history_final_gauss_residual=jnp.abs(history.gauss_norms[-1] - projected_norm),
        min_history_step_reduction_norm=jnp.min(history_reductions),
        projection_momentum_delta_norm=jnp.linalg.norm(projected.sm_momenta - unprojected.sm_momenta),
        family_state_delta_norm=jnp.linalg.norm(projected.family_state - unprojected.family_state),
        higgs_delta_norm=jnp.linalg.norm(projected.higgs - unprojected.higgs),
        higgs_momentum_delta_norm=jnp.linalg.norm(projected.higgs_momenta - unprojected.higgs_momenta),
        sm_link_delta_norm=jnp.linalg.norm(projected.sm_links - unprojected.sm_links),
        higgs_link_delta_norm=jnp.linalg.norm(projected.higgs_links - unprojected.higgs_links),
        sm_link_unitarity_residual=sm_link_unitarity_residual(projected.sm_links),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(projected.higgs_links),
        jit_delta_projected_sm_momenta=jnp.max(jnp.abs(jit_projected.sm_momenta - projected.sm_momenta)),
        jit_delta_projected_history=jnp.max(jnp.abs(jit_history.gauss_norms - history.gauss_norms)),
    )
