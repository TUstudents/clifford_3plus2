"""Energy-refinement audit for the physical-right production tick.

Stage 26 tests the natural next question after the Stage 25 variational audit:
does the monitored total-energy drift improve when the production timestep is
halved at fixed physical time?

For the current hybrid quantum/classical production tick the answer is no.  The
audit records that limitation directly: the refined rollout remains finite,
norm-controlled, and link-unitary, but its monitored energy drift is larger
than the coarser rollout.  This stage therefore prevents the simulator from
overclaiming convergent Hamiltonian energy behavior before the production
integrator is redesigned.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_energy import (
    sm_physical_right_production_energy_history,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
)
from clifford_3plus2_d5.sim.observables import observations_all_finite


class PhysicalRightProductionRefinementDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 26 fixed-time timestep refinement."""

    total_time_match_residual: jnp.ndarray
    base_energy_drift_abs: jnp.ndarray
    refined_energy_drift_abs: jnp.ndarray
    drift_refinement_ratio: jnp.ndarray
    refinement_limitation_detected: jnp.ndarray
    refined_energy_drift_controlled: jnp.ndarray
    base_family_norm_drift: jnp.ndarray
    refined_family_norm_drift: jnp.ndarray
    refined_sm_link_unitarity_residual: jnp.ndarray
    refined_higgs_link_unitarity_residual: jnp.ndarray
    vacuum_refined_energy_drift_abs: jnp.ndarray
    histories_all_finite: jnp.ndarray


def _energy_drift(history: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.abs(history["total_energy"][-1] - history["total_energy"][0])


def _family_norm_drift(history: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.abs(history["family_norm"][-1] - history["family_norm"][0])


def sm_physical_right_production_refinement_pair(
    initial_state: PhysicalRightProductionRolloutState | None = None,
    *,
    base_step_size: float = 1e-3,
    base_steps: int = 2,
) -> tuple[
    PhysicalRightProductionRolloutState,
    dict[str, jnp.ndarray],
    PhysicalRightProductionRolloutState,
    dict[str, jnp.ndarray],
]:
    """Return coarse and half-step histories at the same physical time."""

    if base_steps <= 0:
        raise ValueError(f"base_steps must be positive, got {base_steps}")
    if base_step_size <= 0:
        raise ValueError(f"base_step_size must be positive, got {base_step_size}")
    initial = initial_state or sm_physical_right_production_initial_state()
    base_final, base_history = sm_physical_right_production_energy_history(
        initial,
        steps=base_steps,
        step_size=base_step_size,
    )
    refined_final, refined_history = sm_physical_right_production_energy_history(
        initial,
        steps=2 * base_steps,
        step_size=0.5 * base_step_size,
    )
    return base_final, base_history, refined_final, refined_history


def sm_physical_right_production_refinement_diagnostics() -> PhysicalRightProductionRefinementDiagnostics:
    """Return focused Stage 26 energy-refinement diagnostics."""

    base_step_size = 1e-3
    base_steps = 2
    refined_step_size = 0.5 * base_step_size
    refined_steps = 2 * base_steps
    base_time = jnp.asarray(base_steps * base_step_size, dtype=jnp.float32)
    refined_time = jnp.asarray(refined_steps * refined_step_size, dtype=jnp.float32)
    _, base_history, refined_final, refined_history = sm_physical_right_production_refinement_pair(
        base_step_size=base_step_size,
        base_steps=base_steps,
    )
    vacuum_initial = sm_physical_right_production_vacuum_state()
    _, vacuum_history = sm_physical_right_production_energy_history(
        vacuum_initial,
        steps=refined_steps,
        step_size=refined_step_size,
    )
    base_drift = _energy_drift(base_history)
    refined_drift = _energy_drift(refined_history)
    ratio = refined_drift / jnp.maximum(base_drift, jnp.asarray(1e-12, dtype=refined_drift.dtype))

    return PhysicalRightProductionRefinementDiagnostics(
        total_time_match_residual=jnp.abs(base_time - refined_time),
        base_energy_drift_abs=base_drift,
        refined_energy_drift_abs=refined_drift,
        drift_refinement_ratio=ratio,
        refinement_limitation_detected=refined_drift > 5.0 * base_drift,
        refined_energy_drift_controlled=refined_drift < jnp.asarray(5e-4, dtype=refined_drift.dtype),
        base_family_norm_drift=_family_norm_drift(base_history),
        refined_family_norm_drift=_family_norm_drift(refined_history),
        refined_sm_link_unitarity_residual=sm_link_unitarity_residual(refined_final.sm_links),
        refined_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(refined_final.higgs_links),
        vacuum_refined_energy_drift_abs=_energy_drift(vacuum_history),
        histories_all_finite=observations_all_finite(base_history)
        & observations_all_finite(refined_history)
        & observations_all_finite(vacuum_history),
    )
