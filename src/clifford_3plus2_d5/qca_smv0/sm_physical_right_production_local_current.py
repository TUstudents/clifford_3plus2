"""Local physical-right fermion-current rollout audit.

Stage 42 removes the last dense finite-difference current from the
physical-right production tick.  Stage 37 made the Wilson force local; this
stage does the same for the physical-right BCC streaming fermion current and
keeps the legacy finite-difference current only as a one-site oracle.

This is an implementation stage, not a new dynamics rule.  The analytic
current is the exact local derivative of the already-used physical-right
streaming bilinear.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_gauge import deterministic_sm_link_theta, sm_link_field_from_algebra
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_finite_difference_physical_right_fermion_left_gauge_current,
    sm_local_physical_right_fermion_left_gauge_current,
    sm_physical_right_fermion_left_gauge_current,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import (
    sm_physical_right_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionLocalCurrentDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 42 local fermion-current production use."""

    oracle_site_count: jnp.ndarray
    production_site_count: jnp.ndarray
    finite_difference_current_norm: jnp.ndarray
    local_current_norm: jnp.ndarray
    finite_difference_local_max_residual: jnp.ndarray
    finite_difference_local_norm_residual: jnp.ndarray
    production_local_alias_residual: jnp.ndarray
    zero_state_local_current_norm: jnp.ndarray
    production_fermion_epsilon_residual: jnp.ndarray
    production_family_norm_drift: jnp.ndarray
    max_sm_link_unitarity_residual: jnp.ndarray
    max_higgs_link_unitarity_residual: jnp.ndarray
    legacy_global_streaming_term_count: jnp.ndarray
    local_streaming_term_count: jnp.ndarray
    estimated_work_reduction_ratio: jnp.ndarray


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


def _site_count(lattice_shape: tuple[int, int, int]) -> int:
    if len(lattice_shape) != 3 or any(size < 1 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")
    return lattice_shape[0] * lattice_shape[1] * lattice_shape[2]


def _legacy_streaming_term_count(site_count: int) -> int:
    edge_count = 8
    generator_count = 12
    finite_difference_sides = 2
    global_terms_per_energy = site_count * edge_count
    coordinate_count = site_count * edge_count * generator_count
    return finite_difference_sides * coordinate_count * global_terms_per_energy


def _local_streaming_term_count(site_count: int) -> int:
    edge_count = 8
    generator_count = 12
    return site_count * edge_count * generator_count


def sm_physical_right_production_local_current_diagnostics(
    *,
    oracle_lattice_shape: tuple[int, int, int] = (1, 1, 1),
    production_lattice_shape: tuple[int, int, int] = (2, 2, 1),
    step_size: float = 1e-3,
) -> PhysicalRightProductionLocalCurrentDiagnostics:
    """Return Stage 42 local-current rollout diagnostics."""

    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    oracle_sites = _site_count(oracle_lattice_shape)
    production_sites = _site_count(production_lattice_shape)

    oracle_state = deterministic_sm_family_state(oracle_lattice_shape)
    zero_oracle_state = jnp.zeros_like(oracle_state)
    oracle_links = sm_link_field_from_algebra(deterministic_sm_link_theta(oracle_lattice_shape, scale=0.24))
    finite_difference_current = sm_finite_difference_physical_right_fermion_left_gauge_current(
        oracle_state,
        oracle_links,
    )
    local_current = sm_local_physical_right_fermion_left_gauge_current(
        oracle_state,
        oracle_links,
    )
    production_alias_current = sm_physical_right_fermion_left_gauge_current(
        oracle_state,
        oracle_links,
        epsilon=0.07,
    )
    zero_current = sm_local_physical_right_fermion_left_gauge_current(
        zero_oracle_state,
        oracle_links,
    )

    initial = sm_physical_right_production_initial_state(production_lattice_shape)
    baseline = sm_physical_right_production_step(
        initial,
        step_size=step_size,
        fermion_current_epsilon=0.03,
    )
    changed_epsilon = sm_physical_right_production_step(
        initial,
        step_size=step_size,
        fermion_current_epsilon=0.09,
    )
    legacy_terms = _legacy_streaming_term_count(production_sites)
    local_terms = _local_streaming_term_count(production_sites)

    return PhysicalRightProductionLocalCurrentDiagnostics(
        oracle_site_count=jnp.asarray(oracle_sites, dtype=jnp.int32),
        production_site_count=jnp.asarray(production_sites, dtype=jnp.int32),
        finite_difference_current_norm=jnp.linalg.norm(finite_difference_current),
        local_current_norm=jnp.linalg.norm(local_current),
        finite_difference_local_max_residual=jnp.max(jnp.abs(finite_difference_current - local_current)),
        finite_difference_local_norm_residual=jnp.linalg.norm(finite_difference_current - local_current),
        production_local_alias_residual=jnp.max(jnp.abs(production_alias_current - local_current)),
        zero_state_local_current_norm=jnp.linalg.norm(zero_current),
        production_fermion_epsilon_residual=_max_state_delta(baseline, changed_epsilon),
        production_family_norm_drift=jnp.abs(state_norm_squared(baseline.family_state) - state_norm_squared(initial.family_state)),
        max_sm_link_unitarity_residual=sm_physical_right_link_unitarity_residual(baseline.sm_links),
        max_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(baseline.higgs_links),
        legacy_global_streaming_term_count=jnp.asarray(legacy_terms, dtype=jnp.int32),
        local_streaming_term_count=jnp.asarray(local_terms, dtype=jnp.int32),
        estimated_work_reduction_ratio=jnp.asarray(legacy_terms / local_terms, dtype=jnp.float32),
    )
