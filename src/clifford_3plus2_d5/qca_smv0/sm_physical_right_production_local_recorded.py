"""Multi-step recorded rollout on the local-force production path.

Stage 39 checks the sparse recording layer after the Stage 37 local Wilson
force and Stage 38 one-step rollout smoke test.  It runs the actual
physical-right production tick for two steps on a ``2 x 2 x 1`` lattice and
compares the Python loop runner against the ``lax.scan`` runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_recorded_rollout,
)


class PhysicalRightProductionLocalRecordedDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 39 multi-step local-force recording."""

    site_count: jnp.ndarray
    step_count: jnp.ndarray
    record_count: jnp.ndarray
    loop_scan_final_residual: jnp.ndarray
    loop_scan_observation_residual: jnp.ndarray
    loop_all_finite: jnp.ndarray
    scan_all_finite: jnp.ndarray
    family_norm_drift: jnp.ndarray
    max_family_norm_drift: jnp.ndarray
    max_sm_link_unitarity_residual: jnp.ndarray
    max_higgs_link_unitarity_residual: jnp.ndarray
    higgs_field_total_delta_norm: jnp.ndarray
    sm_link_total_delta_norm: jnp.ndarray
    sm_momentum_total_delta_norm: jnp.ndarray


def _state_max_delta(
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


def _observation_max_delta(
    left: Mapping[str, jnp.ndarray],
    right: Mapping[str, jnp.ndarray],
) -> jnp.ndarray:
    if tuple(left.keys()) != tuple(right.keys()):
        raise ValueError("observation keys must match")
    return jnp.max(jnp.asarray([jnp.max(jnp.abs(left[key] - right[key])) for key in left]))


def sm_physical_right_production_local_recorded_diagnostics(
    lattice_shape: tuple[int, int, int] = (2, 2, 1),
    *,
    steps: int = 2,
) -> PhysicalRightProductionLocalRecordedDiagnostics:
    """Return Stage 39 multi-step local-force recorded-rollout diagnostics."""

    if len(lattice_shape) != 3 or any(size < 1 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    initial = sm_physical_right_production_initial_state(lattice_shape)
    loop_result = sm_physical_right_production_recorded_rollout(
        initial,
        steps=steps,
        record_every=1,
        step_size=0.001,
        use_scan=False,
    )
    scan_result = sm_physical_right_production_recorded_rollout(
        initial,
        steps=steps,
        record_every=1,
        step_size=0.001,
        use_scan=True,
        use_jit=False,
    )
    family_norms = scan_result.observations["family_norm"]
    initial_family_norm = family_norms[0]
    final_state = scan_result.final_state
    site_count = lattice_shape[0] * lattice_shape[1] * lattice_shape[2]

    return PhysicalRightProductionLocalRecordedDiagnostics(
        site_count=jnp.asarray(site_count, dtype=jnp.int32),
        step_count=jnp.asarray(steps, dtype=jnp.int32),
        record_count=jnp.asarray(scan_result.step_indices.shape[0], dtype=jnp.int32),
        loop_scan_final_residual=_state_max_delta(loop_result.final_state, scan_result.final_state),
        loop_scan_observation_residual=_observation_max_delta(loop_result.observations, scan_result.observations),
        loop_all_finite=loop_result.all_finite,
        scan_all_finite=scan_result.all_finite,
        family_norm_drift=jnp.abs(family_norms[-1] - initial_family_norm),
        max_family_norm_drift=jnp.max(jnp.abs(family_norms - initial_family_norm)),
        max_sm_link_unitarity_residual=jnp.max(scan_result.observations["sm_link_unitarity_residual"]),
        max_higgs_link_unitarity_residual=jnp.max(scan_result.observations["higgs_link_unitarity_residual"]),
        higgs_field_total_delta_norm=jnp.linalg.norm(final_state.higgs - initial.higgs),
        sm_link_total_delta_norm=jnp.linalg.norm(final_state.sm_links - initial.sm_links),
        sm_momentum_total_delta_norm=jnp.linalg.norm(final_state.sm_momenta - initial.sm_momenta),
    )

