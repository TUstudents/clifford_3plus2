"""Local-force production rollout smoke test.

Stage 38 verifies that the physical-right production tick actually benefits
from the Stage 37 local Wilson-force replacement.  The certificate runs the
implemented production tick on a larger-than-one-site lattice and checks that
the former finite-difference Wilson ``epsilon`` knob no longer changes the
state, while links remain unitary and the rollout stays finite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    BCC_PLAQUETTE_PAIRS,
    SM_GENERATOR_COUNT,
    sm_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionLocalRolloutDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 38 local-force production rollout."""

    site_count: jnp.ndarray
    wilson_epsilon_invariance_residual: jnp.ndarray
    final_all_finite: jnp.ndarray
    family_norm_drift: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    higgs_delta_norm: jnp.ndarray
    sm_link_delta_norm: jnp.ndarray
    sm_momentum_delta_norm: jnp.ndarray
    local_force_plaquette_holonomies: jnp.ndarray
    legacy_force_plaquette_holonomies: jnp.ndarray
    finite_difference_to_local_work_ratio: jnp.ndarray


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


def _state_all_finite(state: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.all(
        jnp.asarray(
            [
                jnp.all(jnp.isfinite(state.family_state)),
                jnp.all(jnp.isfinite(state.higgs)),
                jnp.all(jnp.isfinite(state.higgs_momenta)),
                jnp.all(jnp.isfinite(state.sm_links)),
                jnp.all(jnp.isfinite(state.sm_momenta)),
                jnp.all(jnp.isfinite(state.higgs_links)),
            ],
        ),
    )


def sm_physical_right_production_local_rollout_diagnostics(
    lattice_shape: tuple[int, int, int] = (2, 2, 1),
) -> PhysicalRightProductionLocalRolloutDiagnostics:
    """Return Stage 38 local-force production rollout diagnostics."""

    if len(lattice_shape) != 3 or any(size < 1 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")

    initial = sm_physical_right_production_initial_state(lattice_shape)
    final = sm_physical_right_production_step(
        initial,
        step_size=0.001,
        wilson_epsilon=1e-3,
    )
    changed_epsilon = sm_physical_right_production_step(
        initial,
        step_size=0.001,
        wilson_epsilon=7e-3,
    )
    site_count = lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
    plaquette_count = len(BCC_PLAQUETTE_PAIRS)
    local_holonomies = site_count * plaquette_count
    legacy_holonomies = 2 * site_count * 8 * SM_GENERATOR_COUNT * local_holonomies

    return PhysicalRightProductionLocalRolloutDiagnostics(
        site_count=jnp.asarray(site_count, dtype=jnp.int32),
        wilson_epsilon_invariance_residual=_state_max_delta(final, changed_epsilon),
        final_all_finite=_state_all_finite(final),
        family_norm_drift=jnp.abs(state_norm_squared(final.family_state) - state_norm_squared(initial.family_state)),
        sm_link_unitarity_residual=sm_link_unitarity_residual(final.sm_links),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(final.higgs_links),
        higgs_delta_norm=jnp.linalg.norm(final.higgs - initial.higgs),
        sm_link_delta_norm=jnp.linalg.norm(final.sm_links - initial.sm_links),
        sm_momentum_delta_norm=jnp.linalg.norm(final.sm_momenta - initial.sm_momenta),
        local_force_plaquette_holonomies=jnp.asarray(local_holonomies, dtype=jnp.int32),
        legacy_force_plaquette_holonomies=jnp.asarray(legacy_holonomies, dtype=jnp.int32),
        finite_difference_to_local_work_ratio=jnp.asarray(legacy_holonomies / local_holonomies, dtype=jnp.float32),
    )

