"""Finite tangent-response audit for the production echo.

Stage 31 extends the Stage 30 Loschmidt echo from one final-time perturbation
to a small tangent basis.  Two independent final-time kicks are pulled back by
the explicit inverse, separately and together.  The combined pullback must
match the sum of the separate pullbacks to local finite-difference precision.

This is a diagnostic of the current discrete production map.  It is not a new
dynamics rule and does not claim an energy-convergent integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo import (
    sm_physical_right_production_apex_momentum_perturbation,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_reversibility import (
    sm_physical_right_production_inverse_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
)


class PhysicalRightProductionTangentDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 31 finite tangent response."""

    steps: jnp.ndarray
    perturbation_size: jnp.ndarray
    base_roundtrip_residual: jnp.ndarray
    sm_momentum_echo_norm: jnp.ndarray
    higgs_momentum_echo_norm: jnp.ndarray
    combined_echo_norm: jnp.ndarray
    superposition_residual: jnp.ndarray
    superposition_relative_residual: jnp.ndarray
    combined_inverse_sm_link_unitarity_residual: jnp.ndarray
    combined_inverse_higgs_link_unitarity_residual: jnp.ndarray


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


def _linear_combination_residual(
    base: PhysicalRightProductionRolloutState,
    sm_echo: PhysicalRightProductionRolloutState,
    higgs_echo: PhysicalRightProductionRolloutState,
    combined_echo: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    def field_residual(combined_field: jnp.ndarray, base_field: jnp.ndarray, sm_field: jnp.ndarray, higgs_field: jnp.ndarray) -> jnp.ndarray:
        return jnp.max(
            jnp.abs(
                (combined_field - base_field)
                - (sm_field - base_field)
                - (higgs_field - base_field),
            ),
        )

    return jnp.max(
        jnp.asarray(
            [
                field_residual(combined_echo.family_state, base.family_state, sm_echo.family_state, higgs_echo.family_state),
                field_residual(combined_echo.higgs, base.higgs, sm_echo.higgs, higgs_echo.higgs),
                field_residual(combined_echo.higgs_momenta, base.higgs_momenta, sm_echo.higgs_momenta, higgs_echo.higgs_momenta),
                field_residual(combined_echo.sm_links, base.sm_links, sm_echo.sm_links, higgs_echo.sm_links),
                field_residual(combined_echo.sm_momenta, base.sm_momenta, sm_echo.sm_momenta, higgs_echo.sm_momenta),
                field_residual(combined_echo.higgs_links, base.higgs_links, sm_echo.higgs_links, higgs_echo.higgs_links),
            ],
        ),
    )


def sm_physical_right_production_apex_higgs_momentum_perturbation(
    state: PhysicalRightProductionRolloutState,
    *,
    epsilon: float,
) -> PhysicalRightProductionRolloutState:
    """Perturb one final-time Higgs momentum coordinate."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    delta = jnp.asarray(epsilon, dtype=state.higgs_momenta.dtype)
    perturbed_momenta = state.higgs_momenta.at[0, 0, 0, 0].add(delta)
    return state._replace(higgs_momenta=perturbed_momenta)


def sm_physical_right_production_tangent_diagnostics(
    *,
    steps: int = 2,
    step_size: float = 1e-3,
    epsilon: float = 1e-4,
) -> PhysicalRightProductionTangentDiagnostics:
    """Return Stage 31 finite tangent-response diagnostics."""

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    initial = sm_physical_right_production_initial_state()
    final = sm_physical_right_production_rollout(initial, steps=steps, step_size=step_size)
    restored = sm_physical_right_production_inverse_rollout(final, steps=steps, step_size=step_size)

    sm_final = sm_physical_right_production_apex_momentum_perturbation(final, epsilon=epsilon)
    higgs_final = sm_physical_right_production_apex_higgs_momentum_perturbation(final, epsilon=epsilon)
    combined_final = sm_physical_right_production_apex_higgs_momentum_perturbation(sm_final, epsilon=epsilon)

    sm_echo = sm_physical_right_production_inverse_rollout(sm_final, steps=steps, step_size=step_size)
    higgs_echo = sm_physical_right_production_inverse_rollout(higgs_final, steps=steps, step_size=step_size)
    combined_echo = sm_physical_right_production_inverse_rollout(combined_final, steps=steps, step_size=step_size)

    sm_norm = _rollout_delta(restored, sm_echo)
    higgs_norm = _rollout_delta(restored, higgs_echo)
    combined_norm = _rollout_delta(restored, combined_echo)
    superposition_residual = _linear_combination_residual(restored, sm_echo, higgs_echo, combined_echo)

    return PhysicalRightProductionTangentDiagnostics(
        steps=jnp.asarray(steps, dtype=jnp.int32),
        perturbation_size=jnp.asarray(epsilon, dtype=jnp.float32),
        base_roundtrip_residual=_rollout_delta(initial, restored),
        sm_momentum_echo_norm=sm_norm,
        higgs_momentum_echo_norm=higgs_norm,
        combined_echo_norm=combined_norm,
        superposition_residual=superposition_residual,
        superposition_relative_residual=superposition_residual
        / jnp.maximum(combined_norm, jnp.asarray(1e-12, dtype=combined_norm.dtype)),
        combined_inverse_sm_link_unitarity_residual=sm_link_unitarity_residual(combined_echo.sm_links),
        combined_inverse_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(combined_echo.higgs_links),
    )
