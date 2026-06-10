"""Loschmidt echo diagnostics for the production tick.

Stage 30 uses the Stage 28/29 inverse machinery as a stability probe.  A small
local perturbation is applied at the final time of a short production
trajectory, then the explicit inverse is used to map that perturbation back to
the initial surface.

This is a diagnostic observable for the existing discrete map.  It does not
change the production dynamics and does not claim an energy-convergent
integrator.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_reversibility import (
    sm_physical_right_production_inverse_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
)


class PhysicalRightProductionEchoDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 30 Loschmidt echo."""

    steps: jnp.ndarray
    perturbation_size: jnp.ndarray
    base_roundtrip_residual: jnp.ndarray
    apex_perturbation_norm: jnp.ndarray
    double_apex_perturbation_norm: jnp.ndarray
    echo_residual: jnp.ndarray
    double_echo_residual: jnp.ndarray
    echo_amplification: jnp.ndarray
    double_echo_ratio: jnp.ndarray
    double_echo_linearity_residual: jnp.ndarray
    perturbed_inverse_sm_link_unitarity_residual: jnp.ndarray
    perturbed_inverse_higgs_link_unitarity_residual: jnp.ndarray


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


def sm_physical_right_production_apex_momentum_perturbation(
    state: PhysicalRightProductionRolloutState,
    *,
    epsilon: float,
) -> PhysicalRightProductionRolloutState:
    """Perturb one final-time SM momentum coordinate."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    delta = jnp.asarray(epsilon, dtype=state.sm_momenta.dtype)
    perturbed_momenta = state.sm_momenta.at[0, 0, 0, 0, 0].add(delta)
    return state._replace(sm_momenta=perturbed_momenta)


def sm_physical_right_production_echo_diagnostics(
    *,
    steps: int = 2,
    step_size: float = 1e-3,
    epsilon: float = 1e-4,
) -> PhysicalRightProductionEchoDiagnostics:
    """Return Stage 30 Loschmidt echo diagnostics."""

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    initial = sm_physical_right_production_initial_state()
    final = sm_physical_right_production_rollout(initial, steps=steps, step_size=step_size)
    restored = sm_physical_right_production_inverse_rollout(final, steps=steps, step_size=step_size)

    perturbed_final = sm_physical_right_production_apex_momentum_perturbation(final, epsilon=epsilon)
    double_perturbed_final = sm_physical_right_production_apex_momentum_perturbation(final, epsilon=2.0 * epsilon)
    perturbed_restored = sm_physical_right_production_inverse_rollout(
        perturbed_final,
        steps=steps,
        step_size=step_size,
    )
    double_perturbed_restored = sm_physical_right_production_inverse_rollout(
        double_perturbed_final,
        steps=steps,
        step_size=step_size,
    )

    apex_perturbation_norm = _rollout_delta(final, perturbed_final)
    double_apex_perturbation_norm = _rollout_delta(final, double_perturbed_final)
    echo_residual = _rollout_delta(restored, perturbed_restored)
    double_echo_residual = _rollout_delta(restored, double_perturbed_restored)

    return PhysicalRightProductionEchoDiagnostics(
        steps=jnp.asarray(steps, dtype=jnp.int32),
        perturbation_size=jnp.asarray(epsilon, dtype=jnp.float32),
        base_roundtrip_residual=_rollout_delta(initial, restored),
        apex_perturbation_norm=apex_perturbation_norm,
        double_apex_perturbation_norm=double_apex_perturbation_norm,
        echo_residual=echo_residual,
        double_echo_residual=double_echo_residual,
        echo_amplification=echo_residual / jnp.maximum(
            apex_perturbation_norm,
            jnp.asarray(1e-12, dtype=echo_residual.dtype),
        ),
        double_echo_ratio=double_echo_residual / jnp.maximum(
            echo_residual,
            jnp.asarray(1e-12, dtype=echo_residual.dtype),
        ),
        double_echo_linearity_residual=jnp.abs(double_echo_residual - 2.0 * echo_residual)
        / jnp.maximum(echo_residual, jnp.asarray(1e-12, dtype=echo_residual.dtype)),
        perturbed_inverse_sm_link_unitarity_residual=sm_link_unitarity_residual(perturbed_restored.sm_links),
        perturbed_inverse_higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(perturbed_restored.higgs_links),
    )
