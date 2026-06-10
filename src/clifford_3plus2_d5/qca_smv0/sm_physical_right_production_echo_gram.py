"""Echo Gram diagnostics for the production tangent map.

Stage 32 measures a small local tangent metric.  Several independent
final-time perturbations are pulled back by the explicit inverse trajectory,
and their initial-surface echoes are assembled into a real Gram matrix.

This is a diagnostic of the current discrete production map.  It is not a new
dynamics rule and does not claim a continuum stability theorem.
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
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tangent import (
    sm_physical_right_production_apex_higgs_momentum_perturbation,
)


class PhysicalRightProductionEchoGramDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 32 echo Gram matrix."""

    basis_count: jnp.ndarray
    perturbation_size: jnp.ndarray
    base_roundtrip_residual: jnp.ndarray
    min_echo_norm: jnp.ndarray
    max_echo_norm: jnp.ndarray
    gram_symmetry_residual: jnp.ndarray
    gram_min_eigenvalue: jnp.ndarray
    gram_max_eigenvalue: jnp.ndarray
    gram_condition_number: jnp.ndarray
    max_offdiag_correlation: jnp.ndarray
    max_inverse_sm_link_unitarity_residual: jnp.ndarray
    max_inverse_higgs_link_unitarity_residual: jnp.ndarray


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


def _state_delta_fields(
    state: PhysicalRightProductionRolloutState,
    base: PhysicalRightProductionRolloutState,
) -> tuple[jnp.ndarray, ...]:
    return (
        state.family_state - base.family_state,
        state.higgs - base.higgs,
        state.higgs_momenta - base.higgs_momenta,
        state.sm_links - base.sm_links,
        state.sm_momenta - base.sm_momenta,
        state.higgs_links - base.higgs_links,
    )


def _state_inner(left: tuple[jnp.ndarray, ...], right: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    return jnp.sum(jnp.asarray([jnp.real(jnp.vdot(a, b)) for a, b in zip(left, right, strict=True)]))


def sm_physical_right_production_apex_family_perturbation(
    state: PhysicalRightProductionRolloutState,
    *,
    epsilon: float,
) -> PhysicalRightProductionRolloutState:
    """Perturb one final-time family-state component."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    delta = jnp.asarray(epsilon, dtype=state.family_state.real.dtype).astype(state.family_state.dtype)
    perturbed_family = state.family_state.at[0, 0, 0, 0, 0, 0].add(delta)
    return state._replace(family_state=perturbed_family)


def sm_physical_right_production_echo_gram_diagnostics(
    *,
    steps: int = 2,
    step_size: float = 1e-3,
    epsilon: float = 1e-4,
) -> PhysicalRightProductionEchoGramDiagnostics:
    """Return Stage 32 local echo-Gram diagnostics."""

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    initial = sm_physical_right_production_initial_state()
    final = sm_physical_right_production_rollout(initial, steps=steps, step_size=step_size)
    restored = sm_physical_right_production_inverse_rollout(final, steps=steps, step_size=step_size)

    perturbed_finals = (
        sm_physical_right_production_apex_momentum_perturbation(final, epsilon=epsilon),
        sm_physical_right_production_apex_higgs_momentum_perturbation(final, epsilon=epsilon),
        sm_physical_right_production_apex_family_perturbation(final, epsilon=epsilon),
    )
    echo_states = tuple(
        sm_physical_right_production_inverse_rollout(perturbed, steps=steps, step_size=step_size)
        for perturbed in perturbed_finals
    )
    echo_deltas = tuple(_state_delta_fields(echo, restored) for echo in echo_states)
    gram = jnp.stack(
        [
            jnp.stack([_state_inner(left, right) for right in echo_deltas])
            for left in echo_deltas
        ],
    )
    gram = 0.5 * (gram + gram.T)
    diag = jnp.diag(gram)
    echo_norms = jnp.sqrt(jnp.maximum(diag, jnp.asarray(0.0, dtype=diag.dtype)))
    eigenvalues = jnp.linalg.eigvalsh(gram)
    denom = jnp.sqrt(jnp.maximum(diag[:, None] * diag[None, :], jnp.asarray(1e-30, dtype=diag.dtype)))
    corr = jnp.abs(gram) / denom
    offdiag = corr - jnp.diag(jnp.diag(corr))
    sm_unitarity = jnp.max(jnp.asarray([sm_link_unitarity_residual(echo.sm_links) for echo in echo_states]))
    higgs_unitarity = jnp.max(jnp.asarray([sm_higgs_link_unitarity_residual(echo.higgs_links) for echo in echo_states]))

    return PhysicalRightProductionEchoGramDiagnostics(
        basis_count=jnp.asarray(len(echo_states), dtype=jnp.int32),
        perturbation_size=jnp.asarray(epsilon, dtype=jnp.float32),
        base_roundtrip_residual=_rollout_delta(initial, restored),
        min_echo_norm=jnp.min(echo_norms),
        max_echo_norm=jnp.max(echo_norms),
        gram_symmetry_residual=jnp.max(jnp.abs(gram - gram.T)),
        gram_min_eigenvalue=jnp.min(eigenvalues),
        gram_max_eigenvalue=jnp.max(eigenvalues),
        gram_condition_number=jnp.max(eigenvalues) / jnp.maximum(
            jnp.min(eigenvalues),
            jnp.asarray(1e-30, dtype=eigenvalues.dtype),
        ),
        max_offdiag_correlation=jnp.max(offdiag),
        max_inverse_sm_link_unitarity_residual=sm_unitarity,
        max_inverse_higgs_link_unitarity_residual=higgs_unitarity,
    )
