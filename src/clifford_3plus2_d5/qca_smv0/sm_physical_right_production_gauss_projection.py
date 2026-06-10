"""Momentum-only Gauss relaxation for the production rollout state.

Stage 46 is the first constraint-solving step after the Gauss monitor.  It
does not change links, matter fields, Higgs fields, or the production tick.
Instead, at fixed production state except for SM link momenta, it minimizes the
current physical-right Gauss residual

    G(P) = div_E(P) - rho_matter

along the exact gradient direction of ``0.5 ||G||^2``.  Since ``G`` is affine in
the momenta for fixed links and charges, the line step is chosen by the exact
one-dimensional least-squares minimizer along that gradient direction.

This is a local algebraic relaxation/projection precursor.  It is not yet a
full nonlinear gauge-orbit projection, a Gauss-preserving production
integrator, or a constraint-preserving update rule.
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
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
)


class PhysicalRightProductionGaussProjectionDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 46 Gauss relaxation."""

    vacuum_initial_gauss_norm: jnp.ndarray
    vacuum_relaxed_gauss_norm: jnp.ndarray
    vacuum_momentum_delta_norm: jnp.ndarray
    initial_gauss_norm: jnp.ndarray
    relaxed_gauss_norm: jnp.ndarray
    gauss_reduction_norm: jnp.ndarray
    gauss_reduction_fraction: jnp.ndarray
    gradient_norm: jnp.ndarray
    line_step: jnp.ndarray
    momentum_delta_norm: jnp.ndarray
    family_state_delta_norm: jnp.ndarray
    higgs_delta_norm: jnp.ndarray
    higgs_momentum_delta_norm: jnp.ndarray
    sm_link_delta_norm: jnp.ndarray
    higgs_link_delta_norm: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray
    jit_delta_momenta: jnp.ndarray


def _gauss_half_norm_squared(
    momenta: jnp.ndarray,
    state: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    trial_state = state._replace(sm_momenta=momenta)
    gauss = sm_physical_right_production_gauss(trial_state)
    return 0.5 * jnp.sum(gauss * gauss)


def sm_physical_right_production_gauss_gradient(
    state: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    """Return the gradient of ``0.5 ||G||^2`` with respect to SM momenta."""

    return jax.grad(_gauss_half_norm_squared)(state.sm_momenta, state)


def sm_physical_right_production_gauss_relaxation_step(
    state: PhysicalRightProductionRolloutState,
    *,
    relaxation_scale: float = 1.0,
) -> tuple[PhysicalRightProductionRolloutState, jnp.ndarray, jnp.ndarray]:
    """Relax SM momenta along the exact Gauss-residual gradient line.

    Returns ``(relaxed_state, line_step, gradient)``.  ``relaxation_scale`` is
    retained as an explicit damping knob; the default value uses the exact
    steepest-descent line minimizer for the current affine Gauss map.
    """

    if relaxation_scale <= 0:
        raise ValueError(f"relaxation_scale must be positive, got {relaxation_scale}")
    gradient = sm_physical_right_production_gauss_gradient(state)
    initial_gauss = sm_physical_right_production_gauss(state)
    unit_gradient_state = state._replace(sm_momenta=state.sm_momenta - gradient)
    unit_gradient_gauss = sm_physical_right_production_gauss(unit_gradient_state)
    gauss_direction = initial_gauss - unit_gradient_gauss
    numerator = jnp.sum(initial_gauss * gauss_direction)
    denominator = jnp.sum(gauss_direction * gauss_direction)
    exact_step = jnp.where(denominator > 0, numerator / denominator, 0.0)
    line_step = jnp.asarray(relaxation_scale, dtype=state.sm_momenta.dtype) * jnp.maximum(exact_step, 0.0)
    relaxed_state = state._replace(sm_momenta=state.sm_momenta - line_step * gradient)
    return relaxed_state, line_step, gradient


def sm_physical_right_production_gauss_projection_diagnostics() -> PhysicalRightProductionGaussProjectionDiagnostics:
    """Return focused Stage 46 Gauss relaxation diagnostics."""

    vacuum = sm_physical_right_production_vacuum_state()
    relaxed_vacuum, _, _ = sm_physical_right_production_gauss_relaxation_step(vacuum)
    initial = sm_physical_right_production_initial_state()
    relaxed, line_step, gradient = sm_physical_right_production_gauss_relaxation_step(initial)
    jitted_relax = jax.jit(sm_physical_right_production_gauss_relaxation_step)
    jit_relaxed, _, _ = jitted_relax(initial)

    vacuum_initial_gauss = sm_physical_right_production_gauss(vacuum)
    vacuum_relaxed_gauss = sm_physical_right_production_gauss(relaxed_vacuum)
    initial_gauss = sm_physical_right_production_gauss(initial)
    relaxed_gauss = sm_physical_right_production_gauss(relaxed)
    initial_norm = jnp.linalg.norm(initial_gauss)
    relaxed_norm = jnp.linalg.norm(relaxed_gauss)
    reduction = initial_norm - relaxed_norm
    fraction = jnp.where(initial_norm > 0, reduction / initial_norm, 0.0)

    return PhysicalRightProductionGaussProjectionDiagnostics(
        vacuum_initial_gauss_norm=jnp.linalg.norm(vacuum_initial_gauss),
        vacuum_relaxed_gauss_norm=jnp.linalg.norm(vacuum_relaxed_gauss),
        vacuum_momentum_delta_norm=jnp.linalg.norm(relaxed_vacuum.sm_momenta - vacuum.sm_momenta),
        initial_gauss_norm=initial_norm,
        relaxed_gauss_norm=relaxed_norm,
        gauss_reduction_norm=reduction,
        gauss_reduction_fraction=fraction,
        gradient_norm=jnp.linalg.norm(gradient),
        line_step=line_step,
        momentum_delta_norm=jnp.linalg.norm(relaxed.sm_momenta - initial.sm_momenta),
        family_state_delta_norm=jnp.linalg.norm(relaxed.family_state - initial.family_state),
        higgs_delta_norm=jnp.linalg.norm(relaxed.higgs - initial.higgs),
        higgs_momentum_delta_norm=jnp.linalg.norm(relaxed.higgs_momenta - initial.higgs_momenta),
        sm_link_delta_norm=jnp.linalg.norm(relaxed.sm_links - initial.sm_links),
        higgs_link_delta_norm=jnp.linalg.norm(relaxed.higgs_links - initial.higgs_links),
        sm_link_unitarity_residual=sm_link_unitarity_residual(relaxed.sm_links),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(relaxed.higgs_links),
        jit_delta_momenta=jnp.max(jnp.abs(jit_relaxed.sm_momenta - relaxed.sm_momenta)),
    )
