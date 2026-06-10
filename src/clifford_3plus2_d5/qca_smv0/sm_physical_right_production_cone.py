"""Multi-step family cone audit for the full production map.

Stage 44 extends the one-tick Stage 43 spatial-support measurement in the
direction that actually tests propagation: a localized family-state
perturbation is evolved by the assembled physical-right production tick for
several ticks, and the measured support radius is compared with the discrete
BCC one-hop-per-tick cone.

This is a finite-horizon discrete support audit.  It is not a continuum light
cone, a Lyapunov estimate, or a claim about all possible perturbation sectors.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
)


class PhysicalRightProductionConeDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 44 production family-cone growth."""

    site_count: jnp.ndarray
    max_steps: jnp.ndarray
    perturbation_size: jnp.ndarray
    support_threshold: jnp.ndarray
    step_one_support_radius: jnp.ndarray
    step_two_support_radius: jnp.ndarray
    step_three_support_radius: jnp.ndarray
    max_support_radius: jnp.ndarray
    step_one_support_count: jnp.ndarray
    step_two_support_count: jnp.ndarray
    step_three_support_count: jnp.ndarray
    max_support_count: jnp.ndarray
    outside_step_cone_site_count: jnp.ndarray
    outside_step_cone_max_norm: jnp.ndarray
    radius_growth_residual: jnp.ndarray
    support_count_growth_min_delta: jnp.ndarray
    max_response_norm: jnp.ndarray
    min_detected_response_norm: jnp.ndarray


def _periodic_chebyshev_distances(
    lattice_shape: tuple[int, int, int],
    center: tuple[int, int, int],
) -> jnp.ndarray:
    axes = []
    for size, origin in zip(lattice_shape, center, strict=True):
        coordinates = jnp.arange(size)
        raw = jnp.abs(coordinates - origin)
        axes.append(jnp.minimum(raw, size - raw))
    dx, dy, dz = jnp.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return jnp.maximum(jnp.maximum(dx, dy), dz)


def _site_norm(field: jnp.ndarray, lattice_shape: tuple[int, int, int]) -> jnp.ndarray:
    return jnp.linalg.norm(field.reshape((*lattice_shape, -1)), axis=-1)


def _combined_state_delta_norm(
    left: PhysicalRightProductionRolloutState,
    right: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    lattice_shape = left.family_state.shape[:3]
    component_norms = (
        _site_norm(left.family_state - right.family_state, lattice_shape),
        _site_norm(left.higgs - right.higgs, lattice_shape),
        _site_norm(left.higgs_momenta - right.higgs_momenta, lattice_shape),
        _site_norm(left.sm_links - right.sm_links, lattice_shape),
        _site_norm(left.sm_momenta - right.sm_momenta, lattice_shape),
        _site_norm(left.higgs_links - right.higgs_links, lattice_shape),
    )
    total = jnp.zeros(lattice_shape, dtype=jnp.result_type(*component_norms))
    for component in component_norms:
        total = total + component * component
    return jnp.sqrt(total)


def _family_perturbation(
    state: PhysicalRightProductionRolloutState,
    center: tuple[int, int, int],
    perturbation_size: float,
) -> PhysicalRightProductionRolloutState:
    value = jnp.asarray(perturbation_size, dtype=jnp.real(state.family_state).dtype)
    return state._replace(family_state=state.family_state.at[center + (0, 0, 0)].add(value))


def _validate_inputs(
    lattice_shape: tuple[int, int, int],
    max_steps: int,
    perturbation_size: float,
    support_threshold: float,
    step_size: float,
) -> None:
    if max_steps < 1 or max_steps > 3:
        raise ValueError(f"max_steps must be 1, 2, or 3, got {max_steps}")
    if len(lattice_shape) != 3 or any(size < 2 * max_steps + 1 for size in lattice_shape):
        raise ValueError("lattice_shape must contain three sizes at least the family-cone diameter")
    if perturbation_size <= 0:
        raise ValueError(f"perturbation_size must be positive, got {perturbation_size}")
    if support_threshold <= 0:
        raise ValueError(f"support_threshold must be positive, got {support_threshold}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")


def sm_physical_right_production_cone_diagnostics(
    lattice_shape: tuple[int, int, int] = (7, 7, 7),
    *,
    max_steps: int = 3,
    perturbation_size: float = 1e-2,
    support_threshold: float = 1e-8,
    step_size: float = 1e-3,
) -> PhysicalRightProductionConeDiagnostics:
    """Return Stage 44 multi-step family-cone diagnostics."""

    _validate_inputs(lattice_shape, max_steps, perturbation_size, support_threshold, step_size)
    center = tuple(size // 2 for size in lattice_shape)
    distances = _periodic_chebyshev_distances(lattice_shape, center)
    initial = sm_physical_right_production_initial_state(lattice_shape)
    perturbed = _family_perturbation(initial, center, perturbation_size)

    radii = []
    counts = []
    outside_counts = []
    outside_maxima = []
    maxima = []
    minima = []
    for steps in range(1, max_steps + 1):
        baseline = sm_physical_right_production_rollout(
            initial,
            steps=steps,
            step_size=step_size,
        )
        response = sm_physical_right_production_rollout(
            perturbed,
            steps=steps,
            step_size=step_size,
        )
        response_norm = _combined_state_delta_norm(response, baseline)
        support = response_norm > jnp.asarray(support_threshold, dtype=response_norm.dtype)
        outside = distances > steps
        support_distances = jnp.where(support, distances, 0)
        support_values = jnp.where(support, response_norm, jnp.inf)
        radii.append(jnp.max(support_distances))
        counts.append(jnp.sum(support).astype(jnp.int32))
        outside_counts.append(jnp.sum(support & outside).astype(jnp.int32))
        outside_maxima.append(jnp.max(jnp.where(support & outside, response_norm, 0.0)))
        maxima.append(jnp.max(response_norm))
        minima.append(jnp.min(support_values))

    radius_values = jnp.pad(jnp.asarray(radii), (0, 3 - max_steps), mode="edge")
    count_values = jnp.pad(jnp.asarray(counts), (0, 3 - max_steps), mode="edge")
    outside_count_values = jnp.asarray(outside_counts)
    outside_max_values = jnp.asarray(outside_maxima)
    maximum_values = jnp.asarray(maxima)
    minimum_values = jnp.asarray(minima)
    expected = jnp.arange(1, max_steps + 1, dtype=radius_values.dtype)
    measured = jnp.asarray(radii, dtype=radius_values.dtype)
    count_deltas = jnp.diff(jnp.asarray(counts), prepend=jnp.asarray(0, dtype=jnp.int32))

    return PhysicalRightProductionConeDiagnostics(
        site_count=jnp.asarray(lattice_shape[0] * lattice_shape[1] * lattice_shape[2], dtype=jnp.int32),
        max_steps=jnp.asarray(max_steps, dtype=jnp.int32),
        perturbation_size=jnp.asarray(perturbation_size, dtype=jnp.float32),
        support_threshold=jnp.asarray(support_threshold, dtype=jnp.float32),
        step_one_support_radius=radius_values[0],
        step_two_support_radius=radius_values[1],
        step_three_support_radius=radius_values[2],
        max_support_radius=jnp.max(measured),
        step_one_support_count=count_values[0],
        step_two_support_count=count_values[1],
        step_three_support_count=count_values[2],
        max_support_count=jnp.max(jnp.asarray(counts)),
        outside_step_cone_site_count=jnp.sum(outside_count_values).astype(jnp.int32),
        outside_step_cone_max_norm=jnp.max(outside_max_values),
        radius_growth_residual=jnp.max(jnp.abs(measured - expected)),
        support_count_growth_min_delta=jnp.min(count_deltas),
        max_response_norm=jnp.max(maximum_values),
        min_detected_response_norm=jnp.min(minimum_values),
    )
