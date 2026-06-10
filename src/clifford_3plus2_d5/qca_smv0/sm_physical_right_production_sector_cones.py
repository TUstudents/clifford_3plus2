"""Two-tick all-sector cone audit for the full production map.

Stage 45 closes the locality measurement thread started in Stage 43/44.  It
keeps the same localized perturbation definitions as the one-tick audit, but
evolves each production sector through two assembled physical-right production
ticks and checks that the measured response remains inside the discrete
step-count cone.

This is a finite-horizon support measurement of the implemented discrete map.
It is not a continuum light-cone theorem, a performance benchmark, or a claim
about exact energy conservation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_spatial_support import (
    _combined_state_delta_norm,
    _periodic_chebyshev_distances,
    _perturbed_state,
)


PERTURBATION_KINDS = (
    "family",
    "higgs",
    "higgs_momentum",
    "sm_link",
    "sm_momentum",
    "higgs_link",
)


class PhysicalRightProductionSectorConesDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 45 all-sector two-tick cone support."""

    site_count: jnp.ndarray
    horizon_steps: jnp.ndarray
    sector_count: jnp.ndarray
    perturbation_size: jnp.ndarray
    support_threshold: jnp.ndarray
    family_support_radius: jnp.ndarray
    higgs_support_radius: jnp.ndarray
    higgs_momentum_support_radius: jnp.ndarray
    sm_link_support_radius: jnp.ndarray
    sm_momentum_support_radius: jnp.ndarray
    higgs_link_support_radius: jnp.ndarray
    max_support_radius: jnp.ndarray
    family_support_count: jnp.ndarray
    higgs_support_count: jnp.ndarray
    higgs_momentum_support_count: jnp.ndarray
    sm_link_support_count: jnp.ndarray
    sm_momentum_support_count: jnp.ndarray
    higgs_link_support_count: jnp.ndarray
    min_support_count: jnp.ndarray
    max_support_count: jnp.ndarray
    outside_step_cone_site_count: jnp.ndarray
    outside_step_cone_max_norm: jnp.ndarray
    radius_overshoot: jnp.ndarray
    max_response_norm: jnp.ndarray
    min_detected_response_norm: jnp.ndarray


def _validate_inputs(
    lattice_shape: tuple[int, int, int],
    horizon_steps: int,
    perturbation_size: float,
    support_threshold: float,
    step_size: float,
) -> None:
    if horizon_steps < 2 or horizon_steps > 2:
        raise ValueError(f"horizon_steps must be exactly 2 for Stage 45, got {horizon_steps}")
    if len(lattice_shape) != 3 or any(size < 2 * horizon_steps + 1 for size in lattice_shape):
        raise ValueError("lattice_shape must contain three sizes at least the two-tick cone diameter")
    if perturbation_size <= 0:
        raise ValueError(f"perturbation_size must be positive, got {perturbation_size}")
    if support_threshold <= 0:
        raise ValueError(f"support_threshold must be positive, got {support_threshold}")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")


def sm_physical_right_production_sector_cones_diagnostics(
    lattice_shape: tuple[int, int, int] = (5, 5, 5),
    *,
    horizon_steps: int = 2,
    perturbation_size: float = 1e-2,
    support_threshold: float = 1e-8,
    step_size: float = 1e-3,
) -> PhysicalRightProductionSectorConesDiagnostics:
    """Return Stage 45 all-sector two-tick cone diagnostics."""

    _validate_inputs(lattice_shape, horizon_steps, perturbation_size, support_threshold, step_size)
    center = tuple(size // 2 for size in lattice_shape)
    distances = _periodic_chebyshev_distances(lattice_shape, center)
    initial = sm_physical_right_production_initial_state(lattice_shape)
    baseline = sm_physical_right_production_rollout(
        initial,
        steps=horizon_steps,
        step_size=step_size,
    )

    radii = []
    counts = []
    outside_counts = []
    outside_maxima = []
    maxima = []
    minima = []
    for perturbation_kind in PERTURBATION_KINDS:
        perturbed = _perturbed_state(initial, perturbation_kind, center, perturbation_size)
        response = sm_physical_right_production_rollout(
            perturbed,
            steps=horizon_steps,
            step_size=step_size,
        )
        response_norm = _combined_state_delta_norm(response, baseline)
        support = response_norm > jnp.asarray(support_threshold, dtype=response_norm.dtype)
        outside = distances > horizon_steps
        support_distances = jnp.where(support, distances, 0)
        support_values = jnp.where(support, response_norm, jnp.inf)
        radii.append(jnp.max(support_distances))
        counts.append(jnp.sum(support).astype(jnp.int32))
        outside_counts.append(jnp.sum(support & outside).astype(jnp.int32))
        outside_maxima.append(jnp.max(jnp.where(support & outside, response_norm, 0.0)))
        maxima.append(jnp.max(response_norm))
        minima.append(jnp.min(support_values))

    radius_values = jnp.asarray(radii)
    count_values = jnp.asarray(counts)
    outside_count_values = jnp.asarray(outside_counts)
    outside_max_values = jnp.asarray(outside_maxima)
    maximum_values = jnp.asarray(maxima)
    minimum_values = jnp.asarray(minima)

    horizon = jnp.asarray(horizon_steps, dtype=radius_values.dtype)
    return PhysicalRightProductionSectorConesDiagnostics(
        site_count=jnp.asarray(lattice_shape[0] * lattice_shape[1] * lattice_shape[2], dtype=jnp.int32),
        horizon_steps=jnp.asarray(horizon_steps, dtype=jnp.int32),
        sector_count=jnp.asarray(len(PERTURBATION_KINDS), dtype=jnp.int32),
        perturbation_size=jnp.asarray(perturbation_size, dtype=jnp.float32),
        support_threshold=jnp.asarray(support_threshold, dtype=jnp.float32),
        family_support_radius=radius_values[0],
        higgs_support_radius=radius_values[1],
        higgs_momentum_support_radius=radius_values[2],
        sm_link_support_radius=radius_values[3],
        sm_momentum_support_radius=radius_values[4],
        higgs_link_support_radius=radius_values[5],
        max_support_radius=jnp.max(radius_values),
        family_support_count=count_values[0],
        higgs_support_count=count_values[1],
        higgs_momentum_support_count=count_values[2],
        sm_link_support_count=count_values[3],
        sm_momentum_support_count=count_values[4],
        higgs_link_support_count=count_values[5],
        min_support_count=jnp.min(count_values),
        max_support_count=jnp.max(count_values),
        outside_step_cone_site_count=jnp.sum(outside_count_values).astype(jnp.int32),
        outside_step_cone_max_norm=jnp.max(outside_max_values),
        radius_overshoot=jnp.max(jnp.maximum(radius_values - horizon, 0)),
        max_response_norm=jnp.max(maximum_values),
        min_detected_response_norm=jnp.min(minimum_values),
    )
