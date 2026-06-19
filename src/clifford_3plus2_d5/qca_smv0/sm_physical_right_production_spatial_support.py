"""One-tick spatial-support audit for the full production map.

Stage 43 is the first numerical support measurement of the assembled
physical-right production tick after both dense force bottlenecks have local
production replacements.  It perturbs each production state sector at one
lattice site/link, advances one full production tick, and checks that the
measured response stays inside the conservative one-tick stencil envelope from
Stage 35.

This is a discrete finite-support certificate for the implemented map.  It is
not a continuum causal-cone theorem or a multi-step spatial echo.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_generators
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_generators
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_stencil import production_tick_stencil


class PhysicalRightProductionSpatialSupportDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 43 production one-tick spatial support."""

    site_count: jnp.ndarray
    perturbation_size: jnp.ndarray
    support_threshold: jnp.ndarray
    outside_support_threshold: jnp.ndarray
    predicted_tick_radius: jnp.ndarray
    family_support_radius: jnp.ndarray
    higgs_support_radius: jnp.ndarray
    higgs_momentum_support_radius: jnp.ndarray
    sm_link_support_radius: jnp.ndarray
    sm_momentum_support_radius: jnp.ndarray
    higgs_link_support_radius: jnp.ndarray
    max_measured_support_radius: jnp.ndarray
    family_support_count: jnp.ndarray
    higgs_support_count: jnp.ndarray
    higgs_momentum_support_count: jnp.ndarray
    sm_link_support_count: jnp.ndarray
    sm_momentum_support_count: jnp.ndarray
    higgs_link_support_count: jnp.ndarray
    max_support_count: jnp.ndarray
    outside_predicted_radius_site_count: jnp.ndarray
    outside_predicted_radius_max_norm: jnp.ndarray
    max_response_norm: jnp.ndarray
    min_detected_response_norm: jnp.ndarray


def _stencil_radius(stencil: tuple[tuple[int, int, int], ...]) -> int:
    return max(max(abs(component) for component in displacement) for displacement in stencil)


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


def _support_summary(
    response_norm: jnp.ndarray,
    distances: jnp.ndarray,
    *,
    threshold: float,
    outside_threshold: float,
    predicted_radius: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    support = response_norm > jnp.asarray(threshold, dtype=response_norm.dtype)
    outside = distances > predicted_radius
    outside_support = response_norm > jnp.asarray(outside_threshold, dtype=response_norm.dtype)
    measured_support = support & (~outside | outside_support)
    support_distances = jnp.where(measured_support, distances, 0)
    support_values = jnp.where(measured_support, response_norm, jnp.inf)
    return (
        jnp.max(support_distances),
        jnp.sum(measured_support).astype(jnp.int32),
        jnp.sum(outside & outside_support).astype(jnp.int32),
        jnp.max(jnp.where(outside & outside_support, response_norm, 0.0)),
        jnp.min(support_values),
    )


def _left_perturb_sm_link(
    state: PhysicalRightProductionRolloutState,
    center: tuple[int, int, int],
    perturbation_size: float,
) -> PhysicalRightProductionRolloutState:
    update = jsp_linalg.expm(
        jnp.asarray(perturbation_size, dtype=jnp.real(state.sm_links).dtype)
        * sm_generators(dtype=state.sm_links.dtype)[0],
    )
    link_index = center + (0,)
    updated_link = update @ state.sm_links[link_index]
    return state._replace(sm_links=state.sm_links.at[link_index].set(updated_link))


def _left_perturb_higgs_link(
    state: PhysicalRightProductionRolloutState,
    center: tuple[int, int, int],
    perturbation_size: float,
) -> PhysicalRightProductionRolloutState:
    update = jsp_linalg.expm(
        jnp.asarray(perturbation_size, dtype=jnp.real(state.higgs_links).dtype)
        * sm_higgs_generators().astype(state.higgs_links.dtype)[0],
    )
    link_index = center + (0,)
    updated_link = update @ state.higgs_links[link_index]
    return state._replace(higgs_links=state.higgs_links.at[link_index].set(updated_link))


def _perturbed_state(
    state: PhysicalRightProductionRolloutState,
    perturbation_kind: str,
    center: tuple[int, int, int],
    perturbation_size: float,
) -> PhysicalRightProductionRolloutState:
    value = jnp.asarray(perturbation_size, dtype=jnp.real(state.family_state).dtype)
    if perturbation_kind == "family":
        return state._replace(family_state=state.family_state.at[center + (0, 0, 0)].add(value))
    if perturbation_kind == "higgs":
        return state._replace(higgs=state.higgs.at[center + (0,)].add(value))
    if perturbation_kind == "higgs_momentum":
        return state._replace(higgs_momenta=state.higgs_momenta.at[center + (0,)].add(value))
    if perturbation_kind == "sm_link":
        return _left_perturb_sm_link(state, center, perturbation_size)
    if perturbation_kind == "sm_momentum":
        return state._replace(sm_momenta=state.sm_momenta.at[center + (0, 0)].add(value))
    if perturbation_kind == "higgs_link":
        return _left_perturb_higgs_link(state, center, perturbation_size)
    raise ValueError(f"unknown perturbation_kind: {perturbation_kind}")


def sm_physical_right_production_spatial_support_diagnostics(
    lattice_shape: tuple[int, int, int] = (7, 7, 7),
    *,
    perturbation_size: float = 1e-2,
    support_threshold: float = 1e-8,
    outside_support_threshold: float = 1e-7,
    step_size: float = 1e-3,
) -> PhysicalRightProductionSpatialSupportDiagnostics:
    """Return Stage 43 one-tick production spatial-support diagnostics."""

    predicted_radius = _stencil_radius(production_tick_stencil())
    if len(lattice_shape) != 3 or any(size <= 2 * predicted_radius + 1 for size in lattice_shape):
        raise ValueError("lattice_shape must contain three sizes larger than the one-tick stencil diameter")
    if perturbation_size <= 0:
        raise ValueError(f"perturbation_size must be positive, got {perturbation_size}")
    if support_threshold <= 0:
        raise ValueError(f"support_threshold must be positive, got {support_threshold}")
    if outside_support_threshold <= 0:
        raise ValueError(f"outside_support_threshold must be positive, got {outside_support_threshold}")
    if outside_support_threshold < support_threshold:
        raise ValueError("outside_support_threshold must be >= support_threshold")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    center = tuple(size // 2 for size in lattice_shape)
    distances = _periodic_chebyshev_distances(lattice_shape, center)
    initial = sm_physical_right_production_initial_state(lattice_shape)
    baseline = sm_physical_right_production_step(initial, step_size=step_size)

    radii = []
    counts = []
    outside_counts = []
    outside_maxima = []
    maxima = []
    minima = []
    for perturbation_kind in (
        "family",
        "higgs",
        "higgs_momentum",
        "sm_link",
        "sm_momentum",
        "higgs_link",
    ):
        perturbed = _perturbed_state(initial, perturbation_kind, center, perturbation_size)
        response = sm_physical_right_production_step(perturbed, step_size=step_size)
        response_norm = _combined_state_delta_norm(response, baseline)
        radius, count, outside_count, outside_maximum, minimum = _support_summary(
            response_norm,
            distances,
            threshold=support_threshold,
            outside_threshold=outside_support_threshold,
            predicted_radius=predicted_radius,
        )
        radii.append(radius)
        counts.append(count)
        outside_counts.append(outside_count)
        outside_maxima.append(outside_maximum)
        maxima.append(jnp.max(response_norm))
        minima.append(minimum)

    radius_values = jnp.asarray(radii)
    count_values = jnp.asarray(counts)
    outside_count_values = jnp.asarray(outside_counts)
    outside_max_values = jnp.asarray(outside_maxima)
    maximum_values = jnp.asarray(maxima)
    minimum_values = jnp.asarray(minima)
    return PhysicalRightProductionSpatialSupportDiagnostics(
        site_count=jnp.asarray(lattice_shape[0] * lattice_shape[1] * lattice_shape[2], dtype=jnp.int32),
        perturbation_size=jnp.asarray(perturbation_size, dtype=jnp.float32),
        support_threshold=jnp.asarray(support_threshold, dtype=jnp.float32),
        outside_support_threshold=jnp.asarray(outside_support_threshold, dtype=jnp.float32),
        predicted_tick_radius=jnp.asarray(predicted_radius, dtype=jnp.int32),
        family_support_radius=radius_values[0],
        higgs_support_radius=radius_values[1],
        higgs_momentum_support_radius=radius_values[2],
        sm_link_support_radius=radius_values[3],
        sm_momentum_support_radius=radius_values[4],
        higgs_link_support_radius=radius_values[5],
        max_measured_support_radius=jnp.max(radius_values),
        family_support_count=count_values[0],
        higgs_support_count=count_values[1],
        higgs_momentum_support_count=count_values[2],
        sm_link_support_count=count_values[3],
        sm_momentum_support_count=count_values[4],
        higgs_link_support_count=count_values[5],
        max_support_count=jnp.max(count_values),
        outside_predicted_radius_site_count=jnp.sum(outside_count_values).astype(jnp.int32),
        outside_predicted_radius_max_norm=jnp.max(outside_max_values),
        max_response_norm=jnp.max(maximum_values),
        min_detected_response_norm=jnp.min(minimum_values),
    )
