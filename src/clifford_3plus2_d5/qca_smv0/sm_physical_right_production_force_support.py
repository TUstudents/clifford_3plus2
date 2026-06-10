"""Numerical spatial-support audit for the local Wilson force.

Stage 41 is the first numerical support measurement after the Stage 37 local
Wilson-force replacement.  It perturbs one SM link on a large-enough periodic
lattice, computes the production Wilson force, and checks that the nonzero
response stays inside the conservative two-hop force envelope from Stage 35.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import sm_left_wilson_force
from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_field_from_algebra
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_stencil import two_hop_force_stencil


class PhysicalRightProductionForceSupportDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 41 local-force spatial support."""

    site_count: jnp.ndarray
    perturbation_size: jnp.ndarray
    predicted_force_radius: jnp.ndarray
    measured_support_radius: jnp.ndarray
    support_site_count: jnp.ndarray
    radius_zero_support_count: jnp.ndarray
    radius_one_support_count: jnp.ndarray
    radius_two_support_count: jnp.ndarray
    max_force_site_norm: jnp.ndarray
    min_support_site_norm: jnp.ndarray
    outside_predicted_radius_max_norm: jnp.ndarray
    outside_predicted_radius_site_count: jnp.ndarray
    center_site_supported: jnp.ndarray


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


def sm_physical_right_production_force_support_diagnostics(
    lattice_shape: tuple[int, int, int] = (7, 7, 7),
    *,
    perturbation_size: float = 0.2,
    support_threshold: float = 1e-10,
) -> PhysicalRightProductionForceSupportDiagnostics:
    """Return Stage 41 local-force spatial-support diagnostics."""

    predicted_radius = _stencil_radius(two_hop_force_stencil())
    if len(lattice_shape) != 3 or any(size <= 2 * predicted_radius + 1 for size in lattice_shape):
        raise ValueError(
            "lattice_shape must contain three sizes larger than the two-hop diameter",
        )
    if perturbation_size <= 0:
        raise ValueError(f"perturbation_size must be positive, got {perturbation_size}")
    if support_threshold <= 0:
        raise ValueError(f"support_threshold must be positive, got {support_threshold}")

    center = tuple(size // 2 for size in lattice_shape)
    theta = jnp.zeros((*lattice_shape, 8, 12), dtype=jnp.float32)
    theta = theta.at[center + (0, 0)].set(jnp.asarray(perturbation_size, dtype=theta.dtype))
    links = sm_link_field_from_algebra(theta)
    force = sm_left_wilson_force(links)
    site_norm = jnp.linalg.norm(force.reshape((*lattice_shape, -1)), axis=-1)
    support = site_norm > jnp.asarray(support_threshold, dtype=site_norm.dtype)
    distances = _periodic_chebyshev_distances(lattice_shape, center)
    outside = distances > predicted_radius

    support_distances = jnp.where(support, distances, 0)
    support_values = jnp.where(support, site_norm, jnp.inf)
    return PhysicalRightProductionForceSupportDiagnostics(
        site_count=jnp.asarray(lattice_shape[0] * lattice_shape[1] * lattice_shape[2], dtype=jnp.int32),
        perturbation_size=jnp.asarray(perturbation_size, dtype=jnp.float32),
        predicted_force_radius=jnp.asarray(predicted_radius, dtype=jnp.int32),
        measured_support_radius=jnp.max(support_distances),
        support_site_count=jnp.sum(support).astype(jnp.int32),
        radius_zero_support_count=jnp.sum(support & (distances == 0)).astype(jnp.int32),
        radius_one_support_count=jnp.sum(support & (distances == 1)).astype(jnp.int32),
        radius_two_support_count=jnp.sum(support & (distances == 2)).astype(jnp.int32),
        max_force_site_norm=jnp.max(site_norm),
        min_support_site_norm=jnp.min(support_values),
        outside_predicted_radius_max_norm=jnp.max(jnp.where(outside, site_norm, 0.0)),
        outside_predicted_radius_site_count=jnp.sum(outside & support).astype(jnp.int32),
        center_site_supported=support[center],
    )

