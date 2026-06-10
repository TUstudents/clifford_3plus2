"""Local Wilson-force replacement for the production map.

Stage 37 replaces the dense finite-difference Wilson-force path identified in
Stage 36 with the analytic BCC plaquette-staple force implemented in
``sm_left_wilson_force``.  The finite-difference force remains available as a
legacy oracle, but the production tick now uses a local derivative whose work is
linear in the number of sites.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    sm_finite_difference_left_wilson_force,
    sm_left_wilson_force,
    sm_local_wilson_force,
    sm_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    BCC_PLAQUETTE_PAIRS,
    SM_GENERATOR_COUNT,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    sm_average_wilson_action_density,
    sm_generators,
    sm_identity_links,
    sm_link_field_from_algebra,
    sm_pure_gauge_links_from_site_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)


AD_CHECK_INDICES: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 0, 0, 0, 0),
    (0, 0, 0, 1, 1),
    (0, 0, 0, 5, 2),
    (1, 0, 0, 3, 8),
    (1, 0, 0, 4, 4),
    (1, 0, 0, 6, 10),
)


class PhysicalRightProductionLocalForceDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 37 local Wilson-force replacement."""

    identity_force_norm: jnp.ndarray
    pure_gauge_force_norm: jnp.ndarray
    nonflat_force_norm: jnp.ndarray
    production_local_alias_residual: jnp.ndarray
    covariance_residual: jnp.ndarray
    ad_coordinate_residual: jnp.ndarray
    legacy_fd_relative_residual: jnp.ndarray
    legacy_force_plaquette_holonomies: jnp.ndarray
    local_force_plaquette_holonomies: jnp.ndarray
    finite_difference_to_local_work_ratio: jnp.ndarray


def _left_update_action_gradient(
    links: jnp.ndarray,
    index: tuple[int, int, int, int, int],
) -> jnp.ndarray:
    generators = sm_generators(dtype=links.dtype)
    generator = generators[index[4]]

    def action(epsilon: jnp.ndarray) -> jnp.ndarray:
        update = jsp_linalg.expm(epsilon * generator)
        updated = links.at[index[:4]].set(update @ links[index[:4]])
        return sm_average_wilson_action_density(updated)

    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    return jax.grad(action)(jnp.asarray(0.0, dtype=real_dtype))


def sm_local_wilson_force_ad_coordinate_residual(
    links: jnp.ndarray,
    indices: tuple[tuple[int, int, int, int, int], ...] = AD_CHECK_INDICES,
) -> jnp.ndarray:
    """Return the max residual against exact AD left-update derivatives."""

    local_force = sm_local_wilson_force(links)
    residuals = [
        jnp.abs(local_force[index] - _left_update_action_gradient(links, index))
        for index in indices
    ]
    return jnp.max(jnp.stack(residuals))


def sm_physical_right_production_local_force_diagnostics(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
) -> PhysicalRightProductionLocalForceDiagnostics:
    """Return Stage 37 local Wilson-force replacement diagnostics."""

    identity_links = sm_identity_links(lattice_shape)
    pure_links = sm_pure_gauge_links_from_site_algebra(deterministic_sm_site_theta(lattice_shape))
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=1.0))
    local_force = sm_local_wilson_force(links)
    production_force = sm_left_wilson_force(links, epsilon=1e-3)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape, scale=0.08))
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_force = sm_left_wilson_force(transformed_links)
    expected_force = sm_transform_momenta(local_force, site_gauge)
    legacy_force = sm_finite_difference_left_wilson_force(links, epsilon=1e-2)
    site_count = lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
    plaquette_count = len(BCC_PLAQUETTE_PAIRS)
    local_holonomies = site_count * plaquette_count
    legacy_holonomies = 2 * site_count * 8 * SM_GENERATOR_COUNT * local_holonomies

    local_norm = jnp.linalg.norm(local_force)
    return PhysicalRightProductionLocalForceDiagnostics(
        identity_force_norm=jnp.linalg.norm(sm_left_wilson_force(identity_links)),
        pure_gauge_force_norm=jnp.linalg.norm(sm_left_wilson_force(pure_links)),
        nonflat_force_norm=local_norm,
        production_local_alias_residual=jnp.linalg.norm(production_force - local_force),
        covariance_residual=jnp.max(jnp.abs(transformed_force - expected_force)),
        ad_coordinate_residual=sm_local_wilson_force_ad_coordinate_residual(links),
        legacy_fd_relative_residual=jnp.linalg.norm(legacy_force - local_force) / jnp.maximum(local_norm, 1e-12),
        legacy_force_plaquette_holonomies=jnp.asarray(legacy_holonomies, dtype=jnp.int32),
        local_force_plaquette_holonomies=jnp.asarray(local_holonomies, dtype=jnp.int32),
        finite_difference_to_local_work_ratio=jnp.asarray(legacy_holonomies / local_holonomies, dtype=jnp.float32),
    )
