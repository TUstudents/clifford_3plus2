"""Dense-workload audit for the production map.

Stage 36 explained why the pre-Stage-37 dense physical-right production tick
was not yet a large-lattice spatial-echo engine.  The raw rollout state was
only linear in the site count, but the finite-difference Wilson force evaluated
a global plaquette action for every link/generator coordinate.  That made the
force certificate quadratic in the number of sites.

This stage records that scaling explicitly so the next implementation target
is clear in the ledger: replace the finite-difference global force with a local
analytic staple force before attempting large-lattice production echoes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_family_higgs import SM_FAMILY_DIM
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    BCC_PLAQUETTE_PAIRS,
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM

COMPLEX64_BYTES = 8
FLOAT32_BYTES = 4
BCC_EDGE_COUNT = 8
DIRAC_SPIN_DIM = 4


class PhysicalRightProductionWorkloadDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 36 dense workload scaling."""

    site_count: jnp.ndarray
    state_bytes_per_site: jnp.ndarray
    state_mib: jnp.ndarray
    sm_link_bytes_per_site: jnp.ndarray
    sm_link_state_fraction: jnp.ndarray
    wilson_force_coordinate_count: jnp.ndarray
    finite_difference_action_evaluations: jnp.ndarray
    finite_difference_plaquette_holonomies: jnp.ndarray
    single_site_plaquette_holonomies: jnp.ndarray
    quadratic_work_ratio_to_single_site: jnp.ndarray
    local_force_target_plaquette_holonomies: jnp.ndarray
    finite_difference_to_local_work_ratio: jnp.ndarray


def _site_count(lattice_shape: tuple[int, int, int]) -> int:
    if len(lattice_shape) != 3 or any(size < 1 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")
    return lattice_shape[0] * lattice_shape[1] * lattice_shape[2]


def physical_right_production_state_bytes_per_site() -> int:
    """Return dense rollout-state storage per lattice site in bytes."""

    family_state = DIRAC_SPIN_DIM * SM_INTERNAL_DIM * SM_FAMILY_DIM * COMPLEX64_BYTES
    higgs = SM_HIGGS_DIM * COMPLEX64_BYTES
    higgs_momenta = SM_HIGGS_DIM * COMPLEX64_BYTES
    sm_links = BCC_EDGE_COUNT * SM_INTERNAL_DIM * SM_INTERNAL_DIM * COMPLEX64_BYTES
    sm_momenta = BCC_EDGE_COUNT * SM_GENERATOR_COUNT * FLOAT32_BYTES
    higgs_links = BCC_EDGE_COUNT * SM_HIGGS_DIM * SM_HIGGS_DIM * COMPLEX64_BYTES
    return family_state + higgs + higgs_momenta + sm_links + sm_momenta + higgs_links


def physical_right_production_sm_link_bytes_per_site() -> int:
    """Return dense SM-link storage per lattice site in bytes."""

    return BCC_EDGE_COUNT * SM_INTERNAL_DIM * SM_INTERNAL_DIM * COMPLEX64_BYTES


def sm_physical_right_production_workload_diagnostics(
    lattice_shape: tuple[int, int, int] = (3, 3, 3),
) -> PhysicalRightProductionWorkloadDiagnostics:
    """Return Stage 36 dense production-workload diagnostics."""

    sites = _site_count(lattice_shape)
    plaquette_count = len(BCC_PLAQUETTE_PAIRS)
    coordinates = sites * BCC_EDGE_COUNT * SM_GENERATOR_COUNT
    finite_difference_actions = 2 * coordinates
    finite_difference_plaquettes = finite_difference_actions * sites * plaquette_count
    single_site_coordinates = BCC_EDGE_COUNT * SM_GENERATOR_COUNT
    single_site_plaquettes = 2 * single_site_coordinates * plaquette_count
    local_force_plaquettes = sites * plaquette_count
    state_bytes_per_site = physical_right_production_state_bytes_per_site()
    sm_link_bytes = physical_right_production_sm_link_bytes_per_site()

    return PhysicalRightProductionWorkloadDiagnostics(
        site_count=jnp.asarray(sites, dtype=jnp.int32),
        state_bytes_per_site=jnp.asarray(state_bytes_per_site, dtype=jnp.int32),
        state_mib=jnp.asarray(sites * state_bytes_per_site / (1024 * 1024), dtype=jnp.float32),
        sm_link_bytes_per_site=jnp.asarray(sm_link_bytes, dtype=jnp.int32),
        sm_link_state_fraction=jnp.asarray(sm_link_bytes / state_bytes_per_site, dtype=jnp.float32),
        wilson_force_coordinate_count=jnp.asarray(coordinates, dtype=jnp.int32),
        finite_difference_action_evaluations=jnp.asarray(finite_difference_actions, dtype=jnp.int32),
        finite_difference_plaquette_holonomies=jnp.asarray(finite_difference_plaquettes, dtype=jnp.int32),
        single_site_plaquette_holonomies=jnp.asarray(single_site_plaquettes, dtype=jnp.int32),
        quadratic_work_ratio_to_single_site=jnp.asarray(
            finite_difference_plaquettes / single_site_plaquettes,
            dtype=jnp.float32,
        ),
        local_force_target_plaquette_holonomies=jnp.asarray(local_force_plaquettes, dtype=jnp.int32),
        finite_difference_to_local_work_ratio=jnp.asarray(
            finite_difference_plaquettes / local_force_plaquettes,
            dtype=jnp.float32,
        ),
    )
