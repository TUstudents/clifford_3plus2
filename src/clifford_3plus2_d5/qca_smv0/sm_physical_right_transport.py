"""Physical-right bridged BCC transport for QCA_SMv0.

Stage 18 applies the Stage 17 antiunitary singlet bridge to finite BCC link
transport.  The input links are still the Stage 2 transport-convention SM link
fields.  Before transport, each link is bridged to the physical-right carrier:

``U_phys = P_L U_transport P_L + P_R conj(U_transport) P_R``.

The resulting kernel is a finite-link transport stage only.  It does not yet
build bridged gauge currents, sourced ticks, or boundary dynamics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import (
    sm_antiunitary_bridge_gauge_from_transport,
    sm_physical_right_gauge_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import (
    sm_family_gauged_dirac_step,
    sm_transform_family_gauge_state,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import SM_FAMILY_DIM, deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    sm_identity_links,
    sm_link_field_from_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge_convention_bridge import (
    sm_left_doublet_projector,
    sm_right_singlet_projector,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightTransportDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 18 physical-right bridged transport."""

    identity_bridge_residual: jnp.ndarray
    finite_bridge_unitarity_residual: jnp.ndarray
    bridged_link_covariance_residual: jnp.ndarray
    identity_transport_reduction_residual: jnp.ndarray
    physical_right_transport_covariance_residual: jnp.ndarray
    physical_right_norm_drift: jnp.ndarray
    transport_kernel_difference_norm: jnp.ndarray
    jit_delta_transport: jnp.ndarray


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, 3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_sm_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if links.ndim != 6 or links.shape[3:] != (8, SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("SM links must have shape (nx, ny, nz, 8, 32, 32)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("SM links must match the lattice")


def sm_physical_right_links_from_transport(transport_links: jnp.ndarray) -> jnp.ndarray:
    """Bridge Stage 2 transport-convention BCC links to physical-right links."""

    _validate_sm_links(transport_links)
    left = sm_left_doublet_projector(dtype=transport_links.dtype)
    right = sm_right_singlet_projector(dtype=transport_links.dtype)
    return left @ transport_links @ left + right @ jnp.conj(transport_links) @ right


def sm_transform_family_physical_right_state(state: jnp.ndarray, transport_site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Transform a family state using the physical-right bridge of a site gauge."""

    lattice_shape = _validate_family_state(state)
    if transport_site_gauge.ndim != 5 or transport_site_gauge.shape[:3] != lattice_shape:
        raise ValueError("transport_site_gauge must match the state lattice")
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_site_gauge)
    return sm_transform_family_gauge_state(state, physical_site_gauge)


def sm_transform_physical_right_links(
    transport_links: jnp.ndarray,
    transport_site_gauge: jnp.ndarray,
) -> jnp.ndarray:
    """Gauge-transform transport links, then bridge them to physical-right links."""

    _validate_sm_links(transport_links)
    transformed_transport_links = sm_transform_links(transport_links, transport_site_gauge)
    return sm_physical_right_links_from_transport(transformed_transport_links)


def sm_family_physical_right_gauged_dirac_step(
    state: jnp.ndarray,
    transport_links: jnp.ndarray,
) -> jnp.ndarray:
    """Transport a family state through bridged physical-right BCC links."""

    lattice_shape = _validate_family_state(state)
    _validate_sm_links(transport_links, lattice_shape)
    physical_links = sm_physical_right_links_from_transport(transport_links)
    return sm_family_gauged_dirac_step(state, physical_links)


def sm_physical_right_link_unitarity_residual(transport_links: jnp.ndarray) -> jnp.ndarray:
    """Return unitarity residual after bridging transport links."""

    physical_links = sm_physical_right_links_from_transport(transport_links)
    return sm_physical_right_gauge_unitarity_residual(physical_links)


def sm_physical_right_transport_diagnostics() -> PhysicalRightTransportDiagnostics:
    """Return focused Stage 18 physical-right transport diagnostics."""

    lattice_shape = (2, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    transport_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.23))
    identity_links = sm_identity_links(lattice_shape, dtype=state.dtype)
    site_theta = deterministic_sm_site_theta(lattice_shape, scale=0.09)
    transport_site_gauge = sm_site_gauge_from_algebra(site_theta)
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_site_gauge)

    physical_links = sm_physical_right_links_from_transport(transport_links)
    transformed_physical_links = sm_transform_physical_right_links(transport_links, transport_site_gauge)
    expected_transformed_physical_links = sm_transform_links(physical_links, physical_site_gauge)

    updated = sm_family_physical_right_gauged_dirac_step(state, transport_links)
    transformed_state = sm_transform_family_physical_right_state(state, transport_site_gauge)
    transformed_updated = sm_family_physical_right_gauged_dirac_step(
        transformed_state,
        sm_transform_links(transport_links, transport_site_gauge),
    )
    expected_updated = sm_transform_family_gauge_state(updated, physical_site_gauge)
    transport_updated = sm_family_gauged_dirac_step(state, transport_links)
    identity_physical_updated = sm_family_physical_right_gauged_dirac_step(state, identity_links)
    identity_transport_updated = sm_family_gauged_dirac_step(state, identity_links)
    jitted = jax.jit(sm_family_physical_right_gauged_dirac_step)
    jit_updated = jitted(state, transport_links)

    return PhysicalRightTransportDiagnostics(
        identity_bridge_residual=jnp.max(jnp.abs(sm_physical_right_links_from_transport(identity_links) - identity_links)),
        finite_bridge_unitarity_residual=sm_physical_right_link_unitarity_residual(transport_links),
        bridged_link_covariance_residual=jnp.max(
            jnp.abs(transformed_physical_links - expected_transformed_physical_links),
        ),
        identity_transport_reduction_residual=jnp.max(jnp.abs(identity_physical_updated - identity_transport_updated)),
        physical_right_transport_covariance_residual=jnp.max(jnp.abs(transformed_updated - expected_updated)),
        physical_right_norm_drift=jnp.abs(state_norm_squared(updated) - state_norm_squared(state)),
        transport_kernel_difference_norm=jnp.linalg.norm(updated - transport_updated),
        jit_delta_transport=jnp.max(jnp.abs(jit_updated - updated)),
    )
