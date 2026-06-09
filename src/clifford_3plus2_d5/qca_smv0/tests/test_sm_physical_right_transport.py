"""Tests for QCA_SMv0 physical-right bridged transport."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import sm_antiunitary_bridge_gauge_from_transport
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import sm_family_gauged_dirac_step, sm_transform_family_gauge_state
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    sm_identity_links,
    sm_link_field_from_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_transport import (
    sm_family_physical_right_gauged_dirac_step,
    sm_physical_right_link_unitarity_residual,
    sm_physical_right_links_from_transport,
    sm_physical_right_transport_diagnostics,
    sm_transform_family_physical_right_state,
    sm_transform_physical_right_links,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def _stage18_fields(lattice_shape: tuple[int, int, int] = (2, 1, 1)):
    state = deterministic_sm_family_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.23))
    site_theta = deterministic_sm_site_theta(lattice_shape, scale=0.09)
    site_gauge = sm_site_gauge_from_algebra(site_theta)
    return state, links, site_gauge


def test_identity_links_are_unchanged_by_physical_right_bridge() -> None:
    identity = sm_identity_links((2, 1, 1))

    assert jnp.max(jnp.abs(sm_physical_right_links_from_transport(identity) - identity)) < 1e-7


def test_finite_bridged_links_are_unitary() -> None:
    _, links, _ = _stage18_fields()

    assert sm_physical_right_link_unitarity_residual(links) < 7e-7


def test_bridged_links_transform_with_physical_right_site_gauge() -> None:
    _, links, transport_site_gauge = _stage18_fields()
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_site_gauge)

    bridged_transformed = sm_transform_physical_right_links(links, transport_site_gauge)
    expected = sm_transform_links(sm_physical_right_links_from_transport(links), physical_site_gauge)

    assert jnp.max(jnp.abs(bridged_transformed - expected)) < 8e-7


def test_physical_right_transport_is_gauge_covariant() -> None:
    state, links, transport_site_gauge = _stage18_fields()
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(transport_site_gauge)

    updated = sm_family_physical_right_gauged_dirac_step(state, links)
    transformed_state = sm_transform_family_physical_right_state(state, transport_site_gauge)
    transformed_links = sm_transform_links(links, transport_site_gauge)
    transformed_updated = sm_family_physical_right_gauged_dirac_step(transformed_state, transformed_links)
    expected = sm_transform_family_gauge_state(updated, physical_site_gauge)

    assert jnp.max(jnp.abs(transformed_updated - expected)) < 8e-7


def test_identity_physical_right_transport_reduces_to_identity_transport() -> None:
    state = deterministic_sm_family_state((2, 1, 1))
    identity = sm_identity_links((2, 1, 1), dtype=state.dtype)

    physical = sm_family_physical_right_gauged_dirac_step(state, identity)
    transport = sm_family_gauged_dirac_step(state, identity)

    assert jnp.max(jnp.abs(physical - transport)) < 1e-7


def test_nontrivial_physical_right_transport_differs_from_transport_convention() -> None:
    state, links, _ = _stage18_fields()

    physical = sm_family_physical_right_gauged_dirac_step(state, links)
    transport = sm_family_gauged_dirac_step(state, links)

    assert jnp.linalg.norm(physical - transport) > 1e-3


def test_physical_right_transport_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_transport_diagnostics()

    assert diagnostics.identity_bridge_residual < 1e-7
    assert diagnostics.finite_bridge_unitarity_residual < 7e-7
    assert diagnostics.bridged_link_covariance_residual < 8e-7
    assert diagnostics.identity_transport_reduction_residual < 1e-7
    assert diagnostics.physical_right_transport_covariance_residual < 8e-7
    assert diagnostics.physical_right_norm_drift < 1e-6
    assert diagnostics.transport_kernel_difference_norm > 1e-3
    assert diagnostics.jit_delta_transport < 2e-7


def test_physical_right_transport_is_jittable_and_preserves_norm() -> None:
    state, links, _ = _stage18_fields()
    expected = sm_family_physical_right_gauged_dirac_step(state, links)
    actual = jax.jit(sm_family_physical_right_gauged_dirac_step)(state, links)

    assert jnp.max(jnp.abs(actual - expected)) < 2e-7
    assert jnp.abs(state_norm_squared(actual) - state_norm_squared(state)) < 1e-6
