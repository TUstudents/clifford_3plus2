"""Tests for QCA_SMv0 physical-right fermion current."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import sm_antiunitary_bridge_gauge_from_transport
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import (
    sm_family_fermion_left_gauge_current,
    sm_transform_family_gauge_state,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    sm_link_field_from_algebra,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_apply_physical_right_fermion_gauge_momentum_kick,
    sm_physical_right_current_diagnostics,
    sm_physical_right_fermion_gauge_kick_then_transport,
    sm_physical_right_fermion_left_gauge_current,
    sm_physical_right_streaming_energy_density,
    sm_physical_right_transform_momenta,
)


def _stage19_fields(lattice_shape: tuple[int, int, int] = (1, 1, 1)):
    state = deterministic_sm_family_state(lattice_shape)
    links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.24))
    momenta = deterministic_sm_momenta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape, scale=0.07))
    return state, links, momenta, site_gauge


def test_physical_right_streaming_energy_is_real() -> None:
    state, links, _, _ = _stage19_fields()

    energy = sm_physical_right_streaming_energy_density(state, links)

    assert jnp.abs(jnp.imag(energy + 0j)) < 1e-7


def test_physical_right_zero_state_current_vanishes_and_nonzero_current_is_nonzero() -> None:
    state, links, _, _ = _stage19_fields()

    zero_current = sm_physical_right_fermion_left_gauge_current(jnp.zeros_like(state), links)
    current = sm_physical_right_fermion_left_gauge_current(state, links)

    assert jnp.linalg.norm(zero_current) < 1e-7
    assert jnp.linalg.norm(current) > 1e-2


def test_physical_right_current_differs_from_transport_current() -> None:
    state, links, _, _ = _stage19_fields()

    physical_right_current = sm_physical_right_fermion_left_gauge_current(state, links)
    transport_current = sm_family_fermion_left_gauge_current(state, links)

    assert jnp.linalg.norm(physical_right_current - transport_current) > 1e-2


def test_physical_right_current_transforms_in_bridge_adjoint() -> None:
    state, links, _, site_gauge = _stage19_fields()
    current = sm_physical_right_fermion_left_gauge_current(state, links)
    expected = sm_physical_right_transform_momenta(current, site_gauge)

    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(site_gauge)
    transformed_state = sm_transform_family_gauge_state(state, physical_site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_current = sm_physical_right_fermion_left_gauge_current(transformed_state, transformed_links)

    assert jnp.max(jnp.abs(transformed_current - expected)) < 7e-6


def test_physical_right_momentum_kick_is_reversible() -> None:
    state, links, momenta, _ = _stage19_fields()
    step_size = 0.004

    kicked = sm_apply_physical_right_fermion_gauge_momentum_kick(momenta, state, links, step_size=step_size)
    restored = sm_apply_physical_right_fermion_gauge_momentum_kick(kicked, state, links, step_size=-step_size)

    assert jnp.linalg.norm(kicked - momenta) > 1e-4
    assert jnp.max(jnp.abs(restored - momenta)) < 3e-10


def test_physical_right_kick_then_transport_is_jittable() -> None:
    state, links, momenta, _ = _stage19_fields()

    expected = sm_physical_right_fermion_gauge_kick_then_transport(state, links, momenta, step_size=0.004)
    actual = jax.jit(sm_physical_right_fermion_gauge_kick_then_transport, static_argnames=("step_size",))(
        state,
        links,
        momenta,
        step_size=0.004,
    )

    for expected_item, actual_item in zip(expected, actual, strict=True):
        assert jnp.max(jnp.abs(expected_item - actual_item)) < 2e-7


def test_physical_right_current_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_current_diagnostics()

    assert diagnostics.streaming_energy_reality_residual < 1e-7
    assert diagnostics.zero_state_current_norm < 1e-7
    assert diagnostics.nonzero_current_norm > 1e-2
    assert diagnostics.transport_current_difference_norm > 1e-2
    assert diagnostics.streaming_energy_covariance_residual < 1e-6
    assert diagnostics.current_covariance_residual < 7e-6
    assert diagnostics.charge_covariance_residual < 3e-7
    assert diagnostics.gauss_covariance_residual < 3e-7
    assert diagnostics.momentum_kick_delta_norm > 1e-4
    assert diagnostics.momentum_kick_reversibility_residual < 3e-10
    assert diagnostics.kicked_link_unitarity_residual < 7e-7
    assert diagnostics.spectator_norm_drift_after_kick < 8e-6
    assert diagnostics.jit_delta_current < 7e-6
    assert diagnostics.jit_delta_transport < 2e-7
