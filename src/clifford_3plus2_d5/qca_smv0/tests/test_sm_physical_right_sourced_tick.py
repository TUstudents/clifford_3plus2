"""Tests for QCA_SMv0 physical-right sourced SM tick."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_antiunitary_bridge import (
    sm_antiunitary_bridge_gauge_from_transport,
    sm_physical_right_algebra_matrix_field,
)
from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_family_gauge import sm_transform_family_gauge_state
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import deterministic_sm_family_state
from clifford_3plus2_d5.qca_smv0.sm_family_sourced_tick import (
    sm_family_sourced_link_force,
    sm_family_sourced_sm_tick,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_site_theta_from_higgs_site_theta
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    deterministic_higgs_theta,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
    sm_higgs_site_gauge_from_algebra,
    sm_identity_higgs_links,
    sm_transform_higgs_field,
    sm_transform_higgs_links,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_physical_right_project_to_coordinates,
    sm_physical_right_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_sourced_tick import (
    sm_apply_physical_right_sourced_sm_momentum_kick,
    sm_physical_right_sourced_gauss_constraint,
    sm_physical_right_sourced_link_force,
    sm_physical_right_sourced_sm_tick,
    sm_physical_right_sourced_tick_diagnostics,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def _stage20_fields(lattice_shape: tuple[int, int, int] = (1, 1, 1)):
    state = deterministic_sm_family_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    return state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links


def test_physical_right_sourced_force_zero_control_and_nonzero_control() -> None:
    lattice_shape = (1, 1, 1)
    state, higgs, _, sm_links, _, higgs_links = _stage20_fields(lattice_shape)
    zero_state = jnp.zeros_like(state)
    vacuum = sm_constant_higgs(lattice_shape)
    identity_sm_links = sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32))
    identity_higgs_links = sm_identity_higgs_links(lattice_shape)

    assert (
        jnp.linalg.norm(sm_physical_right_sourced_link_force(zero_state, vacuum, identity_sm_links, identity_higgs_links))
        < 5e-8
    )
    assert jnp.linalg.norm(sm_physical_right_sourced_link_force(state, higgs, sm_links, higgs_links)) > 1e-4


def test_physical_right_sourced_force_differs_from_transport_convention_force() -> None:
    state, higgs, _, sm_links, _, higgs_links = _stage20_fields()

    physical_force = sm_physical_right_sourced_link_force(state, higgs, sm_links, higgs_links)
    transport_force = sm_family_sourced_link_force(state, higgs, sm_links, higgs_links)

    assert jnp.linalg.norm(physical_force - transport_force) > 1e-2


def test_physical_right_sourced_force_and_gauss_are_bridge_covariant() -> None:
    lattice_shape = (1, 1, 1)
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage20_fields(lattice_shape)
    higgs_site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.08)
    higgs_site_gauge = sm_higgs_site_gauge_from_algebra(higgs_site_theta)
    sm_site_gauge = sm_site_gauge_from_algebra(sm_site_theta_from_higgs_site_theta(higgs_site_theta))
    physical_site_gauge = sm_antiunitary_bridge_gauge_from_transport(sm_site_gauge)

    transformed_state = sm_transform_family_gauge_state(state, physical_site_gauge)
    transformed_higgs = sm_transform_higgs_field(higgs, higgs_site_gauge)
    transformed_higgs_momenta = sm_transform_higgs_field(higgs_momenta, higgs_site_gauge)
    transformed_sm_links = sm_transform_links(sm_links, sm_site_gauge)
    transformed_higgs_links = sm_transform_higgs_links(higgs_links, higgs_site_gauge)
    transformed_sm_momenta = sm_physical_right_transform_momenta(sm_momenta, sm_site_gauge)

    force = sm_physical_right_sourced_link_force(state, higgs, sm_links, higgs_links)
    transformed_force = sm_physical_right_sourced_link_force(
        transformed_state,
        transformed_higgs,
        transformed_sm_links,
        transformed_higgs_links,
    )
    expected_force = sm_physical_right_transform_momenta(force, sm_site_gauge)
    gauss = sm_physical_right_sourced_gauss_constraint(sm_links, sm_momenta, state, higgs, higgs_momenta)
    transformed_gauss = sm_physical_right_sourced_gauss_constraint(
        transformed_sm_links,
        transformed_sm_momenta,
        transformed_state,
        transformed_higgs,
        transformed_higgs_momenta,
    )
    expected_gauss = sm_physical_right_project_to_coordinates(
        physical_site_gauge
        @ sm_physical_right_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(physical_site_gauge), -1, -2),
    )

    assert jnp.max(jnp.abs(transformed_force - expected_force)) < 5e-5
    assert jnp.max(jnp.abs(transformed_gauss - expected_gauss)) < 5e-7


def test_physical_right_sourced_gauss_zero_control() -> None:
    lattice_shape = (1, 1, 1)
    state = jnp.zeros_like(deterministic_sm_family_state(lattice_shape))
    vacuum = sm_constant_higgs(lattice_shape)
    zero_higgs_momenta = jnp.zeros_like(vacuum)
    identity_sm_links = sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32))
    zero_sm_momenta = jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)

    assert (
        jnp.linalg.norm(
            sm_physical_right_sourced_gauss_constraint(
                identity_sm_links,
                zero_sm_momenta,
                state,
                vacuum,
                zero_higgs_momenta,
            ),
        )
        < 1e-7
    )


def test_physical_right_sourced_kick_is_reversible() -> None:
    state, higgs, _, sm_links, sm_momenta, higgs_links = _stage20_fields()

    kicked = sm_apply_physical_right_sourced_sm_momentum_kick(
        sm_momenta,
        state,
        higgs,
        sm_links,
        higgs_links,
        step_size=0.003,
    )
    restored = sm_apply_physical_right_sourced_sm_momentum_kick(
        kicked,
        state,
        higgs,
        sm_links,
        higgs_links,
        step_size=-0.003,
    )

    assert jnp.linalg.norm(kicked - sm_momenta) > 1e-6
    assert jnp.max(jnp.abs(restored - sm_momenta)) < 1e-8


def test_physical_right_sourced_sm_tick_preserves_norm_and_updates_fields() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage20_fields()

    updated_state, updated_higgs, updated_higgs_momenta, updated_sm_links, updated_sm_momenta, updated_higgs_links = (
        sm_physical_right_sourced_sm_tick(
            state,
            higgs,
            higgs_momenta,
            sm_links,
            sm_momenta,
            higgs_links,
            step_size=0.003,
        )
    )

    assert updated_state.shape == state.shape
    assert updated_higgs.shape == higgs.shape
    assert updated_higgs_momenta.shape == higgs_momenta.shape
    assert updated_sm_links.shape == sm_links.shape
    assert updated_sm_momenta.shape == sm_momenta.shape
    assert updated_higgs_links.shape == higgs_links.shape
    assert jnp.abs(state_norm_squared(updated_state) - state_norm_squared(state)) < 5e-6
    assert jnp.linalg.norm(updated_higgs - higgs) > 1e-6
    assert sm_link_unitarity_residual(updated_sm_links) < 7e-7
    assert sm_higgs_link_unitarity_residual(updated_higgs_links) < 7e-7


def test_physical_right_sourced_tick_differs_from_transport_convention_tick() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage20_fields()

    physical_tick = sm_physical_right_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    transport_tick = sm_family_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )

    assert jnp.linalg.norm(physical_tick[0] - transport_tick[0]) > 1e-3


def test_physical_right_sourced_tick_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_sourced_tick_diagnostics()

    assert diagnostics.transport_force_difference_norm > 1e-2
    assert diagnostics.zero_source_force_norm < 5e-8
    assert diagnostics.nonzero_source_force_norm > 1e-4
    assert diagnostics.force_covariance_residual < 5e-5
    assert diagnostics.gauss_covariance_residual < 5e-7
    assert diagnostics.gauss_zero_residual < 1e-7
    assert diagnostics.kick_delta_norm > 1e-6
    assert diagnostics.kick_reversibility_residual < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 7e-7
    assert diagnostics.higgs_link_unitarity_residual < 7e-7
    assert diagnostics.family_norm_drift < 5e-6
    assert diagnostics.higgs_field_delta_norm > 1e-6
    assert diagnostics.transport_tick_state_difference_norm > 1e-3
    assert diagnostics.jit_delta_family_state < 2e-7
    assert diagnostics.jit_delta_sm_links < 1e-8
    assert diagnostics.jit_delta_higgs_links < 1e-8
    assert diagnostics.jit_delta_sm_momenta < 2e-7
    assert diagnostics.jit_delta_higgs_field < 1e-8
    assert diagnostics.jit_delta_higgs_momenta < 1e-8


def test_physical_right_sourced_sm_tick_is_jittable() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage20_fields()
    expected = sm_physical_right_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    actual = jax.jit(
        sm_physical_right_sourced_sm_tick,
        static_argnames=(
            "step_size",
            "beta",
            "parameters",
            "wilson_epsilon",
            "higgs_force_epsilon",
            "fermion_current_epsilon",
        ),
    )(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )

    for expected_item, actual_item in zip(expected, actual, strict=True):
        assert jnp.max(jnp.abs(actual_item - expected_item)) < 2e-7
