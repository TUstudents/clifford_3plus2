"""Tests for QCA_SMv0 coupled sourced SM tick."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import (
    deterministic_sm_momenta,
    sm_project_to_coordinates,
    sm_transform_momenta,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_site_theta_from_higgs_site_theta
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_GENERATOR_COUNT,
    deterministic_sm_link_theta,
    deterministic_sm_state,
    sm_algebra_matrix_field,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_site_gauge_from_algebra,
    sm_transform_links,
    sm_transform_state,
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
from clifford_3plus2_d5.qca_smv0.sm_sourced_tick import (
    sm_apply_sourced_link_update,
    sm_apply_sourced_sm_momentum_kick,
    sm_embed_higgs_site_to_sm_coordinates,
    sm_sourced_gauss_constraint,
    sm_sourced_link_force,
    sm_sourced_sm_tick,
    sm_sourced_tick_diagnostics,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def test_higgs_site_embedding_places_only_electroweak_components() -> None:
    lattice_shape = (1, 1, 1)
    coordinates = deterministic_higgs_site_theta(lattice_shape)

    embedded = sm_embed_higgs_site_to_sm_coordinates(coordinates)

    assert embedded.shape == (*lattice_shape, SM_GENERATOR_COUNT)
    assert jnp.max(jnp.abs(embedded[..., :8])) < 1e-8
    assert jnp.max(jnp.abs(embedded[..., 8:12] - coordinates)) < 1e-8


def test_sourced_force_zero_control_and_nonzero_control() -> None:
    lattice_shape = (1, 1, 1)
    zero_state = jnp.zeros_like(deterministic_sm_state(lattice_shape))
    vacuum = sm_constant_higgs(lattice_shape)
    identity_sm_links = sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32))
    identity_higgs_links = sm_identity_higgs_links(lattice_shape)
    state = deterministic_sm_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))

    assert jnp.linalg.norm(sm_sourced_link_force(zero_state, vacuum, identity_sm_links, identity_higgs_links)) < 5e-8
    assert jnp.linalg.norm(sm_sourced_link_force(state, higgs, sm_links, higgs_links)) > 1e-4


def test_sourced_force_and_gauss_are_gauge_covariant() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    higgs_site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.08)
    higgs_site_gauge = sm_higgs_site_gauge_from_algebra(higgs_site_theta)
    sm_site_gauge = sm_site_gauge_from_algebra(sm_site_theta_from_higgs_site_theta(higgs_site_theta))

    transformed_state = sm_transform_state(state, sm_site_gauge)
    transformed_higgs = sm_transform_higgs_field(higgs, higgs_site_gauge)
    transformed_higgs_momenta = sm_transform_higgs_field(higgs_momenta, higgs_site_gauge)
    transformed_sm_links = sm_transform_links(sm_links, sm_site_gauge)
    transformed_higgs_links = sm_transform_higgs_links(higgs_links, higgs_site_gauge)
    transformed_sm_momenta = sm_transform_momenta(sm_momenta, sm_site_gauge)
    force = sm_sourced_link_force(state, higgs, sm_links, higgs_links)
    transformed_force = sm_sourced_link_force(
        transformed_state,
        transformed_higgs,
        transformed_sm_links,
        transformed_higgs_links,
    )
    expected_force = sm_transform_momenta(force, sm_site_gauge)
    gauss = sm_sourced_gauss_constraint(sm_links, sm_momenta, state, higgs, higgs_momenta)
    transformed_gauss = sm_sourced_gauss_constraint(
        transformed_sm_links,
        transformed_sm_momenta,
        transformed_state,
        transformed_higgs,
        transformed_higgs_momenta,
    )
    expected_gauss = sm_project_to_coordinates(
        sm_site_gauge
        @ sm_algebra_matrix_field(gauss)
        @ jnp.swapaxes(jnp.conj(sm_site_gauge), -1, -2),
    )

    assert jnp.max(jnp.abs(transformed_force - expected_force)) < 5e-5
    assert jnp.max(jnp.abs(transformed_gauss - expected_gauss)) < 2e-6


def test_sourced_gauss_zero_control() -> None:
    lattice_shape = (1, 1, 1)
    zero_state = jnp.zeros_like(deterministic_sm_state(lattice_shape))
    vacuum = sm_constant_higgs(lattice_shape)
    zero_higgs_momenta = jnp.zeros_like(vacuum)
    identity_sm_links = sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32))
    zero_sm_momenta = jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)

    assert jnp.linalg.norm(
        sm_sourced_gauss_constraint(
            identity_sm_links,
            zero_sm_momenta,
            zero_state,
            vacuum,
            zero_higgs_momenta,
        ),
    ) < 1e-7


def test_sourced_kick_and_link_update_are_reversible_and_unitary() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    step_size = 0.003

    kicked = sm_apply_sourced_sm_momentum_kick(
        sm_momenta,
        state,
        higgs,
        sm_links,
        higgs_links,
        step_size=step_size,
    )
    restored = sm_apply_sourced_sm_momentum_kick(
        kicked,
        state,
        higgs,
        sm_links,
        higgs_links,
        step_size=-step_size,
    )
    updated_sm_links, updated_higgs_links = sm_apply_sourced_link_update(
        sm_links,
        higgs_links,
        kicked,
        step_size=step_size,
    )

    assert jnp.linalg.norm(kicked - sm_momenta) > 1e-6
    assert jnp.max(jnp.abs(restored - sm_momenta)) < 1e-8
    assert sm_link_unitarity_residual(updated_sm_links) < 7e-7
    assert sm_higgs_link_unitarity_residual(updated_higgs_links) < 7e-7


def test_sourced_sm_tick_preserves_fermion_norm_and_updates_fields() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))

    updated_state, updated_higgs, updated_higgs_momenta, updated_sm_links, updated_sm_momenta, updated_higgs_links = (
        sm_sourced_sm_tick(
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


def test_sourced_tick_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_sourced_tick_diagnostics()

    assert diagnostics.zero_source_force_norm < 5e-8
    assert diagnostics.nonzero_source_force_norm > 1e-4
    assert diagnostics.force_covariance_residual < 5e-5
    assert diagnostics.gauss_covariance_residual < 2e-6
    assert diagnostics.gauss_zero_residual < 1e-7
    assert diagnostics.kick_delta_norm > 1e-6
    assert diagnostics.kick_reversibility_residual < 1e-8
    assert diagnostics.sm_link_unitarity_residual < 7e-7
    assert diagnostics.higgs_link_unitarity_residual < 7e-7
    assert diagnostics.fermion_norm_drift < 5e-6
    assert diagnostics.higgs_field_delta_norm > 1e-6
    assert diagnostics.jit_delta_state < 2e-7
    assert diagnostics.jit_delta_sm_links < 1e-8
    assert diagnostics.jit_delta_higgs_links < 1e-8
    assert diagnostics.jit_delta_sm_momenta < 2e-7
    assert diagnostics.jit_delta_higgs_field < 1e-8
    assert diagnostics.jit_delta_higgs_momenta < 1e-8


def test_sourced_sm_tick_is_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    expected = sm_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    jitted = jax.jit(
        sm_sourced_sm_tick,
        static_argnames=(
            "step_size",
            "beta",
            "parameters",
            "wilson_epsilon",
            "higgs_force_epsilon",
            "fermion_current_epsilon",
        ),
    )
    actual = jitted(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.max(jnp.abs(actual_part - expected_part)) < 2e-7
