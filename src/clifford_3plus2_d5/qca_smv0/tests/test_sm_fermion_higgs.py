"""Tests for QCA_SMv0 local fermion/Higgs backreaction."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import (
    deterministic_yukawa_source_state,
    sm_apply_yukawa_higgs_momentum_kick,
    sm_family_yukawa_collision_with_higgs_kick,
    sm_fermion_higgs_backreaction_diagnostics,
    sm_transform_family_state,
    sm_yukawa_site_gauge_from_higgs_site_theta,
    sm_yukawa_energy_density,
    sm_yukawa_higgs_force,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import sm_family_recirculated_quark_yukawas
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_momenta,
    deterministic_higgs_site_theta,
    sm_higgs_site_gauge_from_algebra,
    sm_transform_higgs_field,
)


def test_yukawa_higgs_source_vanishes_for_zero_state_and_not_for_seed() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    zero_state = jnp.zeros_like(state)
    higgs = sm_constant_higgs(lattice_shape)

    assert jnp.linalg.norm(sm_yukawa_higgs_force(zero_state, higgs)) < 1e-8
    assert jnp.linalg.norm(sm_yukawa_higgs_force(state, higgs)) > 1e-4


def test_yukawa_higgs_source_defaults_to_fn_recirculation() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    quark_yukawas = sm_family_recirculated_quark_yukawas()

    default_energy = sm_yukawa_energy_density(state, higgs)
    explicit_energy = sm_yukawa_energy_density(state, higgs, quark_yukawas=quark_yukawas)
    default_force = sm_yukawa_higgs_force(state, higgs)
    explicit_force = sm_yukawa_higgs_force(state, higgs, quark_yukawas=quark_yukawas)

    assert jnp.abs(default_energy - explicit_energy) < 1e-8
    assert jnp.max(jnp.abs(default_force - explicit_force)) < 1e-8


def test_yukawa_energy_and_source_are_gauge_covariant() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    site_theta = deterministic_higgs_site_theta(lattice_shape, scale=0.17)
    higgs_gauge = sm_higgs_site_gauge_from_algebra(site_theta)
    sm_gauge = sm_yukawa_site_gauge_from_higgs_site_theta(site_theta)
    transformed_state = sm_transform_family_state(state, sm_gauge)
    transformed_higgs = sm_transform_higgs_field(higgs, higgs_gauge)

    energy = sm_yukawa_energy_density(state, higgs)
    transformed_energy = sm_yukawa_energy_density(transformed_state, transformed_higgs)
    source = sm_yukawa_higgs_force(state, higgs)
    transformed_source = sm_yukawa_higgs_force(transformed_state, transformed_higgs)
    expected_source = sm_transform_higgs_field(source, higgs_gauge)

    assert jnp.abs(transformed_energy - energy) < 5e-7
    assert jnp.max(jnp.abs(transformed_source - expected_source)) < 5e-7


def test_yukawa_higgs_momentum_kick_is_reversible() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)

    kicked = sm_apply_yukawa_higgs_momentum_kick(momenta, state, higgs, step_size=0.03)
    restored = sm_apply_yukawa_higgs_momentum_kick(kicked, state, higgs, step_size=-0.03)

    assert jnp.linalg.norm(kicked - momenta) > 1e-6
    assert jnp.max(jnp.abs(restored - momenta)) < 1e-8


def test_family_yukawa_collision_with_higgs_kick_preserves_fermion_norm() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)

    updated_state, updated_momenta = sm_family_yukawa_collision_with_higgs_kick(
        state,
        higgs,
        momenta,
        step_size=0.03,
    )

    assert updated_state.shape == state.shape
    assert updated_momenta.shape == momenta.shape
    assert jnp.abs(jnp.sum(jnp.abs(updated_state) ** 2) - jnp.sum(jnp.abs(state) ** 2)) < 1e-6
    assert jnp.linalg.norm(updated_momenta - momenta) > 1e-6


def test_fermion_higgs_backreaction_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_fermion_higgs_backreaction_diagnostics()

    assert diagnostics.fn_recirculated_source_residual < 1e-8
    assert diagnostics.energy_reality_residual < 1e-8
    assert diagnostics.zero_state_source_norm < 1e-8
    assert diagnostics.nonzero_source_norm > 1e-4
    assert diagnostics.energy_gauge_invariance_residual < 5e-7
    assert diagnostics.source_covariance_residual < 5e-7
    assert diagnostics.kick_delta_norm > 1e-6
    assert diagnostics.kick_reversibility_residual < 1e-8
    assert diagnostics.collision_norm_drift < 1e-6
    assert diagnostics.source_after_collision_norm > 1e-4
    assert diagnostics.jit_delta_source < 1e-8
    assert diagnostics.jit_delta_kick < 1e-8


def test_yukawa_higgs_source_and_kick_are_jittable() -> None:
    lattice_shape = (1, 1, 1)
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)
    expected_source = sm_yukawa_higgs_force(state, higgs)
    expected_kick = sm_apply_yukawa_higgs_momentum_kick(momenta, state, higgs, step_size=0.03)
    jitted_source = jax.jit(sm_yukawa_higgs_force)
    jitted_kick = jax.jit(sm_apply_yukawa_higgs_momentum_kick, static_argnames=("step_size",))

    assert jnp.max(jnp.abs(jitted_source(state, higgs) - expected_source)) < 1e-8
    assert jnp.max(jnp.abs(jitted_kick(momenta, state, higgs, step_size=0.03) - expected_kick)) < 1e-8
