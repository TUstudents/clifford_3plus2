"""Tests for QCA_SMv0 full family production tick."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_apply_family_production_higgs_momentum_kick,
    sm_family_production_higgs_force,
    sm_family_production_sm_tick,
    sm_family_production_tick_diagnostics,
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_family_sourced_tick import sm_family_sourced_sm_tick
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import deterministic_yukawa_source_state, sm_yukawa_higgs_force
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    deterministic_sm_link_theta,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    deterministic_higgs_field,
    deterministic_higgs_momenta,
    deterministic_higgs_theta,
    sm_higgs_force,
    sm_higgs_link_field_from_algebra,
    sm_higgs_link_unitarity_residual,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def _stage15_fields(lattice_shape: tuple[int, int, int] = (1, 1, 1)):
    state = deterministic_yukawa_source_state(lattice_shape)
    higgs = deterministic_higgs_field(lattice_shape)
    higgs_momenta = deterministic_higgs_momenta(lattice_shape)
    sm_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=0.25))
    sm_momenta = deterministic_sm_momenta(lattice_shape)
    higgs_links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    return state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links


def test_zero_yukawas_reduce_production_tick_to_stage14_tick() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage15_fields()
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()

    expected = sm_family_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    actual = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )

    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert jnp.max(jnp.abs(actual_part - expected_part)) < 2e-7


def test_production_higgs_force_adds_yukawa_source() -> None:
    state, higgs, _, _, _, higgs_links = _stage15_fields()

    production_force = sm_family_production_higgs_force(state, higgs, higgs_links)
    expected_force = sm_higgs_force(higgs, higgs_links) + sm_yukawa_higgs_force(state, higgs)

    assert jnp.max(jnp.abs(production_force - expected_force)) < 1e-7
    assert jnp.linalg.norm(sm_yukawa_higgs_force(state, higgs)) > 1e-4


def test_zero_state_vacuum_has_zero_yukawa_source() -> None:
    state, _, _, _, _, higgs_links = _stage15_fields()
    zero_state = jnp.zeros_like(state)
    vacuum = sm_constant_higgs((1, 1, 1))

    assert jnp.linalg.norm(sm_yukawa_higgs_force(zero_state, vacuum)) < 1e-7
    assert jnp.linalg.norm(sm_family_production_higgs_force(zero_state, vacuum, higgs_links)) > 1e-6


def test_production_higgs_momentum_kick_is_reversible_for_frozen_fields() -> None:
    state, higgs, higgs_momenta, _, _, higgs_links = _stage15_fields()

    kicked = sm_apply_family_production_higgs_momentum_kick(
        higgs_momenta,
        state,
        higgs,
        higgs_links,
        step_size=0.003,
    )
    restored = sm_apply_family_production_higgs_momentum_kick(
        kicked,
        state,
        higgs,
        higgs_links,
        step_size=-0.003,
    )

    assert jnp.linalg.norm(kicked - higgs_momenta) > 1e-6
    assert jnp.max(jnp.abs(restored - higgs_momenta)) < 1e-8


def test_production_tick_preserves_norm_and_updates_local_yukawa_sector() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage15_fields()

    production = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    stage14 = sm_family_sourced_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )

    assert production[0].shape == state.shape
    assert production[1].shape == higgs.shape
    assert production[2].shape == higgs_momenta.shape
    assert production[3].shape == sm_links.shape
    assert production[4].shape == sm_momenta.shape
    assert production[5].shape == higgs_links.shape
    assert jnp.abs(state_norm_squared(production[0]) - state_norm_squared(state)) < 5e-6
    assert jnp.linalg.norm(production[0] - stage14[0]) > 1e-6
    assert jnp.linalg.norm(production[2] - stage14[2]) > 1e-6
    assert sm_link_unitarity_residual(production[3]) < 7e-7
    assert sm_higgs_link_unitarity_residual(production[5]) < 7e-7


def test_family_production_tick_diagnostics_and_jit_pass_stage_thresholds() -> None:
    diagnostics = sm_family_production_tick_diagnostics()

    assert diagnostics.zero_yukawa_state_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_higgs_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_higgs_momentum_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_sm_link_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_sm_momentum_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_higgs_link_reduction_residual < 2e-7
    assert diagnostics.zero_yukawa_source_norm < 1e-7
    assert diagnostics.nonzero_yukawa_source_norm > 1e-4
    assert diagnostics.production_state_delta_norm > 1e-6
    assert diagnostics.production_higgs_momentum_delta_norm > 1e-6
    assert diagnostics.source_kick_reversibility_residual < 1e-8
    assert diagnostics.production_family_norm_drift < 5e-6
    assert diagnostics.sm_link_unitarity_residual < 7e-7
    assert diagnostics.higgs_link_unitarity_residual < 7e-7
    assert diagnostics.jit_delta_family_state < 1e-6
    assert diagnostics.jit_delta_higgs_field < 2e-7
    assert diagnostics.jit_delta_higgs_momenta < 2e-7
    assert diagnostics.jit_delta_sm_links < 2e-7
    assert diagnostics.jit_delta_sm_momenta < 2e-7
    assert diagnostics.jit_delta_higgs_links < 2e-7


def test_family_production_tick_is_jittable() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage15_fields()
    expected = sm_family_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        step_size=0.003,
    )
    jitted = jax.jit(
        sm_family_production_sm_tick,
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
        assert jnp.max(jnp.abs(actual_part - expected_part)) < 1e-6
