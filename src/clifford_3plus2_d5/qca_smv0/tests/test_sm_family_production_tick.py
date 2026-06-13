"""Tests for QCA_SMv0 full family production tick."""

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import deterministic_sm_momenta
from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_family_fn_production_initial_state,
    sm_family_fn_production_higgs_force,
    sm_family_fn_production_rollout,
    sm_family_fn_production_sm_tick,
    sm_family_fn_production_step,
    sm_apply_family_production_higgs_momentum_kick,
    sm_family_production_higgs_force,
    sm_family_production_sm_tick,
    sm_family_production_tick_diagnostics,
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    sm_family_calibrated_quark_path_readouts,
    sm_family_fn_quark_path_higgs_force,
    sm_family_recirculated_quark_yukawas,
    sm_zero_family_fn_quark_path_state_aux,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
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


def test_fn_production_higgs_force_includes_recirculated_quark_paths() -> None:
    state, higgs, _, _, _, higgs_links = _stage15_fields()
    aux = sm_zero_family_fn_quark_path_state_aux(state.shape[:3])
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()

    baseline = sm_family_production_higgs_force(
        state,
        higgs,
        higgs_links,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )
    quark_force = sm_family_fn_quark_path_higgs_force(state, higgs, aux)
    combined = sm_family_fn_production_higgs_force(
        state,
        higgs,
        higgs_links,
        aux,
        lepton_yukawas=zero_leptons,
    )

    assert jnp.linalg.norm(quark_force) > 1e-7
    assert jnp.max(jnp.abs(combined - baseline - quark_force)) < 2e-7


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


def test_fn_production_tick_carries_quark_recirculation_aux_state() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage15_fields()
    aux = sm_zero_family_fn_quark_path_state_aux(state.shape[:3])

    updated = sm_family_fn_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        aux,
        step_size=0.003,
    )

    assert updated.state.shape == state.shape
    assert updated.higgs.shape == higgs.shape
    assert updated.higgs_momenta.shape == higgs_momenta.shape
    assert updated.sm_links.shape == sm_links.shape
    assert updated.sm_momenta.shape == sm_momenta.shape
    assert updated.higgs_links.shape == higgs_links.shape
    assert updated.fn_aux_state.up.shape == aux.up.shape
    assert updated.fn_aux_state.down.shape == aux.down.shape
    assert jnp.linalg.norm(updated.state - state) > 1e-6
    assert jnp.linalg.norm(updated.fn_aux_state.up) > 1e-6
    assert jnp.linalg.norm(updated.fn_aux_state.down) > 1e-6
    assert sm_link_unitarity_residual(updated.sm_links) < 7e-7
    assert sm_higgs_link_unitarity_residual(updated.higgs_links) < 7e-7


def test_fn_production_second_tick_depends_on_persistent_aux_memory() -> None:
    state, higgs, higgs_momenta, sm_links, sm_momenta, higgs_links = _stage15_fields()
    zero_aux = sm_zero_family_fn_quark_path_state_aux(state.shape[:3])
    first = sm_family_fn_production_sm_tick(
        state,
        higgs,
        higgs_momenta,
        sm_links,
        sm_momenta,
        higgs_links,
        zero_aux,
        step_size=0.003,
    )
    with_memory = sm_family_fn_production_sm_tick(
        first.state,
        first.higgs,
        first.higgs_momenta,
        first.sm_links,
        first.sm_momenta,
        first.higgs_links,
        first.fn_aux_state,
        step_size=0.003,
    )
    with_reset = sm_family_fn_production_sm_tick(
        first.state,
        first.higgs,
        first.higgs_momenta,
        first.sm_links,
        first.sm_momenta,
        first.higgs_links,
        zero_aux,
        step_size=0.003,
    )

    assert jnp.linalg.norm(with_memory.state - with_reset.state) > 1e-8
    assert jnp.linalg.norm(with_memory.fn_aux_state.up - with_reset.fn_aux_state.up) > 1e-8
    assert jnp.linalg.norm(with_memory.fn_aux_state.down - with_reset.fn_aux_state.down) > 1e-8


def test_fn_production_rollout_iterates_tick_with_aux_memory() -> None:
    initial = sm_family_fn_production_initial_state((1, 1, 1))
    one_direct = sm_family_fn_production_step(initial, step_size=0.003)
    one_rollout = sm_family_fn_production_rollout(initial, steps=1, step_size=0.003)
    two_direct = sm_family_fn_production_step(one_direct, step_size=0.003)
    two_rollout = sm_family_fn_production_rollout(initial, steps=2, step_size=0.003)
    zero_rollout = sm_family_fn_production_rollout(initial, steps=0, step_size=0.003)

    for actual, expected in (
        (one_rollout, one_direct),
        (two_rollout, two_direct),
        (zero_rollout, initial),
    ):
        assert jnp.max(jnp.abs(actual.state - expected.state)) < 1e-8
        assert jnp.max(jnp.abs(actual.higgs - expected.higgs)) < 1e-8
        assert jnp.max(jnp.abs(actual.higgs_momenta - expected.higgs_momenta)) < 1e-8
        assert jnp.max(jnp.abs(actual.sm_links - expected.sm_links)) < 1e-8
        assert jnp.max(jnp.abs(actual.sm_momenta - expected.sm_momenta)) < 1e-8
        assert jnp.max(jnp.abs(actual.higgs_links - expected.higgs_links)) < 1e-8
        assert jnp.max(jnp.abs(actual.fn_aux_state.up - expected.fn_aux_state.up)) < 1e-8
        assert jnp.max(jnp.abs(actual.fn_aux_state.down - expected.fn_aux_state.down)) < 1e-8
    assert jnp.linalg.norm(two_rollout.fn_aux_state.up) > 1e-6
    assert jnp.linalg.norm(two_rollout.fn_aux_state.down) > 1e-6
    assert jnp.linalg.norm(two_rollout.fn_aux_state.up - one_rollout.fn_aux_state.up) > 1e-6
    assert jnp.linalg.norm(two_rollout.fn_aux_state.down - one_rollout.fn_aux_state.down) > 1e-6


def test_fn_production_rollout_accepts_calibrated_path_readouts() -> None:
    initial = sm_family_fn_production_initial_state((1, 1, 1))
    default_yukawas = sm_family_recirculated_quark_yukawas()
    target_yukawas = FNQuarkYukawas(up=1.35 * default_yukawas.up, down=0.70 * default_yukawas.down)
    readouts = sm_family_calibrated_quark_path_readouts(target_yukawas)
    direct = sm_family_fn_production_step(initial, step_size=0.003, readouts=readouts)
    rollout = sm_family_fn_production_rollout(initial, steps=1, step_size=0.003, readouts=readouts)
    default_rollout = sm_family_fn_production_rollout(initial, steps=1, step_size=0.003)

    assert jnp.max(jnp.abs(rollout.state - direct.state)) < 1e-8
    assert jnp.max(jnp.abs(rollout.fn_aux_state.up - direct.fn_aux_state.up)) < 1e-8
    assert jnp.max(jnp.abs(rollout.fn_aux_state.down - direct.fn_aux_state.down)) < 1e-8
    assert jnp.linalg.norm(rollout.state - default_rollout.state) > 1e-7
    assert jnp.linalg.norm(rollout.higgs_momenta - default_rollout.higgs_momenta) > 1e-7


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
