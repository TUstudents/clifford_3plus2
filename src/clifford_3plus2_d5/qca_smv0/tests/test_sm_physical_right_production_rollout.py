"""Tests for QCA_SMv0 physical-right production rollout."""

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_observables,
    sm_physical_right_production_recorded_rollout,
    sm_physical_right_production_rollout,
    sm_physical_right_production_rollout_diagnostics,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tick import (
    sm_physical_right_production_sm_tick,
)
from clifford_3plus2_d5.sim.state import state_norm_squared


def _assert_rollout_state_close(
    actual: PhysicalRightProductionRolloutState,
    expected: PhysicalRightProductionRolloutState,
    *,
    atol: float,
) -> None:
    assert jnp.max(jnp.abs(actual.family_state - expected.family_state)) < atol
    assert jnp.max(jnp.abs(actual.higgs - expected.higgs)) < atol
    assert jnp.max(jnp.abs(actual.higgs_momenta - expected.higgs_momenta)) < atol
    assert jnp.max(jnp.abs(actual.sm_links - expected.sm_links)) < atol
    assert jnp.max(jnp.abs(actual.sm_momenta - expected.sm_momenta)) < atol
    assert jnp.max(jnp.abs(actual.higgs_links - expected.higgs_links)) < atol


def test_physical_right_production_initial_state_shapes_and_observables() -> None:
    initial = sm_physical_right_production_initial_state()
    observations = sm_physical_right_production_observables(initial)

    assert initial.family_state.shape == (1, 1, 1, 4, 32, 3)
    assert initial.higgs.shape == (1, 1, 1, 2)
    assert initial.higgs_momenta.shape == (1, 1, 1, 2)
    assert initial.sm_links.shape == (1, 1, 1, 8, 32, 32)
    assert initial.sm_momenta.shape == (1, 1, 1, 8, 12)
    assert initial.higgs_links.shape == (1, 1, 1, 8, 2, 2)
    assert tuple(observations.keys()) == (
        "family_norm",
        "higgs_norm",
        "higgs_momentum_norm",
        "sm_momentum_norm",
        "sm_link_unitarity_residual",
        "higgs_link_unitarity_residual",
    )
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in observations.values())


def test_one_step_rollout_matches_direct_stage21_tick() -> None:
    initial = sm_physical_right_production_initial_state()
    expected = sm_physical_right_production_sm_tick(
        initial.family_state,
        initial.higgs,
        initial.higgs_momenta,
        initial.sm_links,
        initial.sm_momenta,
        initial.higgs_links,
        step_size=0.001,
    )
    actual = sm_physical_right_production_rollout(initial, steps=1, step_size=0.001)

    _assert_rollout_state_close(actual, PhysicalRightProductionRolloutState(*expected), atol=2e-7)


def test_recorded_rollout_records_sparse_steps_and_preserves_controls() -> None:
    initial = sm_physical_right_production_initial_state()
    result = sm_physical_right_production_recorded_rollout(
        initial,
        steps=4,
        record_every=2,
        step_size=0.001,
    )

    assert tuple(int(index) for index in result.step_indices) == (0, 2, 4)
    assert bool(result.all_finite)
    assert result.observations["family_norm"].shape == (3,)
    assert jnp.abs(state_norm_squared(result.final_state.family_state) - state_norm_squared(initial.family_state)) < 2e-5
    assert jnp.max(result.observations["sm_link_unitarity_residual"]) < 8e-7
    assert jnp.max(result.observations["higgs_link_unitarity_residual"]) < 8e-7
    assert jnp.linalg.norm(result.final_state.higgs - initial.higgs) > 1e-7
    assert jnp.linalg.norm(result.final_state.sm_momenta - initial.sm_momenta) > 1e-7


def test_physical_right_production_step_wrapper_matches_rollout_step() -> None:
    initial = sm_physical_right_production_initial_state()
    expected = sm_physical_right_production_step(initial, step_size=0.001)
    actual = sm_physical_right_production_rollout(initial, steps=1, step_size=0.001)

    _assert_rollout_state_close(actual, expected, atol=2e-7)


def test_physical_right_production_rollout_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_rollout_diagnostics()

    assert diagnostics.one_step_direct_residual < 2e-7
    assert diagnostics.loop_scan_final_residual < 5e-7
    assert diagnostics.loop_scan_observation_residual < 1e-6
    assert diagnostics.rollout_family_norm_drift < 1e-5
    assert diagnostics.rollout_max_family_norm_drift < 1e-5
    assert diagnostics.max_sm_link_unitarity_residual < 8e-7
    assert diagnostics.max_higgs_link_unitarity_residual < 8e-7
    assert diagnostics.higgs_field_total_delta_norm > 1e-7
    assert diagnostics.higgs_momentum_total_delta_norm > 1e-7
    assert diagnostics.sm_momentum_total_delta_norm > 1e-7
    assert diagnostics.production_zero_yukawa_family_difference_norm > 1e-6
    assert diagnostics.production_zero_yukawa_higgs_momentum_difference_norm > 1e-6
    assert diagnostics.record_count == 3
    assert bool(diagnostics.scan_all_finite)
