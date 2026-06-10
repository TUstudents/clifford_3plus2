"""Tests for production trajectory reversibility."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_reversibility import (
    sm_physical_right_production_inverse_rollout,
    sm_physical_right_production_reversibility_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
    sm_physical_right_production_rollout,
)


def _state_delta(left, right) -> jnp.ndarray:
    return jnp.max(
        jnp.asarray(
            [
                jnp.max(jnp.abs(left.family_state - right.family_state)),
                jnp.max(jnp.abs(left.higgs - right.higgs)),
                jnp.max(jnp.abs(left.higgs_momenta - right.higgs_momenta)),
                jnp.max(jnp.abs(left.sm_links - right.sm_links)),
                jnp.max(jnp.abs(left.sm_momenta - right.sm_momenta)),
                jnp.max(jnp.abs(left.higgs_links - right.higgs_links)),
            ],
        ),
    )


def test_physical_right_production_inverse_rollout_restores_multistep_trajectory() -> None:
    initial = sm_physical_right_production_initial_state()
    step_size = 1e-3
    forward = sm_physical_right_production_rollout(initial, steps=2, step_size=step_size)
    restored = sm_physical_right_production_inverse_rollout(forward, steps=2, step_size=step_size)

    assert _state_delta(initial, restored) < 1e-5


def test_physical_right_production_reversibility_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_reversibility_diagnostics()

    assert diagnostics.steps >= 2
    assert diagnostics.trajectory_roundtrip_residual < 1e-5
    assert diagnostics.trajectory_replay_residual < 1e-5
    assert diagnostics.max_path_restore_residual < 1e-5
    assert diagnostics.inverse_family_norm_drift < 1e-5
    assert diagnostics.forward_family_norm_drift < 1e-5
    assert diagnostics.max_forward_sm_link_unitarity_residual < 1e-5
    assert diagnostics.max_forward_higgs_link_unitarity_residual < 1e-5
    assert diagnostics.max_inverse_sm_link_unitarity_residual < 1e-5
    assert diagnostics.max_inverse_higgs_link_unitarity_residual < 1e-5
    assert diagnostics.naive_negative_trajectory_residual > 1e-2
    assert diagnostics.trajectory_inverse_improvement_ratio > 1e3
    assert diagnostics.jit_inverse_rollout_delta < 1e-5


def test_physical_right_production_inverse_rollout_validates_steps() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        sm_physical_right_production_inverse_rollout(state, steps=-1, step_size=1e-3)

    with pytest.raises(ValueError, match="steps must be positive"):
        sm_physical_right_production_reversibility_diagnostics(steps=0)
