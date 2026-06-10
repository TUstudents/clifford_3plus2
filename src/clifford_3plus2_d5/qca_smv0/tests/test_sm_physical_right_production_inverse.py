"""Tests for the explicit physical-right production inverse."""

import jax.numpy as jnp
import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_inverse import (
    sm_physical_right_production_inverse_diagnostics,
    sm_physical_right_production_inverse_step,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
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


def test_physical_right_production_inverse_roundtrips_forward_tick() -> None:
    initial = sm_physical_right_production_initial_state()
    step_size = 1e-3
    forward = sm_physical_right_production_step(initial, step_size=step_size)
    restored = sm_physical_right_production_inverse_step(forward, step_size=step_size)

    assert _state_delta(initial, restored) < 5e-6


def test_physical_right_production_inverse_replays_forward_state() -> None:
    initial = sm_physical_right_production_initial_state()
    step_size = 1e-3
    forward = sm_physical_right_production_step(initial, step_size=step_size)
    restored = sm_physical_right_production_inverse_step(forward, step_size=step_size)
    replayed = sm_physical_right_production_step(restored, step_size=step_size)

    assert _state_delta(forward, replayed) < 5e-6


def test_physical_right_production_inverse_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_inverse_diagnostics()

    assert diagnostics.inverse_roundtrip_residual < 5e-6
    assert diagnostics.forward_inverse_forward_residual < 5e-6
    assert diagnostics.inverse_family_norm_drift < 1e-5
    assert diagnostics.forward_family_norm_drift < 1e-5
    assert diagnostics.inverse_sm_link_unitarity_residual < 1e-5
    assert diagnostics.inverse_higgs_link_unitarity_residual < 1e-5
    assert diagnostics.naive_negative_residual > 1e-2
    assert diagnostics.explicit_inverse_improvement_ratio > 1e3
    assert diagnostics.jit_delta_family_state < 1e-5
    assert diagnostics.jit_delta_sm_momenta < 1e-6


def test_physical_right_production_inverse_validates_inputs() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="family SM Dirac state"):
        sm_physical_right_production_inverse_step(state._replace(family_state=state.family_state[..., 0]), step_size=1e-3)
