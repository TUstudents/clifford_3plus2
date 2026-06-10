"""Tests for finite tangent-response production echo."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_tangent import (
    sm_physical_right_production_apex_higgs_momentum_perturbation,
    sm_physical_right_production_tangent_diagnostics,
)


def test_physical_right_production_tangent_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_tangent_diagnostics()

    assert diagnostics.steps >= 2
    assert diagnostics.perturbation_size >= 9e-5
    assert diagnostics.base_roundtrip_residual < 1e-5
    assert 5e-5 <= diagnostics.sm_momentum_echo_norm < 2e-4
    assert 5e-5 <= diagnostics.higgs_momentum_echo_norm < 2e-4
    assert 5e-5 <= diagnostics.combined_echo_norm < 2e-4
    assert diagnostics.superposition_residual < 2e-6
    assert diagnostics.superposition_relative_residual < 2e-2
    assert diagnostics.combined_inverse_sm_link_unitarity_residual < 1e-5
    assert diagnostics.combined_inverse_higgs_link_unitarity_residual < 1e-5


def test_physical_right_production_tangent_validates_inputs() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_apex_higgs_momentum_perturbation(state, epsilon=0.0)

    with pytest.raises(ValueError, match="steps must be positive"):
        sm_physical_right_production_tangent_diagnostics(steps=0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_tangent_diagnostics(epsilon=0.0)
