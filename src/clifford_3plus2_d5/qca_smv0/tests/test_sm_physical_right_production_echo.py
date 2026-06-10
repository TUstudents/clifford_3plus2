"""Tests for the production Loschmidt echo diagnostic."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo import (
    sm_physical_right_production_apex_momentum_perturbation,
    sm_physical_right_production_echo_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_physical_right_production_echo_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_echo_diagnostics()

    assert diagnostics.steps >= 2
    assert diagnostics.perturbation_size >= 9e-5
    assert diagnostics.base_roundtrip_residual < 1e-5
    assert 5e-5 <= diagnostics.apex_perturbation_norm < 2e-4
    assert 1e-4 <= diagnostics.double_apex_perturbation_norm < 4e-4
    assert 5e-5 <= diagnostics.echo_residual < 2e-4
    assert 1e-4 <= diagnostics.double_echo_residual < 4e-4
    assert 0.25 <= diagnostics.echo_amplification < 4.0
    assert 1.5 <= diagnostics.double_echo_ratio < 2.5
    assert diagnostics.double_echo_linearity_residual < 0.25
    assert diagnostics.perturbed_inverse_sm_link_unitarity_residual < 1e-5
    assert diagnostics.perturbed_inverse_higgs_link_unitarity_residual < 1e-5


def test_physical_right_production_echo_validates_inputs() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_apex_momentum_perturbation(state, epsilon=0.0)

    with pytest.raises(ValueError, match="steps must be positive"):
        sm_physical_right_production_echo_diagnostics(steps=0)
