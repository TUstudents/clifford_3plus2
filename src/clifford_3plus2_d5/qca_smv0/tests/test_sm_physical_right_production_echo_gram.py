"""Tests for the production echo Gram diagnostic."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_gram import (
    sm_physical_right_production_apex_family_perturbation,
    sm_physical_right_production_echo_gram_diagnostics,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    sm_physical_right_production_initial_state,
)


def test_physical_right_production_echo_gram_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_echo_gram_diagnostics()

    assert diagnostics.basis_count >= 3
    assert diagnostics.perturbation_size >= 9e-5
    assert diagnostics.base_roundtrip_residual < 1e-5
    assert 5e-5 <= diagnostics.min_echo_norm < 2e-4
    assert 5e-5 <= diagnostics.max_echo_norm < 2e-4
    assert diagnostics.gram_symmetry_residual < 1e-12
    assert diagnostics.gram_min_eigenvalue > 1e-10
    assert diagnostics.gram_max_eigenvalue < 1e-6
    assert diagnostics.gram_condition_number < 10.0
    assert diagnostics.max_offdiag_correlation < 0.1
    assert diagnostics.max_inverse_sm_link_unitarity_residual < 1e-5
    assert diagnostics.max_inverse_higgs_link_unitarity_residual < 1e-5


def test_physical_right_production_echo_gram_validates_inputs() -> None:
    state = sm_physical_right_production_initial_state()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_apex_family_perturbation(state, epsilon=0.0)

    with pytest.raises(ValueError, match="steps must be positive"):
        sm_physical_right_production_echo_gram_diagnostics(steps=0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_echo_gram_diagnostics(epsilon=0.0)
