"""Tests for echo-Gram scale stability."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_scale import (
    sm_physical_right_production_echo_scale_diagnostics,
)


def test_physical_right_production_echo_scale_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_echo_scale_diagnostics()

    assert diagnostics.small_epsilon >= 9e-5
    assert diagnostics.large_epsilon >= 1.9e-4
    assert 1.99 <= diagnostics.epsilon_ratio < 2.01
    assert diagnostics.small_base_roundtrip_residual < 1e-5
    assert diagnostics.large_base_roundtrip_residual < 1e-5
    assert 0.95 <= diagnostics.min_norm_scale_ratio < 1.05
    assert 0.95 <= diagnostics.max_norm_scale_ratio < 1.05
    assert 0.90 <= diagnostics.min_eigenvalue_scale_ratio < 1.10
    assert 0.90 <= diagnostics.max_eigenvalue_scale_ratio < 1.10
    assert diagnostics.condition_number_delta < 0.10
    assert diagnostics.offdiag_correlation_delta < 0.05
    assert diagnostics.max_link_unitarity_residual < 1e-5


def test_physical_right_production_echo_scale_validates_inputs() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_echo_scale_diagnostics(epsilon=0.0)

    with pytest.raises(ValueError, match="scale must be greater than 1"):
        sm_physical_right_production_echo_scale_diagnostics(scale=1.0)
