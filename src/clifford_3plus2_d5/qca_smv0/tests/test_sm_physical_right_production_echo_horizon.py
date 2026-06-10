"""Tests for finite-horizon production echo spectrum."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_horizon import (
    sm_physical_right_production_echo_horizon_diagnostics,
)


def test_physical_right_production_echo_horizon_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_echo_horizon_diagnostics()

    assert diagnostics.short_steps >= 1
    assert diagnostics.long_steps >= 2
    assert diagnostics.perturbation_size >= 9e-5
    assert diagnostics.short_base_roundtrip_residual < 1e-5
    assert diagnostics.long_base_roundtrip_residual < 1e-5
    assert 0.95 <= diagnostics.short_min_gain < 1.05
    assert 0.95 <= diagnostics.short_max_gain < 1.05
    assert 0.95 <= diagnostics.long_min_gain < 1.05
    assert 0.95 <= diagnostics.long_max_gain < 1.05
    assert 0.90 <= diagnostics.min_gain_growth_ratio < 1.10
    assert 0.90 <= diagnostics.max_gain_growth_ratio < 1.10
    assert diagnostics.max_abs_log_gain_growth_per_tick < 0.05
    assert diagnostics.condition_number_delta < 0.10
    assert diagnostics.offdiag_correlation_delta < 0.05
    assert diagnostics.max_link_unitarity_residual < 1e-5


def test_physical_right_production_echo_horizon_validates_inputs() -> None:
    with pytest.raises(ValueError, match="short_steps must be positive"):
        sm_physical_right_production_echo_horizon_diagnostics(short_steps=0)

    with pytest.raises(ValueError, match="long_steps must exceed short_steps"):
        sm_physical_right_production_echo_horizon_diagnostics(short_steps=2, long_steps=2)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        sm_physical_right_production_echo_horizon_diagnostics(epsilon=0.0)
