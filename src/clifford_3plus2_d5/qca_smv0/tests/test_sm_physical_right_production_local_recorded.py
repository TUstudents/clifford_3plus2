"""Tests for the Stage 39 local-force recorded rollout."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_recorded import (
    sm_physical_right_production_local_recorded_diagnostics,
)


def test_physical_right_production_local_recorded_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_local_recorded_diagnostics()

    assert diagnostics.site_count == 4
    assert diagnostics.step_count == 2
    assert diagnostics.record_count == 3
    assert diagnostics.loop_scan_final_residual < 1e-6
    assert diagnostics.loop_scan_observation_residual < 1e-6
    assert bool(diagnostics.loop_all_finite)
    assert bool(diagnostics.scan_all_finite)
    assert diagnostics.family_norm_drift < 3e-6
    assert diagnostics.max_family_norm_drift < 3e-6
    assert diagnostics.max_sm_link_unitarity_residual < 2e-6
    assert diagnostics.max_higgs_link_unitarity_residual < 2e-6
    assert diagnostics.higgs_field_total_delta_norm > 1e-4
    assert diagnostics.sm_link_total_delta_norm > 1e-4
    assert diagnostics.sm_momentum_total_delta_norm > 1e-5


def test_physical_right_production_local_recorded_validates_inputs() -> None:
    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_recorded_diagnostics((2, 0, 1))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_recorded_diagnostics((2, 1))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="steps must be positive"):
        sm_physical_right_production_local_recorded_diagnostics(steps=0)

