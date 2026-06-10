"""Tests for the physical-right production refinement-limitation audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_refinement import (
    sm_physical_right_production_refinement_diagnostics,
    sm_physical_right_production_refinement_pair,
)


def test_physical_right_production_refinement_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="base_steps"):
        sm_physical_right_production_refinement_pair(base_steps=0)
    with pytest.raises(ValueError, match="base_step_size"):
        sm_physical_right_production_refinement_pair(base_step_size=0.0)


def test_physical_right_production_refinement_diagnostics_record_limitation() -> None:
    diagnostics = sm_physical_right_production_refinement_diagnostics()

    assert diagnostics.total_time_match_residual < 1e-12
    assert diagnostics.base_energy_drift_abs < 1e-5
    assert diagnostics.refined_energy_drift_abs < 5e-4
    assert diagnostics.drift_refinement_ratio > 5.0
    assert bool(diagnostics.refinement_limitation_detected)
    assert bool(diagnostics.refined_energy_drift_controlled)
    assert diagnostics.base_family_norm_drift < 1e-5
    assert diagnostics.refined_family_norm_drift < 1e-4
    assert diagnostics.refined_sm_link_unitarity_residual < 1e-5
    assert diagnostics.refined_higgs_link_unitarity_residual < 1e-5
    assert diagnostics.vacuum_refined_energy_drift_abs < 1e-10
    assert bool(diagnostics.histories_all_finite)
