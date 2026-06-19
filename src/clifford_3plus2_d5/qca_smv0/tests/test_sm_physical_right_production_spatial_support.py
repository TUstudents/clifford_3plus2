"""Tests for the Stage 43 one-tick production spatial-support audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_spatial_support import (
    sm_physical_right_production_spatial_support_diagnostics,
)


def test_physical_right_production_spatial_support_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_spatial_support_diagnostics()

    assert diagnostics.site_count == 343
    assert diagnostics.predicted_tick_radius == 2
    assert diagnostics.max_measured_support_radius <= diagnostics.predicted_tick_radius
    assert diagnostics.family_support_radius == 1
    assert diagnostics.higgs_support_radius == 1
    assert diagnostics.higgs_momentum_support_radius == 0
    assert diagnostics.sm_link_support_radius == 0
    assert diagnostics.sm_momentum_support_radius == 0
    assert diagnostics.higgs_link_support_radius == 0
    assert diagnostics.family_support_count >= 4
    assert diagnostics.higgs_support_count >= 1
    assert diagnostics.higgs_momentum_support_count == 1
    assert diagnostics.sm_link_support_count == 1
    assert diagnostics.sm_momentum_support_count == 1
    assert diagnostics.higgs_link_support_count == 1
    assert diagnostics.outside_predicted_radius_site_count == 0
    assert diagnostics.outside_predicted_radius_max_norm < 1e-12
    assert diagnostics.max_response_norm > 1e-3
    assert diagnostics.min_detected_response_norm > diagnostics.support_threshold


def test_physical_right_production_spatial_support_validates_inputs() -> None:
    with pytest.raises(ValueError, match="larger than the one-tick stencil diameter"):
        sm_physical_right_production_spatial_support_diagnostics((5, 5, 5))

    with pytest.raises(ValueError, match="larger than the one-tick stencil diameter"):
        sm_physical_right_production_spatial_support_diagnostics((7, 7))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="perturbation_size must be positive"):
        sm_physical_right_production_spatial_support_diagnostics(perturbation_size=0.0)

    with pytest.raises(ValueError, match="support_threshold must be positive"):
        sm_physical_right_production_spatial_support_diagnostics(support_threshold=0.0)

    with pytest.raises(ValueError, match="step_size must be positive"):
        sm_physical_right_production_spatial_support_diagnostics(step_size=0.0)
