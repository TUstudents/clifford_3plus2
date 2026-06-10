"""Tests for the Stage 45 all-sector production cone audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_sector_cones import (
    sm_physical_right_production_sector_cones_diagnostics,
)


def test_physical_right_production_sector_cones_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_sector_cones_diagnostics()

    assert diagnostics.site_count == 125
    assert diagnostics.horizon_steps == 2
    assert diagnostics.sector_count == 6
    assert diagnostics.max_support_radius <= diagnostics.horizon_steps
    assert diagnostics.radius_overshoot < 1e-12
    assert diagnostics.family_support_count >= 8
    assert diagnostics.higgs_support_count >= 8
    assert diagnostics.higgs_momentum_support_count >= 1
    assert diagnostics.sm_link_support_count >= 1
    assert diagnostics.sm_momentum_support_count >= 1
    assert diagnostics.higgs_link_support_count >= 1
    assert diagnostics.min_support_count >= 1
    assert diagnostics.outside_step_cone_site_count == 0
    assert diagnostics.outside_step_cone_max_norm < 1e-12
    assert diagnostics.max_response_norm > 1e-3
    assert diagnostics.min_detected_response_norm > 1e-8


def test_physical_right_production_sector_cones_validates_inputs() -> None:
    with pytest.raises(ValueError, match="horizon_steps must be exactly 2"):
        sm_physical_right_production_sector_cones_diagnostics(horizon_steps=1)

    with pytest.raises(ValueError, match="horizon_steps must be exactly 2"):
        sm_physical_right_production_sector_cones_diagnostics(horizon_steps=3)

    with pytest.raises(ValueError, match="at least the two-tick cone diameter"):
        sm_physical_right_production_sector_cones_diagnostics((4, 5, 5))

    with pytest.raises(ValueError, match="at least the two-tick cone diameter"):
        sm_physical_right_production_sector_cones_diagnostics((5, 5))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="perturbation_size must be positive"):
        sm_physical_right_production_sector_cones_diagnostics(perturbation_size=0.0)

    with pytest.raises(ValueError, match="support_threshold must be positive"):
        sm_physical_right_production_sector_cones_diagnostics(support_threshold=0.0)

    with pytest.raises(ValueError, match="step_size must be positive"):
        sm_physical_right_production_sector_cones_diagnostics(step_size=0.0)
