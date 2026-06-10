"""Tests for the Stage 44 production family-cone audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_cone import (
    sm_physical_right_production_cone_diagnostics,
)


def test_physical_right_production_cone_passes_two_step_thresholds() -> None:
    diagnostics = sm_physical_right_production_cone_diagnostics((5, 5, 5), max_steps=2)

    assert diagnostics.site_count == 125
    assert diagnostics.max_steps == 2
    assert diagnostics.step_one_support_radius == 1
    assert diagnostics.step_two_support_radius == 2
    assert diagnostics.max_support_radius == 2
    assert diagnostics.step_one_support_count == 8
    assert diagnostics.step_two_support_count >= 20
    assert diagnostics.outside_step_cone_site_count == 0
    assert diagnostics.outside_step_cone_max_norm < 1e-12
    assert diagnostics.radius_growth_residual < 1e-12
    assert diagnostics.support_count_growth_min_delta >= 8
    assert diagnostics.max_response_norm > 1e-3
    assert diagnostics.min_detected_response_norm > 1e-8


def test_physical_right_production_cone_validates_inputs() -> None:
    with pytest.raises(ValueError, match="max_steps must be 1, 2, or 3"):
        sm_physical_right_production_cone_diagnostics(max_steps=0)

    with pytest.raises(ValueError, match="max_steps must be 1, 2, or 3"):
        sm_physical_right_production_cone_diagnostics(max_steps=4)

    with pytest.raises(ValueError, match="at least the family-cone diameter"):
        sm_physical_right_production_cone_diagnostics((5, 5, 5), max_steps=3)

    with pytest.raises(ValueError, match="at least the family-cone diameter"):
        sm_physical_right_production_cone_diagnostics((7, 7), max_steps=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="perturbation_size must be positive"):
        sm_physical_right_production_cone_diagnostics(perturbation_size=0.0)

    with pytest.raises(ValueError, match="support_threshold must be positive"):
        sm_physical_right_production_cone_diagnostics(support_threshold=0.0)

    with pytest.raises(ValueError, match="step_size must be positive"):
        sm_physical_right_production_cone_diagnostics(step_size=0.0)
