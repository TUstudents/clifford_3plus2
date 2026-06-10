"""Tests for the Stage 41 local-force spatial-support audit."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_force_support import (
    sm_physical_right_production_force_support_diagnostics,
)


def test_physical_right_production_force_support_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_force_support_diagnostics()

    assert diagnostics.site_count == 343
    assert diagnostics.predicted_force_radius == 2
    assert diagnostics.measured_support_radius == diagnostics.predicted_force_radius
    assert diagnostics.support_site_count == 11
    assert diagnostics.radius_zero_support_count == 1
    assert diagnostics.radius_one_support_count == 7
    assert diagnostics.radius_two_support_count == 3
    assert diagnostics.max_force_site_norm > 1e-5
    assert diagnostics.min_support_site_norm > 1e-6
    assert diagnostics.outside_predicted_radius_max_norm < 1e-12
    assert diagnostics.outside_predicted_radius_site_count == 0
    assert bool(diagnostics.center_site_supported)


def test_physical_right_production_force_support_validates_inputs() -> None:
    with pytest.raises(ValueError, match="larger than the two-hop diameter"):
        sm_physical_right_production_force_support_diagnostics((5, 5, 5))

    with pytest.raises(ValueError, match="larger than the two-hop diameter"):
        sm_physical_right_production_force_support_diagnostics((7, 7))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="perturbation_size must be positive"):
        sm_physical_right_production_force_support_diagnostics(perturbation_size=0.0)

    with pytest.raises(ValueError, match="support_threshold must be positive"):
        sm_physical_right_production_force_support_diagnostics(support_threshold=0.0)

