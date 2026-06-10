"""Tests for dense production workload scaling."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_workload import (
    physical_right_production_sm_link_bytes_per_site,
    physical_right_production_state_bytes_per_site,
    sm_physical_right_production_workload_diagnostics,
)


def test_physical_right_production_workload_diagnostics_pass_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_workload_diagnostics()

    assert diagnostics.site_count == 27
    assert diagnostics.state_bytes_per_site == 69_280
    assert diagnostics.state_mib < 4.0
    assert diagnostics.sm_link_bytes_per_site == 65_536
    assert diagnostics.sm_link_state_fraction > 0.90
    assert diagnostics.wilson_force_coordinate_count == 2_592
    assert diagnostics.finite_difference_action_evaluations == 5_184
    assert diagnostics.finite_difference_plaquette_holonomies == 839_808
    assert diagnostics.single_site_plaquette_holonomies == 1_152
    assert diagnostics.quadratic_work_ratio_to_single_site == 729.0
    assert diagnostics.local_force_target_plaquette_holonomies == 162
    assert diagnostics.finite_difference_to_local_work_ratio == 5_184.0


def test_physical_right_production_workload_scales_quadratically_with_sites() -> None:
    one_site = sm_physical_right_production_workload_diagnostics((1, 1, 1))
    eight_sites = sm_physical_right_production_workload_diagnostics((2, 2, 2))

    assert eight_sites.state_mib / one_site.state_mib == 8.0
    assert eight_sites.finite_difference_plaquette_holonomies / one_site.finite_difference_plaquette_holonomies == 64.0
    assert eight_sites.finite_difference_to_local_work_ratio / one_site.finite_difference_to_local_work_ratio == 8.0


def test_physical_right_production_workload_validates_inputs() -> None:
    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_workload_diagnostics((3, 3, 0))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_workload_diagnostics((3, 3))  # type: ignore[arg-type]


def test_physical_right_production_workload_byte_helpers() -> None:
    assert physical_right_production_state_bytes_per_site() == 69_280
    assert physical_right_production_sm_link_bytes_per_site() == 65_536
