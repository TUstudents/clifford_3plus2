"""Tests for the Stage 42 local physical-right fermion current."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_current import (
    sm_physical_right_production_local_current_diagnostics,
)


def test_physical_right_production_local_current_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_local_current_diagnostics()

    assert diagnostics.oracle_site_count == 1
    assert diagnostics.production_site_count == 4
    assert diagnostics.finite_difference_current_norm > 1e-2
    assert diagnostics.local_current_norm > 1e-2
    assert diagnostics.finite_difference_local_max_residual < 1e-4
    assert diagnostics.finite_difference_local_norm_residual < 1e-4
    assert diagnostics.production_local_alias_residual < 1e-12
    assert diagnostics.zero_state_local_current_norm < 1e-7
    assert diagnostics.production_fermion_epsilon_residual < 1e-12
    assert diagnostics.production_family_norm_drift < 8e-5
    assert diagnostics.max_sm_link_unitarity_residual < 8e-7
    assert diagnostics.max_higgs_link_unitarity_residual < 8e-7
    assert diagnostics.legacy_global_streaming_term_count == 24576
    assert diagnostics.local_streaming_term_count == 384
    assert diagnostics.estimated_work_reduction_ratio == 64.0


def test_physical_right_production_local_current_validates_inputs() -> None:
    with pytest.raises(ValueError, match="three positive sizes"):
        sm_physical_right_production_local_current_diagnostics(
            oracle_lattice_shape=(0, 1, 1),
        )

    with pytest.raises(ValueError, match="three positive sizes"):
        sm_physical_right_production_local_current_diagnostics(
            production_lattice_shape=(2, 2),  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="step_size must be positive"):
        sm_physical_right_production_local_current_diagnostics(step_size=0.0)
