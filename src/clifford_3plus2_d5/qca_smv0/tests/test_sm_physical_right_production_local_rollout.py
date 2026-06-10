"""Tests for the Stage 38 local-force production rollout."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_rollout import (
    sm_physical_right_production_local_rollout_diagnostics,
)


def test_physical_right_production_local_rollout_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_local_rollout_diagnostics()

    assert diagnostics.site_count == 4
    assert diagnostics.wilson_epsilon_invariance_residual < 1e-12
    assert bool(diagnostics.final_all_finite)
    assert diagnostics.family_norm_drift < 2e-6
    assert diagnostics.sm_link_unitarity_residual < 2e-6
    assert diagnostics.higgs_link_unitarity_residual < 2e-6
    assert diagnostics.higgs_delta_norm > 1e-5
    assert diagnostics.sm_link_delta_norm > 1e-5
    assert diagnostics.sm_momentum_delta_norm > 1e-6
    assert diagnostics.local_force_plaquette_holonomies == 24
    assert diagnostics.legacy_force_plaquette_holonomies == 18_432
    assert diagnostics.finite_difference_to_local_work_ratio == 768.0


def test_physical_right_production_local_rollout_validates_shape() -> None:
    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_rollout_diagnostics((2, 0, 1))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_rollout_diagnostics((2, 1))  # type: ignore[arg-type]

