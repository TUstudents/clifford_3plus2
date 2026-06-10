"""Tests for the Stage 40 local-force production profile."""

import pytest

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_local_profile import (
    sm_physical_right_production_local_profile_diagnostics,
)


def test_physical_right_production_local_profile_passes_stage_thresholds() -> None:
    diagnostics = sm_physical_right_production_local_profile_diagnostics()

    assert diagnostics.site_count == 2
    assert bool(diagnostics.eager_profile_all_finite)
    assert diagnostics.eager_profile_python_seconds >= 0.0
    assert diagnostics.eager_profile_payload_key_count >= 7
    assert diagnostics.jit_compile_seconds >= 0.0
    assert diagnostics.jit_run_seconds >= 0.0
    assert diagnostics.jit_eager_state_residual < 1e-6
    assert bool(diagnostics.final_all_finite)
    assert diagnostics.family_norm_drift < 2e-6
    assert diagnostics.sm_link_unitarity_residual < 2e-6
    assert diagnostics.higgs_link_unitarity_residual < 2e-6


def test_physical_right_production_local_profile_validates_shape() -> None:
    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_profile_diagnostics((2, 0, 1))

    with pytest.raises(ValueError, match="lattice_shape must contain three positive sizes"):
        sm_physical_right_production_local_profile_diagnostics((2, 1))  # type: ignore[arg-type]

