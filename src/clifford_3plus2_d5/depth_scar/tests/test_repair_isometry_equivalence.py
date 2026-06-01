"""V10 tests for no-leakage and unit-weight equivalence."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.repair_isometry import (
    no_leakage_forces_unit_weights,
    repair_isometry_saturation_pass,
    unit_weights_force_no_leakage,
)


def test_no_leakage_forces_unit_active_repair_weights() -> None:
    assert no_leakage_forces_unit_weights()


def test_unit_active_repair_weights_force_no_leakage() -> None:
    assert unit_weights_force_no_leakage()


def test_repair_isometry_saturation_gate_passes() -> None:
    assert repair_isometry_saturation_pass()
