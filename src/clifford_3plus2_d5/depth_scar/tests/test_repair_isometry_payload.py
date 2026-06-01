"""Aggregate payload tests for V10 repair isometry."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.repair_isometry import repair_isometry_payload


def test_repair_isometry_payload_passes() -> None:
    payload = repair_isometry_payload()

    assert payload.final_verdict == "V10_REPAIR_ISOMETRY_SATURATION_PASS"
    assert payload.unitarity_balance_formula_pass
    assert payload.no_leakage_forces_unit_weights
    assert payload.unit_weights_force_no_leakage
    assert payload.tree_phases_removable
    assert payload.target_spectrum_forces_unit_weights
    assert payload.symmetric_leakage_preserves_ratio_but_rescales
    assert payload.unequal_leakage_breaks_ratio
    assert payload.no_leakage_microscopically_derived is False
