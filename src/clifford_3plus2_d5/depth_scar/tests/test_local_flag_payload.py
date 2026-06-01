"""Aggregate payload tests for V6 local flag unitarity."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import local_flag_unitarity_payload


def test_local_flag_unitarity_payload_passes() -> None:
    payload = local_flag_unitarity_payload()

    assert payload.final_verdict == "LOCAL_FLAG_PARTIAL_ISOMETRY_PASS"
    assert payload.projection_formula_pass
    assert payload.partial_isometry_magnitudes_pass
    assert payload.phase_gauge_pass
    assert payload.canonical_representative_pass
    assert payload.laplacian_and_transfer_pass
    assert payload.rank_one_control_rejected
    assert payload.contractive_control_rejected
    assert payload.unequal_control_rejected
    assert payload.cyclic_control_rejected
    assert payload.length_three_support_microscopically_derived is False
