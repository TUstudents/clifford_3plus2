"""Aggregate payload tests for V11 selection no-leakage."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.selection_no_leakage import (
    selection_no_leakage_payload,
)


def test_selection_no_leakage_payload_passes() -> None:
    payload = selection_no_leakage_payload()

    assert payload.final_verdict == "V11_SELECTION_SIGNATURE_NO_LEAKAGE_PASS"
    assert payload.unique_successor_sets == {"a": ("u",), "b": ("a",)}
    assert payload.unique_successors_force_no_leakage
    assert payload.unique_successor_block_is_isometry
    assert payload.unique_successor_block_matches_path_support
    assert payload.repair_isometry_saturation_available
    assert payload.leaky_successor_control_rejected
    assert payload.small_leakage_bounds_pass
    assert payload.microscopic_successor_enumeration_done is False
