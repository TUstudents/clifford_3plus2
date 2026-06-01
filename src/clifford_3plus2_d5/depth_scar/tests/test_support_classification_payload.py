"""Aggregate payload tests for V7 minimal support classification."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.support_classification import (
    minimal_nilpotent_support_payload,
)


def test_minimal_nilpotent_support_payload_passes() -> None:
    payload = minimal_nilpotent_support_payload()

    assert payload.final_verdict == "MINIMAL_NILPOTENT_SUPPORT_CLASSIFICATION_PASS"
    assert payload.total_support_count == 64
    assert payload.accepted_minimal_count == 6
    assert payload.accepted_minimal_orbit_count == 1
    assert payload.all_minimal_equivalent_to_flag
    assert payload.minimal_support_spectra_pass
    assert payload.rank_one_controls_exist
    assert payload.cycle_controls_rejected
    assert payload.nonminimal_shortcut_count == 6
    assert payload.minimality_load_bearing
    assert payload.microscopic_minimality_derived is False
