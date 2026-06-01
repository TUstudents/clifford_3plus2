"""Aggregate payload tests for V12 successor certificate."""

from __future__ import annotations

from clifford_3plus2_d5.depth_scar.successor_certificate import (
    successor_certificate_payload,
)


def test_successor_certificate_payload_passes() -> None:
    payload = successor_certificate_payload()

    assert payload.final_verdict == "V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS"
    assert payload.basis_size == 12
    assert payload.row_count == 24
    assert payload.allowed_successors == {"a": ("u",), "b": ("a",)}
    assert payload.certificate_complete
    assert payload.forbidden_rows_have_vetoes
    assert payload.shortcut_repair_control_rejected
    assert payload.external_leakage_controls_rejected
    assert payload.certificate_implies_v11_no_leakage
    assert payload.microscopic_basis_completeness_derived is False
    assert len(payload.certificate) == payload.row_count
