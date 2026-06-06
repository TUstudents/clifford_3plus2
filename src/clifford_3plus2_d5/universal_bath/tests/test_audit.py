"""Tests for the Session 01 aggregate audit."""

from clifford_3plus2_d5.universal_bath.audit import universal_bath_audit_payload


def test_universal_bath_audit_reports_spine_pass() -> None:
    payload = universal_bath_audit_payload()
    assert payload.final_verdict == "UNIVERSAL_BATH_SPINE_PASS"
    assert payload.tail_fixed_point
    assert payload.silver_value
    assert payload.toy_measure_positive
    assert payload.moment_round_trip
    assert payload.response_round_trip
    assert payload.finite_head_schur_match
    assert payload.alternate_tail_changes_response
    assert payload.reduction_taxonomy_pass
    assert payload.opuc_free_tail_pass
