"""Tests for the aggregate depth-hop-Walsh probe."""

from __future__ import annotations

from clifford_3plus2_d5.depth_hop_walsh.depth_hop_walsh_audit import (
    REMAINING_DECLARED_INPUTS,
    depth_hop_walsh_audit_payload,
)


def test_final_verdict_is_w2_combined() -> None:
    payload = depth_hop_walsh_audit_payload()
    assert payload.final_verdict == payload.support.final_verdict
    assert payload.final_verdict == "DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT"


def test_does_not_establish_claim_a() -> None:
    payload = depth_hop_walsh_audit_payload()
    assert payload.establishes_claim_a is False


def test_claim_a_killed_under_both_lenses() -> None:
    # Coefficient-Walsh (W2) and covariant (W4) both kill -> escape hatch closed.
    payload = depth_hop_walsh_audit_payload()
    assert payload.claim_a_killed_both_lenses is True
    assert payload.covariant.escape_hatch_closed is True
    assert payload.covariant.final_verdict == "COVARIANT_KILL_FORBIDDEN_QUADRUPOLE"


def test_diagnostic_cannot_alter_verdict() -> None:
    payload = depth_hop_walsh_audit_payload()
    # W3 is carried but diagnostic-only; the verdict equals W2's regardless.
    assert payload.diagnostic.diagnostic_only is True
    assert payload.final_verdict == payload.support.final_verdict


def test_schur_obstruction_is_recorded() -> None:
    payload = depth_hop_walsh_audit_payload()
    assert payload.depth_hierarchy_requires_s3_breaking is True
    assert payload.schur_obstruction.final_verdict == "DEPTH_HIERARCHY_REQUIRES_S3_BREAKING"


def test_remaining_inputs_chain() -> None:
    payload = depth_hop_walsh_audit_payload()
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
    assert "generation_depth_embedding_derived" in payload.remaining_declared_inputs
    assert "radial_depth_equals_twice_walsh_degree" in payload.remaining_declared_inputs
