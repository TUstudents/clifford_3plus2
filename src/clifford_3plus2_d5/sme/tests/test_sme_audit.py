"""Tests for ``sme_audit.py``."""

from __future__ import annotations

from clifford_3plus2_d5.sme.sme_audit import sme_audit_payload


def test_audit_all_phases_consistent() -> None:
    payload = sme_audit_payload()
    assert payload.all_phases_consistent


def test_audit_links_to_subphase_payloads() -> None:
    payload = sme_audit_payload()
    assert payload.framework.all_classes_consistent
    assert payload.mapping.nonzero_component_count == 3
    assert payload.epsilon_constraint.scale_verdict in {
        "SUB-PLANCK KILL",
        "PLANCK-CONSISTENT",
        "UNFALSIFIABLE PASS",
        "OBSERVABLE POSITIVE",
    }


def test_audit_final_scale_verdict_matches_constraint() -> None:
    payload = sme_audit_payload()
    assert payload.final_scale_verdict == payload.epsilon_constraint.scale_verdict


def test_audit_with_current_bound_is_unfalsifiable_pass() -> None:
    # With the current representative bound (10⁻¹⁷ GeV⁻¹), the verdict
    # should be UNFALSIFIABLE PASS.  This test pins the present-state
    # result; updates to the bound constant will change the expected
    # verdict and require updating this test alongside.
    payload = sme_audit_payload()
    assert payload.final_scale_verdict == "UNFALSIFIABLE PASS"


def test_audit_verdict_string_contains_scale_class() -> None:
    payload = sme_audit_payload()
    assert "UNFALSIFIABLE PASS" in payload.verdict


def test_audit_interpretation_mentions_dim5_target() -> None:
    payload = sme_audit_payload()
    assert "d^{(5)}" in payload.interpretation


def test_audit_reports_provisional_with_unverified_caveats() -> None:
    # While F-sme-5 is unchecked, KM normalization is not derived, and KR
    # entry-ids are unverified, the audit must report itself as provisional
    # and surface the caveats in the verdict string and caveat tuple.
    payload = sme_audit_payload()
    assert payload.is_provisional is True
    assert len(payload.provisional_caveats) >= 1
    assert "PROVISIONAL" in payload.verdict
    assert "Provisional caveats" in payload.interpretation


def test_audit_provisional_caveats_cover_all_three_followups() -> None:
    payload = sme_audit_payload()
    joined = " | ".join(payload.provisional_caveats)
    assert "F-sme-5" in joined
    assert "Kostelecky-Mewes" in joined
    assert "Kostelecky-Russell" in joined
