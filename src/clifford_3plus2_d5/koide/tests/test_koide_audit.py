"""Tests for ``koide_audit.py`` (aggregates KO-1..KO-4)."""

from __future__ import annotations

from clifford_3plus2_d5.koide.koide_audit import koide_audit_payload


def test_audit_sub_payloads_consistent() -> None:
    p = koide_audit_payload()
    assert p.geometry.koide_holds_empirically
    assert p.bcc_z3_on_flavor.z3_fixes_diagonal
    assert p.yukawa_locus.z3_yukawa_is_circulant
    assert p.cone_locus_compat.locus_intersection_non_empty


def test_audit_final_verdict_is_consistent() -> None:
    p = koide_audit_payload()
    assert p.final_verdict == "KOIDE CONSISTENT"


def test_audit_pdg_not_in_locus() -> None:
    p = koide_audit_payload()
    assert not p.pdg_in_locus


def test_audit_verdict_string_contains_class() -> None:
    p = koide_audit_payload()
    assert "KOIDE CONSISTENT" in p.verdict
    assert "PDG NOT IN LOCUS" in p.verdict


def test_audit_interpretation_mentions_key_findings() -> None:
    p = koide_audit_payload()
    text = p.interpretation
    assert "K_PDG" in text
    assert "3 + 2√2" in text
    assert "Bold-B" in text or "dynamical Higgs" in text
    assert "TWO of three" in text or "degenerate" in text.lower()
