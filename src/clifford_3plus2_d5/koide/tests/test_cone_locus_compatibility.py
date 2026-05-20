"""Tests for ``cone_locus_compatibility.py``."""

from __future__ import annotations

from clifford_3plus2_d5.koide.cone_locus_compatibility import (
    classify_verdict,
    cone_locus_compatibility_payload,
    locus_intersection_is_non_empty,
    locus_is_strictly_inside_cone,
    pdg_in_locus,
)


def test_locus_intersection_is_non_empty() -> None:
    # At |v_t|/|v_o| = 3 + 2√2, the eigenvalue triple is on the Koide cone.
    assert locus_intersection_is_non_empty()


def test_locus_is_not_strictly_inside_cone() -> None:
    # Counterexample: |v_t| = |v_o| = 1 gives K ≠ 2/3.
    assert not locus_is_strictly_inside_cone()


def test_pdg_not_in_locus() -> None:
    # PDG has three distinct masses; L_Z3 has 2-fold degeneracy.
    assert not pdg_in_locus()


def test_classify_verdict_is_consistent() -> None:
    assert classify_verdict() == "KOIDE CONSISTENT"


def test_payload_has_consistent_verdict() -> None:
    p = cone_locus_compatibility_payload()
    assert p.locus_intersection_non_empty
    assert not p.locus_strictly_inside_cone
    assert not p.pdg_in_locus
    assert p.final_verdict == "KOIDE CONSISTENT"
    assert "PDG NOT IN LOCUS" in p.additional_tags


def test_payload_interpretation_mentions_3_plus_2_root_2() -> None:
    p = cone_locus_compatibility_payload()
    assert "3 + 2√2" in p.interpretation
    assert "CONSISTENT" in p.interpretation


def test_payload_interpretation_mentions_bold_b_followup() -> None:
    p = cone_locus_compatibility_payload()
    assert "Bold-B" in p.interpretation or "dynamical Higgs" in p.interpretation
