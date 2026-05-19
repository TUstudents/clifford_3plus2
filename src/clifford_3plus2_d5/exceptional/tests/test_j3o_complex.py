"""Tests for ``j3o_complex.py`` — Phase 2b kill test."""

from __future__ import annotations

from clifford_3plus2_d5.exceptional.j3o_complex import (
    chiral16_copies_count,
    decomposition_total_dimension,
    j3o_complex_decomposition_audit_payload,
    j3o_complex_real_dimension,
    standard_decomposition_realified,
    three_chiral16_fits_inside_j3o_complex,
    three_chiral16_required_dimension,
)


def test_j3o_complex_dimension_is_fifty_four() -> None:
    assert j3o_complex_real_dimension() == 54


def test_decomposition_sums_to_fifty_four() -> None:
    assert decomposition_total_dimension() == 54


def test_decomposition_contains_two_chiral_sixteen_subreps() -> None:
    decomposition = standard_decomposition_realified()
    sixteens = [key for key, value in decomposition.items() if value == 16]
    assert len(sixteens) == 2  # 16 + 16* (particle + antiparticle)


def test_chiral16_copies_count() -> None:
    assert chiral16_copies_count() == 2


def test_three_chiral16_dimensional_fit_is_marginal_true() -> None:
    # Dimensionally 3 × 16 = 48 ≤ 54; but rep theory still forbids 3 copies.
    assert three_chiral16_required_dimension() == 48
    assert three_chiral16_fits_inside_j3o_complex()


def test_phase2b_audit_payload_passes_negative() -> None:
    payload = j3o_complex_decomposition_audit_payload()
    assert payload.j3o_complex_dimension == 54
    assert payload.decomposition_sum == 54
    assert payload.chiral16_copies == 2
    assert payload.three_chiral16_dimensional_fit  # 48 ≤ 54
    assert not payload.three_chiral16_representation_fit  # but only 2 copies
    assert "J_3^C(O) KILL" in payload.verdict
    assert "particle" in payload.interpretation
    assert "antiparticle" in payload.interpretation
