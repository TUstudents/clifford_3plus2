"""Tests for ``yukawa_overlaps.py`` — BT-1 kill test."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.broken_triality.yukawa_overlaps import (
    all_off_diagonals_nonzero,
    bt1_audit_payload,
    default_yukawa,
    eigenvalues_non_degenerate,
    project_onto_sm_cartan,
    projected_triality_orbit,
    sm_cartan_span_matrix,
    triality_orbit,
    yukawa_eigenvalues,
    yukawa_nonzero_eigenvalues,
    yukawa_off_diagonal_entries,
    yukawa_rank,
)
from clifford_3plus2_d5.broken_triality.reuse import (
    restricted_hypercharge_cartan_coords,
)


def test_sm_cartan_span_has_rank_three() -> None:
    span = sm_cartan_span_matrix()
    assert span.shape == (4, 3)
    assert span.rank() == 3


def test_projection_preserves_sm_cartan_vectors() -> None:
    y_prime = restricted_hypercharge_cartan_coords()
    projected = project_onto_sm_cartan(y_prime)
    assert (projected - y_prime).applyfunc(sp.simplify) == sp.zeros(4, 1)


def test_triality_orbit_has_period_three() -> None:
    y_prime = restricted_hypercharge_cartan_coords()
    v_0, v_1, v_2 = triality_orbit(y_prime)
    from clifford_3plus2_d5.broken_triality.reuse import apply_triality_to_cartan_vector

    next_after_v_2 = apply_triality_to_cartan_vector(v_2)
    assert (next_after_v_2 - v_0).applyfunc(sp.simplify) == sp.zeros(4, 1)


def test_projected_orbit_lives_in_sm_cartan_span() -> None:
    y_prime = restricted_hypercharge_cartan_coords()
    span = sm_cartan_span_matrix()
    span_rank = span.rank()
    for u in projected_triality_orbit(y_prime):
        augmented = span.row_join(u)
        assert augmented.rank() == span_rank


def test_yukawa_is_three_by_three_real_symmetric() -> None:
    y = default_yukawa()
    assert y.shape == (3, 3)
    assert (y - y.T).applyfunc(sp.simplify) == sp.zeros(3, 3)


def test_yukawa_diagonal_entries() -> None:
    y = default_yukawa()
    assert y[0, 0] == sp.Rational(7, 12)
    assert y[1, 1] == sp.Rational(169, 336)
    assert y[2, 2] == sp.Rational(59, 1008)


def test_yukawa_off_diagonal_entries_all_nonzero() -> None:
    assert all_off_diagonals_nonzero()
    entries = yukawa_off_diagonal_entries()
    assert all(entry != 0 for entry in entries)


def test_yukawa_eigenvalues_are_simple_and_distinct() -> None:
    spectrum = yukawa_eigenvalues()
    assert len(spectrum) == 3
    assert eigenvalues_non_degenerate()
    assert all(multiplicity == 1 for multiplicity in spectrum.values())


def test_yukawa_eigenvalues_match_expected_rationals() -> None:
    spectrum = yukawa_eigenvalues()
    expected = {sp.Rational(5, 7), sp.Rational(31, 72), sp.Integer(0)}
    assert set(spectrum.keys()) == expected


def test_yukawa_rank_is_two_under_default_v_star() -> None:
    # Documented rank deficit from the H_1 <-> H_2 symmetry of Y'.
    assert yukawa_rank() == 2


def test_yukawa_has_two_nonzero_eigenvalues_in_ratio_under_two() -> None:
    nonzero = yukawa_nonzero_eigenvalues()
    assert len(nonzero) == 2
    ratio = nonzero[0] / nonzero[1]
    # 5/7 over 31/72 = 360/217 approximately 1.66 — a hint that BT-2 will fail
    assert float(ratio) < 2


def test_bt1_audit_payload_reports_pass_with_rank_deficit() -> None:
    payload = bt1_audit_payload()
    assert payload.passes
    assert "PASS" in payload.verdict
    assert payload.yukawa_rank == 2
    assert payload.distinct_eigenvalue_count == 3
    assert payload.off_diagonal_entries_all_nonzero
    assert len(payload.nonzero_eigenvalues) == 2
    assert "rank deficit" in payload.interpretation
