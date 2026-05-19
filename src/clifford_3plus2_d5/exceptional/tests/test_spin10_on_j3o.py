"""Tests for ``spin10_on_j3o.py`` — Phase 2 kill test."""

from __future__ import annotations

from clifford_3plus2_d5.exceptional.spin10_on_j3o import (
    chiral16_real_dimension,
    decomposition_for_preferred_row,
    pairwise_spinor_overlap,
    spin10_decomposition_audit_payload,
    three_chiral16_fits_inside_j3o,
    three_chiral16_required_dimension,
    total_off_diagonal_dimension,
    union_of_all_spinors,
)


def test_per_row_decomposition_dimensions() -> None:
    for row in (0, 1, 2):
        singlet, vector, spinor = decomposition_for_preferred_row(row)
        assert len(singlet) == 1
        assert len(vector) == 10
        assert len(spinor) == 16
        assert len(singlet) + len(vector) + len(spinor) == 27


def test_per_row_decomposition_indices_are_disjoint() -> None:
    for row in (0, 1, 2):
        singlet, vector, spinor = decomposition_for_preferred_row(row)
        all_indices = set(singlet) | set(vector) | set(spinor)
        assert len(all_indices) == 27


def test_pairwise_spinor_overlap_is_eight() -> None:
    # Each pair of preferred-row spinors shares exactly one off-diagonal
    # octonion (8 components).
    assert pairwise_spinor_overlap(0, 1) == 8
    assert pairwise_spinor_overlap(0, 2) == 8
    assert pairwise_spinor_overlap(1, 2) == 8


def test_union_of_three_spinors_covers_all_off_diagonal() -> None:
    # All 24 off-diagonal octonion components are touched by at least one
    # of the three preferred-row spinors.
    assert union_of_all_spinors() == 24


def test_dimensional_obstruction() -> None:
    assert total_off_diagonal_dimension() == 24
    assert chiral16_real_dimension() == 16
    assert three_chiral16_required_dimension() == 48
    assert not three_chiral16_fits_inside_j3o()


def test_kill_test_audit_payload_passes_negative() -> None:
    payload = spin10_decomposition_audit_payload()
    assert payload.j3o_total_dimension == 27
    assert payload.decomposition_matches
    assert payload.singlet_dimension == 1
    assert payload.vector_dimension == 10
    assert payload.spinor_dimension == 16
    assert not payload.three_chiral16_fits
    assert payload.pairwise_spinor_overlaps == (8, 8, 8)
    assert "J_3(O) KILL" in payload.verdict
    assert "27 = 16 + 10 + 1" in payload.verdict
    assert "CANNOT arise" in payload.interpretation


def test_audit_interpretation_cites_e6_result() -> None:
    payload = spin10_decomposition_audit_payload()
    assert "E_6" in payload.interpretation
