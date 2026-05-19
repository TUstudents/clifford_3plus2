"""Tests for ``fano_lines.py`` — Phase 0b triage."""

from __future__ import annotations

from clifford_3plus2_d5.exceptional.fano_lines import (
    candidate_closes_as_lie_algebra,
    candidate_su2_generators,
    fano_e7_lines,
    fano_lines_audit_payload,
    fano_lines_through,
    shared_generator_count,
    union_dimension,
)


def test_three_fano_lines_through_e7() -> None:
    lines = fano_e7_lines()
    assert len(lines) == 3
    for line in lines:
        assert 7 in line


def test_lines_through_e7_match_expected() -> None:
    lines = set(fano_e7_lines())
    expected = {(1, 6, 7), (2, 5, 7), (3, 4, 7)}
    assert lines == expected


def test_no_candidate_closes_as_lie_algebra() -> None:
    # Octonion non-associativity prevents closure.
    lines = fano_e7_lines()
    for line in lines:
        assert not candidate_closes_as_lie_algebra(candidate_su2_generators(line))


def test_shared_generator_is_e7() -> None:
    assert shared_generator_count() == 1


def test_union_span_is_seven_not_nine() -> None:
    # 3 SU(2)s through e_7 share e_7, so 9 - shared = 9 - 2 = 7
    # (the e_7 is counted three times across lines, so 9 - 2 = 7).
    assert union_dimension() == 7


def test_audit_payload_confirms_kill() -> None:
    payload = fano_lines_audit_payload()
    assert payload.line_count == 3
    assert payload.shared_generator_count == 1
    assert payload.union_span_dimension == 7
    assert not payload.any_candidate_is_lie_algebra
    assert "FANO KILL" in payload.verdict


def test_fano_lines_through_index_filter() -> None:
    # Sanity-check the helper.
    lines = fano_lines_through(1)
    assert len(lines) == 3
    for line in lines:
        assert 1 in line
