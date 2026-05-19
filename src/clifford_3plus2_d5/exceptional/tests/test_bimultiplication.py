"""Tests for ``bimultiplication.py`` — Phase 0a triage."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.exceptional.bimultiplication import (
    all_generators_skew_symmetric,
    bimultiplication_audit_payload,
    bimultiplication_dimension,
    bimultiplication_generators,
    left_multiplications,
    right_multiplications,
)


def test_left_multiplications_count_and_shape() -> None:
    L = left_multiplications()
    assert len(L) == 7
    for matrix in L:
        assert matrix.shape == (8, 8)


def test_right_multiplications_count_and_shape() -> None:
    R = right_multiplications()
    assert len(R) == 7
    for matrix in R:
        assert matrix.shape == (8, 8)


def test_all_generators_skew_symmetric() -> None:
    assert all_generators_skew_symmetric()


def test_bimultiplication_dimension_is_so8() -> None:
    assert bimultiplication_dimension() == 28


def test_bimultiplication_generators_count() -> None:
    # 14 base + 21 LL commutators + 49 LR commutators + 21 RR commutators
    matrices = bimultiplication_generators()
    assert len(matrices) == 14 + 21 + 49 + 21


def test_audit_payload_confirms_kill() -> None:
    payload = bimultiplication_audit_payload()
    assert payload.span_dimension == 28
    assert payload.matches_so8
    assert payload.all_skew_symmetric
    assert "BIMULT KILL" in payload.verdict
    assert "triality" in payload.interpretation
