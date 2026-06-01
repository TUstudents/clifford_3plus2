"""Tests for finite support enumeration in V7."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.support_classification import (
    OFF_DIAGONAL_POSITIONS,
    all_no_self_loop_binary_supports,
    support_edge_count,
    support_matrix_from_bits,
)


def test_no_self_loop_binary_support_census_has_64_elements() -> None:
    supports = all_no_self_loop_binary_supports()

    assert len(OFF_DIAGONAL_POSITIONS) == 6
    assert len(supports) == 64
    assert all(support[row, row] == 0 for support in supports for row in range(3))
    assert all(entry in (0, 1) for support in supports for entry in support)


def test_support_matrix_from_bits_uses_row_target_column_source_convention() -> None:
    support = support_matrix_from_bits((0, 1, 0, 0, 1, 0))

    assert support == sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    assert support_edge_count(support) == 2
