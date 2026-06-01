"""Tests for the canonical nilpotent repair flag."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_adjacency_operator,
    nilpotent_flag_operator,
    nilpotent_order_pass,
    weighted_nilpotent_flag_operator,
)


def test_canonical_nilpotent_flag_has_length_three() -> None:
    operator = nilpotent_flag_operator()

    assert operator == sp.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert operator**2 == sp.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    assert operator**3 == sp.zeros(3, 3)
    assert nilpotent_order_pass(operator)


def test_weighted_nilpotent_flag_keeps_the_same_flag_shape() -> None:
    x, y = sp.symbols("x y")
    operator = weighted_nilpotent_flag_operator(x, y)

    assert operator == sp.Matrix([[0, x, 0], [0, 0, y], [0, 0, 0]])
    assert operator**3 == sp.zeros(3, 3)
    assert flag_adjacency_operator(operator) == sp.Matrix(
        [
            [0, x, 0],
            [x, 0, y],
            [0, y, 0],
        ]
    )
