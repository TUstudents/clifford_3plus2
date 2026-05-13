"""Exact SymPy matrix primitives for the real carrier."""

from __future__ import annotations

import sympy as sp


def epsilon() -> sp.Matrix:
    """Return the real quarter-turn matrix on R^2."""

    return sp.Matrix([[0, -1], [1, 0]])


def identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return left * right - right * left


def is_zero_matrix(matrix: sp.Matrix) -> bool:
    return all(value == 0 for value in matrix)


def skew_generator(row: int, column: int, dimension: int = 10) -> sp.Matrix:
    """Return A_ij = E_ij - E_ji using zero-based indices."""

    if row == column:
        raise ValueError("skew generator needs two distinct indices")
    if not (0 <= row < dimension and 0 <= column < dimension):
        raise ValueError("skew generator indices out of range")

    generator = sp.zeros(dimension)
    generator[row, column] = 1
    generator[column, row] = -1
    return generator
