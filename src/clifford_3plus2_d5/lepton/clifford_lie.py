"""Minimal Clifford Lie data for the Session 16 continuum-limit work.

This module is deliberately small. It exposes exact bases and complements
needed by the next dynamics session; it does not try to re-audit textbook
branching theory.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_dynamics import (
    chiral_basis_matrix,
    chiral_so8_generators,
)
from clifford_3plus2_d5.lepton.clifford_octonion import (
    octonion_derivation_basis,
    su3_stabilizer_basis,
)

MatrixTuple = tuple[sp.Matrix, ...]


def unit_octonion_vector(index: int) -> sp.Matrix:
    if not 0 <= index < 8:
        raise ValueError("octonion basis index must be in 0..7")
    vector = sp.zeros(8, 1)
    vector[index] = 1
    return vector


def _zero(rows: int, cols: int | None = None) -> sp.Matrix:
    return sp.zeros(rows, rows if cols is None else cols)


def _flatten(matrix: sp.Matrix) -> sp.Matrix:
    return sp.Matrix(matrix.rows * matrix.cols, 1, list(matrix))


def basis_span_dimension(matrices: Sequence[sp.Matrix]) -> int:
    if not matrices:
        return 0
    return sp.Matrix.hstack(*(_flatten(matrix) for matrix in matrices)).rank()


def matrix_in_span(matrix: sp.Matrix, basis: Sequence[sp.Matrix]) -> bool:
    return basis_span_dimension(tuple(basis)) == basis_span_dimension((*basis, matrix))


def all_in_span(matrices: Sequence[sp.Matrix], basis: Sequence[sp.Matrix]) -> bool:
    return all(matrix_in_span(matrix, basis) for matrix in matrices)


def is_skew_symmetric(matrix: sp.Matrix) -> bool:
    return (matrix + matrix.T).applyfunc(sp.simplify) == _zero(matrix.rows, matrix.cols)


def annihilates_vector(matrix: sp.Matrix, vector: sp.Matrix) -> bool:
    return (matrix * vector).applyfunc(sp.simplify) == _zero(matrix.rows, 1)


def chiral_basis_consistency_check() -> bool:
    """Certify the Session 13/14 convention for the positive chiral block.

    The octonion basis is the standard coordinate basis of the extracted
    ``R^8_+`` block. This check records the convention by verifying that the
    extracted chiral basis has full rank and that the coordinate identity is
    the identity in that block.
    """

    basis = chiral_basis_matrix("+")
    gram = (basis.T * basis).applyfunc(sp.simplify)
    return basis.shape == (16, 8) and basis.rank() == 8 and gram.det() != 0


@lru_cache(maxsize=1)
def chiral_so8_bivector_basis() -> MatrixTuple:
    return chiral_so8_generators()


@lru_cache(maxsize=1)
def clifford_g2_basis() -> MatrixTuple:
    return octonion_derivation_basis()


@lru_cache(maxsize=1)
def clifford_su3_basis() -> MatrixTuple:
    return su3_stabilizer_basis(7)


def _basis_complement(ambient_basis: Sequence[sp.Matrix], subspace_basis: Sequence[sp.Matrix]) -> MatrixTuple:
    complement: list[sp.Matrix] = []
    current = list(subspace_basis)
    current_rank = basis_span_dimension(current)
    for candidate in ambient_basis:
        new_rank = basis_span_dimension((*current, candidate))
        if new_rank > current_rank:
            complement.append(candidate)
            current.append(candidate)
            current_rank = new_rank
    return tuple(complement)


@lru_cache(maxsize=1)
def complement_to_g2_in_so8() -> MatrixTuple:
    return _basis_complement(chiral_so8_bivector_basis(), clifford_g2_basis())


@lru_cache(maxsize=1)
def complement_to_su3_in_g2() -> MatrixTuple:
    return _basis_complement(clifford_g2_basis(), clifford_su3_basis())


def imaginary_octonion_transverse_to_e7_basis() -> tuple[sp.Matrix, ...]:
    return tuple(unit_octonion_vector(index) for index in range(1, 7))
