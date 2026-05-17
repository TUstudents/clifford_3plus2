"""Small exact Pauli-matrix helpers for spacetime QCA audits."""

from __future__ import annotations

import sympy as sp


def identity2() -> sp.Matrix:
    return sp.eye(2)


def sigma_x() -> sp.Matrix:
    return sp.Matrix([[0, 1], [1, 0]])


def sigma_y() -> sp.Matrix:
    return sp.Matrix([[0, -sp.I], [sp.I, 0]])


def sigma_z() -> sp.Matrix:
    return sp.Matrix([[1, 0], [0, -1]])


def pauli_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    return sigma_x(), sigma_y(), sigma_z()


def matrix_zero(matrix: sp.Matrix) -> bool:
    return matrix.applyfunc(sp.simplify) == sp.zeros(matrix.rows, matrix.cols)


def same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def is_hermitian(matrix: sp.Matrix) -> bool:
    return same_matrix(matrix.H, matrix)


def is_unitary(matrix: sp.Matrix) -> bool:
    if matrix.rows != matrix.cols:
        return False
    return same_matrix(matrix.H * matrix, sp.eye(matrix.rows))
