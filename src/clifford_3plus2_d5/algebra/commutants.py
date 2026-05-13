"""Exact commutant and realification helpers for the SM gate oracle."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.sm.embedding import (
    GaugeGenerator,
    complex_block_scalar,
    complex_projector_2,
    complex_projector_3,
    sm_gauge_generators,
)


def commutes_with_all(matrix: sp.Matrix, generators: Sequence[GaugeGenerator]) -> bool:
    return all(is_zero_matrix(commutator(matrix, generator.matrix)) for generator in generators)


def commutes_with_sm_gauge(matrix: sp.Matrix) -> bool:
    return commutes_with_all(matrix, sm_gauge_generators())


def is_scalar_block(matrix: sp.Matrix) -> bool:
    if matrix.rows != matrix.cols or matrix.rows == 0:
        return False
    scalar = matrix[0, 0]
    return matrix == scalar * identity(matrix.rows)


def is_complex_block_scalar_3plus2(matrix: sp.Matrix) -> bool:
    if matrix.shape != (5, 5):
        return False

    color_block = matrix[:3, :3]
    weak_block = matrix[3:5, 3:5]
    upper_right = matrix[:3, 3:5]
    lower_left = matrix[3:5, :3]
    return (
        is_scalar_block(color_block)
        and is_scalar_block(weak_block)
        and upper_right == sp.zeros(3, 2)
        and lower_left == sp.zeros(2, 3)
    )


def is_sm_commutant_matrix(matrix: sp.Matrix) -> bool:
    return matrix.shape == (5, 5) and commutes_with_sm_gauge(matrix)


def safe_commutant_basis() -> tuple[sp.Matrix, sp.Matrix]:
    return (complex_projector_3(), complex_projector_2())


def sm_commutant_basis_from_linear_system() -> tuple[sp.Matrix, ...]:
    variables = sp.symbols("x0:25")
    candidate = sp.Matrix(5, 5, variables)
    equations = [
        value
        for generator in sm_gauge_generators()
        for value in commutator(candidate, generator.matrix)
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(sp.Matrix(5, 5, vector) for vector in coefficient_matrix.nullspace())


def matrix_span_rank(matrices: Sequence[sp.Matrix]) -> int:
    if not matrices:
        return 0
    columns = [sp.Matrix(matrix.rows * matrix.cols, 1, list(matrix)) for matrix in matrices]
    return sp.Matrix.hstack(*columns).rank()


def sm_commutant_basis_matches_expected() -> bool:
    computed = sm_commutant_basis_from_linear_system()
    expected = safe_commutant_basis()
    return (
        matrix_span_rank(computed) == 2
        and matrix_span_rank(expected) == 2
        and matrix_span_rank(computed + expected) == 2
    )


def safe_commutant_element(lambda_3: sp.Expr, lambda_2: sp.Expr) -> sp.Matrix:
    return complex_block_scalar(lambda_3, lambda_2)


def realify(complex_matrix: sp.Matrix) -> sp.Matrix:
    real_part = complex_matrix.applyfunc(lambda value: sp.expand(value).as_real_imag()[0])
    imag_part = complex_matrix.applyfunc(lambda value: sp.expand(value).as_real_imag()[1])
    return real_part.row_join(-imag_part).col_join(imag_part.row_join(real_part))


def is_complex_linear_real(matrix: sp.Matrix) -> bool:
    carrier = standard_real_carrier()
    return matrix.shape == (carrier.dimension, carrier.dimension) and is_zero_matrix(
        commutator(matrix, carrier.complex_structure)
    )


def complex_from_real(matrix: sp.Matrix) -> sp.Matrix | None:
    if not is_complex_linear_real(matrix):
        return None

    dimension = matrix.rows // 2
    real_part = matrix[:dimension, :dimension]
    imag_part = matrix[dimension:, :dimension]
    if matrix[:dimension, dimension:] != -imag_part:
        return None
    if matrix[dimension:, dimension:] != real_part:
        return None
    return real_part + sp.I * imag_part


def safe_real_commutant_basis() -> tuple[sp.Matrix, ...]:
    p3, p2 = safe_commutant_basis()
    return (
        realify(p3),
        realify(p2),
        realify(sp.I * p3),
        realify(sp.I * p2),
    )


def safe_commutant_closure_examples_pass() -> bool:
    left = safe_commutant_element(sp.Rational(2), sp.Rational(-1))
    right = safe_commutant_element(sp.I, sp.Rational(3, 2))
    examples = (
        left,
        right,
        left + right,
        left * right,
        sp.Rational(5, 7) * left,
    )
    return all(
        is_complex_block_scalar_3plus2(example) and is_sm_commutant_matrix(example)
        for example in examples
    )
