"""Exact commutant and realification helpers for the SM gate oracle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp
from sympy.polys.polyerrors import CoercionFailed

from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.sm.embedding import (
    GaugeGenerator,
    complex_block_scalar,
    complex_projector_2,
    complex_projector_3,
    sm_gauge_generators,
)


def _is_zero_expr(value: sp.Expr) -> bool:
    return value == 0 or sp.simplify(value) == 0


def _normalize_expr(value: sp.Expr) -> sp.Expr:
    return sp.simplify(value)


def _flatten_matrix(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    return tuple(matrix)


@dataclass(frozen=True)
class _SpanRow:
    pivot: int
    values: dict[int, sp.Expr]
    coordinates: tuple[sp.Expr, ...]


class IncrementalMatrixSpan:
    """Incremental exact span over flattened matrices.

    SymPy's full rank recomputation is expensive in algebra-closure loops. This
    class keeps a sparse row-echelon representation and tests independence by
    reducing only the new candidate vector.
    """

    def __init__(self, *, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.vector_size = rows * cols
        self.basis_matrices: list[sp.Matrix] = []
        self._echelon_rows: list[_SpanRow] = []

    @property
    def rank(self) -> int:
        return len(self.basis_matrices)

    def add(self, matrix: sp.Matrix) -> bool:
        if matrix.shape != (self.rows, self.cols):
            msg = f"matrix shape {matrix.shape} does not match {(self.rows, self.cols)}"
            raise ValueError(msg)

        vector = list(_flatten_matrix(matrix))
        coordinates = [sp.Integer(0)] * len(self.basis_matrices) + [sp.Integer(1)]
        for row in self._echelon_rows:
            coefficient = _normalize_expr(vector[row.pivot])
            if _is_zero_expr(coefficient):
                vector[row.pivot] = sp.Integer(0)
                continue
            for index, value in row.values.items():
                vector[index] = _normalize_expr(vector[index] - coefficient * value)
            for index, value in enumerate(row.coordinates):
                coordinates[index] = _normalize_expr(coordinates[index] - coefficient * value)

        pivot = self._first_nonzero_index(vector)
        if pivot is None:
            return False

        pivot_value = _normalize_expr(vector[pivot])
        values = {
            index: normalized
            for index, value in enumerate(vector)
            if not _is_zero_expr(normalized := _normalize_expr(value / pivot_value))
        }
        normalized_coordinates = tuple(
            _normalize_expr(value / pivot_value) for value in coordinates
        )
        self.basis_matrices.append(matrix)
        extended_rows = [
            _SpanRow(
                pivot=row.pivot,
                values=row.values,
                coordinates=(*row.coordinates, sp.Integer(0)),
            )
            for row in self._echelon_rows
        ]
        extended_rows.append(
            _SpanRow(
                pivot=pivot,
                values=values,
                coordinates=normalized_coordinates,
            )
        )
        self._echelon_rows = sorted(extended_rows, key=lambda row: row.pivot)
        return True

    def coordinates(self, matrix: sp.Matrix) -> tuple[sp.Expr, ...] | None:
        if matrix.shape != (self.rows, self.cols):
            msg = f"matrix shape {matrix.shape} does not match {(self.rows, self.cols)}"
            raise ValueError(msg)

        vector = list(_flatten_matrix(matrix))
        coordinates = [sp.Integer(0)] * len(self.basis_matrices)
        for row in self._echelon_rows:
            coefficient = _normalize_expr(vector[row.pivot])
            if _is_zero_expr(coefficient):
                vector[row.pivot] = sp.Integer(0)
                continue
            for index, value in row.values.items():
                vector[index] = _normalize_expr(vector[index] - coefficient * value)
            for index, value in enumerate(row.coordinates):
                coordinates[index] = _normalize_expr(coordinates[index] + coefficient * value)

        if self._first_nonzero_index(vector) is not None:
            return None
        return tuple(_normalize_expr(value) for value in coordinates)

    def contains(self, matrix: sp.Matrix) -> bool:
        return self.coordinates(matrix) is not None

    @staticmethod
    def _first_nonzero_index(vector: Sequence[sp.Expr]) -> int | None:
        for index, value in enumerate(vector):
            if not _is_zero_expr(value):
                return index
        return None


_SQRT3 = sp.sqrt(3)
_SQRT3_SYMBOL = sp.Symbol("__sqrt3")
_FZERO = (sp.Rational(0), sp.Rational(0), sp.Rational(0), sp.Rational(0))
_FONE = (sp.Rational(1), sp.Rational(0), sp.Rational(0), sp.Rational(0))
_FBASIS = ((0, 0), (1, 0), (0, 1), (1, 1))
_FBASIS_INDEX = {item: index for index, item in enumerate(_FBASIS)}
_FieldElement = tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational]


def _sqrt3_pair(expr: sp.Expr) -> tuple[sp.Rational, sp.Rational] | None:
    replaced = sp.expand(expr).xreplace({_SQRT3: _SQRT3_SYMBOL})
    try:
        poly = sp.Poly(replaced, _SQRT3_SYMBOL, domain=sp.QQ)
    except (CoercionFailed, sp.PolynomialError, TypeError, ValueError):
        return None
    if poly.degree() > 1:
        return None
    return (
        sp.Rational(poly.coeff_monomial(1)),
        sp.Rational(poly.coeff_monomial(_SQRT3_SYMBOL)),
    )


def _field_from_expr(expr: sp.Expr) -> _FieldElement | None:
    real_part, imag_part = sp.expand(expr).as_real_imag()
    real_pair = _sqrt3_pair(real_part)
    imag_pair = _sqrt3_pair(imag_part)
    if real_pair is None or imag_pair is None:
        return None
    return (real_pair[0], real_pair[1], imag_pair[0], imag_pair[1])


def _field_to_expr(value: _FieldElement) -> sp.Expr:
    return value[0] + value[1] * _SQRT3 + value[2] * sp.I + value[3] * _SQRT3 * sp.I


def _field_is_zero(value: _FieldElement) -> bool:
    return value == _FZERO


def _field_add(left: _FieldElement, right: _FieldElement) -> _FieldElement:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _field_sub(left: _FieldElement, right: _FieldElement) -> _FieldElement:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _field_mul(left: _FieldElement, right: _FieldElement) -> _FieldElement:
    result = [sp.Rational(0), sp.Rational(0), sp.Rational(0), sp.Rational(0)]
    for left_index, left_coefficient in enumerate(left):
        if left_coefficient == 0:
            continue
        left_sqrt_power, left_i_power = _FBASIS[left_index]
        for right_index, right_coefficient in enumerate(right):
            if right_coefficient == 0:
                continue
            sqrt_power = left_sqrt_power + _FBASIS[right_index][0]
            i_power = left_i_power + _FBASIS[right_index][1]
            coefficient = left_coefficient * right_coefficient
            if sqrt_power == 2:
                sqrt_power = 0
                coefficient *= 3
            if i_power == 2:
                i_power = 0
                coefficient *= -1
            result[_FBASIS_INDEX[(sqrt_power, i_power)]] += coefficient
    return tuple(result)  # type: ignore[return-value]


def _sqrt3_pair_mul(
    left: tuple[sp.Rational, sp.Rational],
    right: tuple[sp.Rational, sp.Rational],
) -> tuple[sp.Rational, sp.Rational]:
    return (left[0] * right[0] + 3 * left[1] * right[1], left[0] * right[1] + left[1] * right[0])


def _sqrt3_pair_add(
    left: tuple[sp.Rational, sp.Rational],
    right: tuple[sp.Rational, sp.Rational],
) -> tuple[sp.Rational, sp.Rational]:
    return (left[0] + right[0], left[1] + right[1])


def _sqrt3_pair_neg(
    value: tuple[sp.Rational, sp.Rational],
) -> tuple[sp.Rational, sp.Rational]:
    return (-value[0], -value[1])


def _sqrt3_pair_inv(
    value: tuple[sp.Rational, sp.Rational],
) -> tuple[sp.Rational, sp.Rational]:
    denominator = value[0] ** 2 - 3 * value[1] ** 2
    if denominator == 0:
        raise ZeroDivisionError("field element is not invertible")
    return (value[0] / denominator, -value[1] / denominator)


def _field_inv(value: _FieldElement) -> _FieldElement:
    real = (value[0], value[1])
    imag = (value[2], value[3])
    denominator = _sqrt3_pair_add(
        _sqrt3_pair_mul(real, real),
        _sqrt3_pair_mul(imag, imag),
    )
    denominator_inverse = _sqrt3_pair_inv(denominator)
    inverse_real = _sqrt3_pair_mul(real, denominator_inverse)
    inverse_imag = _sqrt3_pair_mul(_sqrt3_pair_neg(imag), denominator_inverse)
    return (inverse_real[0], inverse_real[1], inverse_imag[0], inverse_imag[1])


def _field_div(left: _FieldElement, right: _FieldElement) -> _FieldElement:
    return _field_mul(left, _field_inv(right))


@dataclass(frozen=True)
class _FastSpanRow:
    pivot: int
    values: dict[int, _FieldElement]
    coordinates: tuple[_FieldElement, ...]


class Sqrt3IMatrixSpan:
    """Exact matrix span over QQ(sqrt(3), i)."""

    def __init__(self, *, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.basis_matrices: list[sp.Matrix] = []
        self._echelon_rows: list[_FastSpanRow] = []

    @property
    def rank(self) -> int:
        return len(self.basis_matrices)

    @staticmethod
    def supports(matrix: sp.Matrix) -> bool:
        return all(_field_from_expr(value) is not None for value in matrix)

    def add(self, matrix: sp.Matrix) -> bool:
        vector = self._field_vector(matrix)
        coordinates = [_FZERO] * len(self.basis_matrices) + [_FONE]
        for row in self._echelon_rows:
            coefficient = vector[row.pivot]
            if _field_is_zero(coefficient):
                continue
            for index, value in row.values.items():
                vector[index] = _field_sub(vector[index], _field_mul(coefficient, value))
            for index, value in enumerate(row.coordinates):
                coordinates[index] = _field_sub(coordinates[index], _field_mul(coefficient, value))

        pivot = self._first_nonzero_index(vector)
        if pivot is None:
            return False

        pivot_value = vector[pivot]
        values = {
            index: normalized
            for index, value in enumerate(vector)
            if not _field_is_zero(normalized := _field_div(value, pivot_value))
        }
        normalized_coordinates = tuple(_field_div(value, pivot_value) for value in coordinates)
        self.basis_matrices.append(matrix)
        extended_rows = [
            _FastSpanRow(
                pivot=row.pivot,
                values=row.values,
                coordinates=(*row.coordinates, _FZERO),
            )
            for row in self._echelon_rows
        ]
        extended_rows.append(
            _FastSpanRow(
                pivot=pivot,
                values=values,
                coordinates=normalized_coordinates,
            )
        )
        self._echelon_rows = sorted(extended_rows, key=lambda row: row.pivot)
        return True

    def coordinates(self, matrix: sp.Matrix) -> tuple[sp.Expr, ...] | None:
        vector = self._field_vector(matrix)
        coordinates = [_FZERO] * len(self.basis_matrices)
        for row in self._echelon_rows:
            coefficient = vector[row.pivot]
            if _field_is_zero(coefficient):
                continue
            for index, value in row.values.items():
                vector[index] = _field_sub(vector[index], _field_mul(coefficient, value))
            for index, value in enumerate(row.coordinates):
                coordinates[index] = _field_add(coordinates[index], _field_mul(coefficient, value))

        if self._first_nonzero_index(vector) is not None:
            return None
        return tuple(_field_to_expr(value) for value in coordinates)

    def contains(self, matrix: sp.Matrix) -> bool:
        return self.coordinates(matrix) is not None

    def _field_vector(self, matrix: sp.Matrix) -> list[_FieldElement]:
        if matrix.shape != (self.rows, self.cols):
            msg = f"matrix shape {matrix.shape} does not match {(self.rows, self.cols)}"
            raise ValueError(msg)
        vector = [_field_from_expr(value) for value in matrix]
        if any(value is None for value in vector):
            raise ValueError("matrix entries are not in QQ(sqrt(3), i)")
        return list(vector)  # type: ignore[arg-type]

    @staticmethod
    def _first_nonzero_index(vector: Sequence[_FieldElement]) -> int | None:
        for index, value in enumerate(vector):
            if not _field_is_zero(value):
                return index
        return None


def exact_matrix_span(
    matrices: Sequence[sp.Matrix] = (),
    *,
    rows: int | None = None,
    cols: int | None = None,
    add_matrices: bool = True,
) -> IncrementalMatrixSpan | Sqrt3IMatrixSpan:
    if matrices:
        rows = matrices[0].rows
        cols = matrices[0].cols
    if rows is None or cols is None:
        raise ValueError("rows and cols are required when matrices is empty")

    use_fast_field = bool(matrices) and all(Sqrt3IMatrixSpan.supports(matrix) for matrix in matrices)
    span: IncrementalMatrixSpan | Sqrt3IMatrixSpan
    span = (
        Sqrt3IMatrixSpan(rows=rows, cols=cols)
        if use_fast_field
        else IncrementalMatrixSpan(rows=rows, cols=cols)
    )
    if add_matrices:
        for matrix in matrices:
            span.add(matrix)
    return span


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
    return exact_matrix_span(matrices).rank


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
