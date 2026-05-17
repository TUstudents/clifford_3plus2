"""Exact gate-algebra checks for the 3+2 split."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from clifford_3plus2_d5.exterior import even_basis_3plus2


Matrix = tuple[tuple[Fraction, ...], ...]
FractionInput = Fraction | int | str


@dataclass(frozen=True)
class NamedMatrix:
    name: str
    matrix: Matrix


@dataclass(frozen=True)
class GateAlgebraAudit:
    generators: tuple[str, ...]
    off_block_generators_present: bool
    block_diagonal_gate_algebra: bool
    sm_commutant_gate_algebra: bool
    unsafe_generators: tuple[str, ...]


def parse_fraction(value: object) -> Fraction:
    """Parse exact scalar input, rejecting floats and booleans."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("booleans are not exact matrix scalars")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError(f"unsupported exact scalar: {value!r}")


def parse_fraction_matrix(rows: Sequence[Sequence[object]]) -> Matrix:
    if isinstance(rows, str) or not rows:
        raise ValueError("matrix must be a nonempty rectangular sequence")

    parsed = tuple(tuple(parse_fraction(value) for value in row) for row in rows)
    width = len(parsed[0])
    if width == 0 or any(len(row) != width for row in parsed):
        raise ValueError("matrix must be rectangular and nonempty")
    return parsed


def matrix_shape(matrix: Matrix) -> tuple[int, int]:
    return (len(matrix), len(matrix[0]) if matrix else 0)


def require_square(matrix: Matrix) -> None:
    rows, columns = matrix_shape(matrix)
    if rows == 0 or rows != columns:
        raise ValueError("matrix must be square")


def identity_matrix(dimension: int, *, scale: Fraction = Fraction(1)) -> Matrix:
    return tuple(
        tuple(scale if row == column else Fraction(0) for column in range(dimension))
        for row in range(dimension)
    )


def diagonal_matrix(diagonal: Sequence[FractionInput]) -> Matrix:
    parsed = tuple(parse_fraction(value) for value in diagonal)
    return tuple(
        tuple(value if row == column else Fraction(0) for column, value in enumerate(parsed))
        for row in range(len(parsed))
    )


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    if matrix_shape(left) != matrix_shape(right):
        raise ValueError("matrix shapes do not match")
    return tuple(
        tuple(left_value + right_value for left_value, right_value in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def matrix_subtract(left: Matrix, right: Matrix) -> Matrix:
    if matrix_shape(left) != matrix_shape(right):
        raise ValueError("matrix shapes do not match")
    return tuple(
        tuple(left_value - right_value for left_value, right_value in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_columns = matrix_shape(left)
    right_rows, right_columns = matrix_shape(right)
    if left_columns != right_rows:
        raise ValueError("matrix shapes are not multiplicable")

    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(left_columns))
            for column in range(right_columns)
        )
        for row in range(left_rows)
    )


def commutator(left: Matrix, right: Matrix) -> Matrix:
    return matrix_subtract(matrix_multiply(left, right), matrix_multiply(right, left))


def is_zero_matrix(matrix: Matrix) -> bool:
    return all(value == 0 for row in matrix for value in row)


def is_diagonal_matrix(matrix: Matrix) -> bool:
    require_square(matrix)
    return all(
        value == 0
        for row_index, row in enumerate(matrix)
        for column_index, value in enumerate(row)
        if row_index != column_index
    )


def has_off_block_entries(matrix: Matrix, *, v3_dimension: int = 3) -> bool:
    require_square(matrix)
    dimension = len(matrix)
    if not 0 < v3_dimension < dimension:
        raise ValueError("v3_dimension must define two nonempty blocks")

    return any(
        matrix[row][column] != 0
        for row in range(dimension)
        for column in range(dimension)
        if (row < v3_dimension) != (column < v3_dimension)
    )


def is_block_diagonal(matrix: Matrix, *, v3_dimension: int = 3) -> bool:
    return not has_off_block_entries(matrix, v3_dimension=v3_dimension)


def _is_scalar_block(matrix: Matrix, start: int, stop: int) -> bool:
    scalar = matrix[start][start]
    return all(
        matrix[row][column] == (scalar if row == column else 0)
        for row in range(start, stop)
        for column in range(start, stop)
    )


def is_block_scalar(matrix: Matrix, *, v3_dimension: int = 3) -> bool:
    require_square(matrix)
    dimension = len(matrix)
    if not is_block_diagonal(matrix, v3_dimension=v3_dimension):
        return False
    return _is_scalar_block(matrix, 0, v3_dimension) and _is_scalar_block(
        matrix, v3_dimension, dimension
    )


def audit_one_particle_gate_algebra(
    generators: Sequence[NamedMatrix], *, v3_dimension: int = 3
) -> GateAlgebraAudit:
    off_block: list[str] = []
    unsafe: list[str] = []

    for generator in generators:
        try:
            if has_off_block_entries(generator.matrix, v3_dimension=v3_dimension):
                off_block.append(generator.name)
            if not is_block_scalar(generator.matrix, v3_dimension=v3_dimension):
                unsafe.append(generator.name)
        except ValueError:
            unsafe.append(generator.name)

    return GateAlgebraAudit(
        generators=tuple(generator.name for generator in generators),
        off_block_generators_present=bool(off_block),
        block_diagonal_gate_algebra=not off_block,
        sm_commutant_gate_algebra=not unsafe,
        unsafe_generators=tuple(unsafe),
    )


def number_function_on_even_fock(
    values_by_sector: Mapping[tuple[int, int], FractionInput],
) -> Matrix:
    diagonal = tuple(parse_fraction(values_by_sector[element.sector]) for element in even_basis_3plus2())
    return diagonal_matrix(diagonal)


def is_even_fock_number_function(matrix: Matrix) -> bool:
    basis = even_basis_3plus2()
    if matrix_shape(matrix) != (len(basis), len(basis)) or not is_diagonal_matrix(matrix):
        return False

    values_by_sector: dict[tuple[int, int], Fraction] = {}
    for index, element in enumerate(basis):
        value = matrix[index][index]
        if element.sector in values_by_sector and values_by_sector[element.sector] != value:
            return False
        values_by_sector[element.sector] = value
    return True
