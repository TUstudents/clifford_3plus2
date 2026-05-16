"""Exact Cl(0,8) and octonion stabilizer audit helpers.

This module is an audit surface, not a QCA rule search. It records exactly
which Clifford/octonion structures are derived from algebraic relations and
which choices remain explicit.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import TypeAlias

import sympy as sp

MatrixTuple: TypeAlias = tuple[sp.Matrix, ...]


def _identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _basis_vector(index: int, dimension: int = 8) -> sp.Matrix:
    vector = sp.zeros(dimension, 1)
    vector[index] = 1
    return vector


def _kronecker_string(symbols: tuple[int, ...]) -> sp.Matrix:
    identity_2 = sp.eye(2)
    x_matrix = sp.Matrix([[0, 1], [1, 0]])
    z_matrix = sp.diag(1, -1)
    epsilon = sp.Matrix([[0, 1], [-1, 0]])
    alphabet = (identity_2, x_matrix, z_matrix, epsilon)
    matrix = sp.Matrix([[1]])
    for symbol in symbols:
        matrix = sp.kronecker_product(matrix, alphabet[symbol])
    return matrix


@lru_cache(maxsize=1)
def cl08_gamma_matrices() -> MatrixTuple:
    """Return an exact real 16x16 representation of Cl(0,8).

    The signature is pinned: each gamma squares to ``-I`` and distinct
    gammas anticommute.
    """

    strings = (
        (0, 0, 0, 3),
        (0, 0, 3, 1),
        (0, 1, 3, 2),
        (0, 2, 3, 2),
        (0, 3, 0, 2),
        (0, 3, 1, 1),
        (1, 3, 2, 1),
        (2, 3, 2, 1),
    )
    return tuple(_kronecker_string(item) for item in strings)


def clifford_relations_pass(gammas: MatrixTuple | None = None) -> bool:
    gammas = cl08_gamma_matrices() if gammas is None else gammas
    dimension = gammas[0].rows
    identity = _identity(dimension)
    zero = _zero(dimension)
    for left_index, left in enumerate(gammas):
        for right_index, right in enumerate(gammas):
            anticommutator = (left * right + right * left).applyfunc(sp.simplify)
            expected = -2 * identity if left_index == right_index else zero
            if anticommutator != expected:
                return False
    return True


def volume_element(gammas: MatrixTuple | None = None) -> sp.Matrix:
    gammas = cl08_gamma_matrices() if gammas is None else gammas
    volume = _identity(gammas[0].rows)
    for gamma in gammas:
        volume = (volume * gamma).applyfunc(sp.simplify)
    return volume


def chirality_projectors(gammas: MatrixTuple | None = None) -> tuple[sp.Matrix, sp.Matrix]:
    gammas = cl08_gamma_matrices() if gammas is None else gammas
    omega = volume_element(gammas)
    identity = _identity(omega.rows)
    return (
        ((identity + omega) / 2).applyfunc(sp.simplify),
        ((identity - omega) / 2).applyfunc(sp.simplify),
    )


def matrix_commutant_basis(
    generators: MatrixTuple,
    *,
    dimension: int,
) -> MatrixTuple:
    variables = sp.symbols(f"x0:{dimension * dimension}")
    candidate = sp.Matrix(dimension, dimension, variables)
    equations = [
        value
        for generator in generators
        for value in candidate * generator - generator * candidate
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(
        sp.Matrix(dimension, dimension, vector).applyfunc(sp.simplify)
        for vector in coefficient_matrix.nullspace()
    )


@lru_cache(maxsize=1)
def cl08_full_commutant_basis() -> MatrixTuple:
    return matrix_commutant_basis(cl08_gamma_matrices(), dimension=16)


@lru_cache(maxsize=1)
def cl08_even_generators() -> MatrixTuple:
    gammas = cl08_gamma_matrices()
    return tuple((gammas[i] * gammas[j]).applyfunc(sp.simplify) for i, j in combinations(range(8), 2))


@lru_cache(maxsize=1)
def cl08_even_commutant_basis() -> MatrixTuple:
    return matrix_commutant_basis(cl08_even_generators(), dimension=16)


def cl02_generators() -> tuple[sp.Matrix, sp.Matrix]:
    identity_2 = sp.eye(2)
    z_matrix = sp.diag(1, -1)
    epsilon = sp.Matrix([[0, 1], [-1, 0]])
    first = sp.kronecker_product(epsilon, identity_2)
    second = sp.kronecker_product(z_matrix, epsilon)
    return first, second


def cl02_complex_structure_candidates() -> MatrixTuple:
    first, second = cl02_generators()
    return (
        first,
        second,
        (first * second).applyfunc(sp.simplify),
    )


def octonion_fano_triples() -> tuple[tuple[int, int, int], ...]:
    """Chosen oriented Fano table for octonion multiplication.

    This table choice is explicit. Different valid tables are related by
    relabeling/sign choices, but the audit does not pretend that the table is
    derived from the Clifford representation alone.
    """

    return (
        (1, 2, 3),
        (1, 4, 5),
        (1, 6, 7),
        (4, 2, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 5, 6),
    )


@lru_cache(maxsize=1)
def octonion_structure_constants() -> dict[tuple[int, int], tuple[int, int]]:
    table: dict[tuple[int, int], tuple[int, int]] = {}
    for index in range(8):
        table[(0, index)] = (1, index)
        table[(index, 0)] = (1, index)
    for index in range(1, 8):
        table[(index, index)] = (-1, 0)
    for first, second, third in octonion_fano_triples():
        for left, right, product in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            table[(left, right)] = (1, product)
            table[(right, left)] = (-1, product)
    return table


def octonion_multiply(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    table = octonion_structure_constants()
    product = sp.zeros(8, 1)
    for left_index, left_coeff in enumerate(left):
        if left_coeff == 0:
            continue
        for right_index, right_coeff in enumerate(right):
            if right_coeff == 0:
                continue
            sign, basis_index = table[(left_index, right_index)]
            product[basis_index] += sign * left_coeff * right_coeff
    return product.applyfunc(sp.simplify)


def octonion_left_multiplication(index: int) -> sp.Matrix:
    matrix = sp.zeros(8)
    left = _basis_vector(index)
    for column in range(8):
        product = octonion_multiply(left, _basis_vector(column))
        matrix[:, column] = product
    return matrix


def octonion_right_multiplication(index: int) -> sp.Matrix:
    matrix = sp.zeros(8)
    right = _basis_vector(index)
    for column in range(8):
        product = octonion_multiply(_basis_vector(column), right)
        matrix[:, column] = product
    return matrix


@lru_cache(maxsize=1)
def octonion_derivation_basis() -> MatrixTuple:
    variables = sp.symbols("d0:64")
    derivation = sp.Matrix(8, 8, variables)
    equations = list(derivation * _basis_vector(0))
    for left_index in range(8):
        left = _basis_vector(left_index)
        for right_index in range(8):
            right = _basis_vector(right_index)
            product = octonion_multiply(left, right)
            defect = (
                derivation * product
                - octonion_multiply(derivation * left, right)
                - octonion_multiply(left, derivation * right)
            )
            equations.extend(list(defect))
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(
        sp.Matrix(8, 8, vector).applyfunc(sp.simplify)
        for vector in coefficient_matrix.nullspace()
    )


def su3_stabilizer_basis(imaginary_index: int = 7) -> MatrixTuple:
    if imaginary_index not in range(1, 8):
        raise ValueError("imaginary_index must be one of 1..7")
    g2_basis = octonion_derivation_basis()
    coefficients = sp.symbols(f"c0:{len(g2_basis)}")
    candidate = sp.zeros(8)
    for coefficient, basis_matrix in zip(coefficients, g2_basis, strict=True):
        candidate += coefficient * basis_matrix
    equations = list(candidate * _basis_vector(imaginary_index))
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, coefficients)
    stabilizer_coordinates = coefficient_matrix.nullspace()
    stabilizer = []
    for coordinate_vector in stabilizer_coordinates:
        matrix = sp.zeros(8)
        for coordinate, basis_matrix in zip(coordinate_vector, g2_basis, strict=True):
            matrix += coordinate * basis_matrix
        stabilizer.append(matrix.applyfunc(sp.simplify))
    return tuple(stabilizer)


def clifford_octonion_audit_payload() -> dict[str, object]:
    gammas = cl08_gamma_matrices()
    omega = volume_element(gammas)
    chirality = chirality_projectors(gammas)
    cl02_candidates = cl02_complex_structure_candidates()
    g2_basis = octonion_derivation_basis()
    su3_basis = su3_stabilizer_basis(7)
    return {
        "signature": "Cl(0,8)",
        "gamma_count": len(gammas),
        "gamma_dimension": gammas[0].rows,
        "clifford_relations_pass": clifford_relations_pass(gammas),
        "volume_square": "I" if omega * omega == _identity(16) else "not_I",
        "chirality_ranks": tuple(projector.rank() for projector in chirality),
        "full_commutant_dimension": len(cl08_full_commutant_basis()),
        "even_commutant_dimension": len(cl08_even_commutant_basis()),
        "cl02_j_candidate_count": len(cl02_candidates),
        "octonion_table_choice": "fano:" + ",".join(str(item) for item in octonion_fano_triples()),
        "g2_derivation_dimension": len(g2_basis),
        "su3_stabilizer_dimension": len(su3_basis),
        "su3_fixed_imaginary_direction": "e7",
        "choices": (
            "octonion multiplication table",
            "imaginary direction e7",
            "which Cl(0,2) unit is called J",
        ),
    }
