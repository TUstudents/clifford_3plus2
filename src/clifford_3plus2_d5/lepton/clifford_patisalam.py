"""Pati-Salam factorization of the Cl(0,10) chiral-16 carrier.

Session 18 keeps the same real chiral-16 target as Session 17 but changes
the Clifford factorization from ``Cl(0,8) tensor Cl(0,2)`` to
``Cl(0,6) tensor Cl(0,4)``.  This makes the Pati-Salam Lie algebra visible:

``Spin(0,6) x Spin(0,4) ~= SU(4) x SU(2)_L x SU(2)_R``.

The chosen complex structure is a right-quaternionic unit in the commutant of
the ``Cl(0,4)`` factor.  It is a residual H-unit choice, but it is compatible
with the full Pati-Salam algebra.  A simple Spin(4) bivector also squares to
``-I``; the audit records that it breaks both SU(2) factors to U(1) Cartans,
so it is not the default chiral-16 complex structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_lie import (
    basis_span_dimension,
    is_skew_symmetric,
)

MatrixTuple = tuple[sp.Matrix, ...]


def _identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _basis_left_inverse(basis: sp.Matrix) -> sp.Matrix:
    return ((basis.T * basis).inv() * basis.T).applyfunc(sp.simplify)


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


def _clifford_relations_pass(gammas: Sequence[sp.Matrix]) -> bool:
    identity = _identity(gammas[0].rows)
    zero = _zero(gammas[0].rows)
    for left_index, left in enumerate(gammas):
        for right_index, right in enumerate(gammas):
            anticommutator = (left * right + right * left).applyfunc(sp.simplify)
            expected = -2 * identity if left_index == right_index else zero
            if anticommutator != expected:
                return False
    return True


def _volume_element(gammas: Sequence[sp.Matrix]) -> sp.Matrix:
    volume = _identity(gammas[0].rows)
    for gamma in gammas:
        volume = (volume * gamma).applyfunc(sp.simplify)
    return volume


def _matrix_span_basis(matrices: Sequence[sp.Matrix]) -> MatrixTuple:
    if not matrices:
        return ()
    rows = matrices[0].rows
    cols = matrices[0].cols
    vector_columns = [matrix.reshape(rows * cols, 1) for matrix in matrices]
    _, pivots = sp.Matrix.hstack(*vector_columns).rref()
    return tuple(matrices[index] for index in pivots)


@lru_cache(maxsize=1)
def cl06_gamma_matrices() -> MatrixTuple:
    """Exact 8x8 real representation of ``Cl(0,6)``."""

    strings = (
        (0, 0, 3),
        (0, 3, 1),
        (1, 3, 2),
        (2, 3, 2),
        (3, 0, 2),
        (3, 1, 1),
    )
    return tuple(_kronecker_string(item) for item in strings)


@lru_cache(maxsize=1)
def cl04_gamma_matrices() -> MatrixTuple:
    """Exact 8x8 real representation of ``Cl(0,4)``."""

    strings = (
        (0, 0, 3),
        (0, 3, 1),
        (1, 3, 2),
        (2, 3, 2),
    )
    return tuple(_kronecker_string(item) for item in strings)


def cl06_relations_pass(gammas: MatrixTuple | None = None) -> bool:
    return _clifford_relations_pass(cl06_gamma_matrices() if gammas is None else gammas)


def cl04_relations_pass(gammas: MatrixTuple | None = None) -> bool:
    return _clifford_relations_pass(cl04_gamma_matrices() if gammas is None else gammas)


def cl04_volume_element(gammas: MatrixTuple | None = None) -> sp.Matrix:
    return _volume_element(cl04_gamma_matrices() if gammas is None else gammas)


@lru_cache(maxsize=1)
def patisalam_cl010_gamma_matrices() -> MatrixTuple:
    """Tensor ``Cl(0,6)`` and ``Cl(0,4)`` into a ``Cl(0,10)`` system."""

    cl06 = cl06_gamma_matrices()
    cl04 = cl04_gamma_matrices()
    omega4 = cl04_volume_element(cl04)
    return tuple(
        sp.kronecker_product(gamma, omega4).applyfunc(sp.simplify)
        for gamma in cl06
    ) + tuple(
        sp.kronecker_product(_identity(8), gamma).applyfunc(sp.simplify)
        for gamma in cl04
    )


def patisalam_cl010_relations_pass(gammas: MatrixTuple | None = None) -> bool:
    return _clifford_relations_pass(
        patisalam_cl010_gamma_matrices() if gammas is None else gammas,
    )


def patisalam_cl010_volume_element(gammas: MatrixTuple | None = None) -> sp.Matrix:
    return _volume_element(patisalam_cl010_gamma_matrices() if gammas is None else gammas)


@lru_cache(maxsize=1)
def cl04_commutant_complex_structures() -> MatrixTuple:
    """Return square-root ``-I`` units in the commutant of the Cl(0,4) action."""

    variables = sp.symbols("x0:64")
    candidate = sp.Matrix(8, 8, variables)
    equations = [
        value
        for gamma in cl04_gamma_matrices()
        for value in candidate * gamma - gamma * candidate
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    units: list[sp.Matrix] = []
    for vector in coefficient_matrix.nullspace():
        matrix = sp.Matrix(8, 8, vector).applyfunc(sp.simplify)
        if _same_matrix(matrix * matrix, -_identity(8)):
            units.append(matrix)
    return tuple(units)


def cl04_chosen_commutant_j() -> sp.Matrix:
    units = cl04_commutant_complex_structures()
    if not units:
        raise RuntimeError("Cl(0,4) commutant supplied no complex structure")
    return units[0]


def patisalam_complex_structure_full() -> sp.Matrix:
    return sp.kronecker_product(_identity(8), cl04_chosen_commutant_j()).applyfunc(sp.simplify)


def patisalam_chirality_operator(gammas: MatrixTuple | None = None) -> sp.Matrix:
    """Return the real chiral involution for this factorization."""

    return (patisalam_cl010_volume_element(gammas) * patisalam_complex_structure_full()).applyfunc(sp.simplify)


def patisalam_chirality_projectors(gammas: MatrixTuple | None = None) -> tuple[sp.Matrix, sp.Matrix]:
    chirality = patisalam_chirality_operator(gammas)
    identity = _identity(chirality.rows)
    return (
        ((identity + chirality) / 2).applyfunc(sp.simplify),
        ((identity - chirality) / 2).applyfunc(sp.simplify),
    )


@lru_cache(maxsize=2)
def patisalam_chiral16_basis_matrix(sign: str = "+") -> sp.Matrix:
    if sign not in {"+", "-"}:
        raise ValueError("sign must be '+' or '-'")
    projector = patisalam_chirality_projectors()[0 if sign == "+" else 1]
    columns = projector.columnspace()
    if len(columns) != 32:
        raise RuntimeError("Pati-Salam chiral projector did not produce rank 32")
    return sp.Matrix.hstack(*columns).applyfunc(sp.simplify)


@lru_cache(maxsize=2)
def _patisalam_chiral16_left_inverse(sign: str = "+") -> sp.Matrix:
    return _basis_left_inverse(patisalam_chiral16_basis_matrix(sign))


def patisalam_chiral16_block_matrix(matrix: sp.Matrix, sign: str = "+") -> sp.Matrix | None:
    if matrix.shape != (64, 64):
        raise ValueError("Pati-Salam chiral16 block extraction expects a 64x64 matrix")
    projector = patisalam_chirality_projectors()[0 if sign == "+" else 1]
    basis = patisalam_chiral16_basis_matrix(sign)
    image = (matrix * basis).applyfunc(sp.simplify)
    if not _same_matrix(projector * image, image):
        return None
    return (_patisalam_chiral16_left_inverse(sign) * image).applyfunc(sp.simplify)


def patisalam_chosen_complex_structure(sign: str = "+") -> sp.Matrix:
    block = patisalam_chiral16_block_matrix(patisalam_complex_structure_full(), sign)
    if block is None:
        raise RuntimeError("chosen Pati-Salam complex structure did not preserve chiral16")
    return block


def _spin_generator(first: int, second: int) -> sp.Matrix:
    gammas = patisalam_cl010_gamma_matrices()
    return (sp.Rational(1, 2) * gammas[first] * gammas[second]).applyfunc(sp.simplify)


@lru_cache(maxsize=1)
def spin06_generators_chiral16() -> MatrixTuple:
    generators = []
    for first, second in combinations(range(6), 2):
        block = patisalam_chiral16_block_matrix(_spin_generator(first, second))
        if block is None:
            raise RuntimeError("Spin(0,6) generator failed to preserve chiral16")
        generators.append(block)
    return tuple(generators)


def su4_generators_from_spin06() -> MatrixTuple:
    return spin06_generators_chiral16()


@lru_cache(maxsize=1)
def spin04_generators_chiral16() -> MatrixTuple:
    generators = []
    for first, second in combinations(range(6, 10), 2):
        block = patisalam_chiral16_block_matrix(_spin_generator(first, second))
        if block is None:
            raise RuntimeError("Spin(0,4) generator failed to preserve chiral16")
        generators.append(block)
    return tuple(generators)


def _spin04_generator_by_local_pair(first: int, second: int) -> sp.Matrix:
    return _spin_generator(6 + first, 6 + second)


@lru_cache(maxsize=1)
def su2_l_generators_from_spin04() -> MatrixTuple:
    """Self-dual Spin(4) basis, one SU(2) factor."""

    m01 = _spin04_generator_by_local_pair(0, 1)
    m02 = _spin04_generator_by_local_pair(0, 2)
    m03 = _spin04_generator_by_local_pair(0, 3)
    m12 = _spin04_generator_by_local_pair(1, 2)
    m13 = _spin04_generator_by_local_pair(1, 3)
    m23 = _spin04_generator_by_local_pair(2, 3)
    full = (
        (m01 + m23).applyfunc(sp.simplify),
        (m02 - m13).applyfunc(sp.simplify),
        (m03 + m12).applyfunc(sp.simplify),
    )
    blocks = tuple(patisalam_chiral16_block_matrix(generator) for generator in full)
    if any(block is None for block in blocks):
        raise RuntimeError("SU(2)_L generator failed to preserve chiral16")
    return blocks  # type: ignore[return-value]


@lru_cache(maxsize=1)
def su2_r_generators_from_spin04() -> MatrixTuple:
    """Anti-self-dual Spin(4) basis, the commuting SU(2) factor."""

    m01 = _spin04_generator_by_local_pair(0, 1)
    m02 = _spin04_generator_by_local_pair(0, 2)
    m03 = _spin04_generator_by_local_pair(0, 3)
    m12 = _spin04_generator_by_local_pair(1, 2)
    m13 = _spin04_generator_by_local_pair(1, 3)
    m23 = _spin04_generator_by_local_pair(2, 3)
    full = (
        (m01 - m23).applyfunc(sp.simplify),
        (m02 + m13).applyfunc(sp.simplify),
        (m03 - m12).applyfunc(sp.simplify),
    )
    blocks = tuple(patisalam_chiral16_block_matrix(generator) for generator in full)
    if any(block is None for block in blocks):
        raise RuntimeError("SU(2)_R generator failed to preserve chiral16")
    return blocks  # type: ignore[return-value]


def patisalam_spin_span_dimensions() -> dict[str, int]:
    return {
        "spin06_su4": basis_span_dimension(spin06_generators_chiral16()),
        "spin04": basis_span_dimension(spin04_generators_chiral16()),
        "su2_l": basis_span_dimension(su2_l_generators_from_spin04()),
        "su2_r": basis_span_dimension(su2_r_generators_from_spin04()),
    }


def patisalam_su2_factors_commute() -> bool:
    return all(
        _same_matrix(left * right, right * left)
        for left in su2_l_generators_from_spin04()
        for right in su2_r_generators_from_spin04()
    )


def patisalam_all_commute_with_chosen_j(matrices: Sequence[sp.Matrix]) -> bool:
    chosen_j = patisalam_chosen_complex_structure()
    return all(_same_matrix(chosen_j * matrix, matrix * chosen_j) for matrix in matrices)


def patisalam_generator_is_valid(generator: sp.Matrix) -> bool:
    return generator.shape == (32, 32) and is_skew_symmetric(generator)


def spin04_simple_bivector_j(sign: str = "+") -> sp.Matrix:
    """A simple Spin(4) bivector complex structure for comparison."""

    gammas = patisalam_cl010_gamma_matrices()
    block = patisalam_chiral16_block_matrix((gammas[6] * gammas[7]).applyfunc(sp.simplify), sign)
    if block is None:
        raise RuntimeError("simple Spin(4) bivector did not preserve chiral16")
    return block


def spin04_simple_bivector_j_audit() -> dict[str, object]:
    simple_j = spin04_simple_bivector_j()
    su2_l = su2_l_generators_from_spin04()
    su2_r = su2_r_generators_from_spin04()
    return {
        "squares_to_minus_identity": _same_matrix(simple_j * simple_j, -_identity(32)),
        "commuting_su2_l_generators": sum(
            _same_matrix(simple_j * generator, generator * simple_j)
            for generator in su2_l
        ),
        "commuting_su2_r_generators": sum(
            _same_matrix(simple_j * generator, generator * simple_j)
            for generator in su2_r
        ),
        "note": "A simple Spin(4) bivector J breaks both SU(2) factors to U(1) Cartans.",
    }


def patisalam_j_compatibility_audit() -> dict[str, object]:
    chosen_j = patisalam_chosen_complex_structure()
    return {
        "chosen_j_source": "right-quaternionic Cl(0,4) commutant",
        "chosen_j_squares_to_minus_identity": _same_matrix(
            chosen_j * chosen_j,
            -_identity(32),
        ),
        "chosen_j_is_orthogonal": _same_matrix(chosen_j.T * chosen_j, _identity(32)),
        "chosen_j_commutes_with_su4": patisalam_all_commute_with_chosen_j(
            su4_generators_from_spin06(),
        ),
        "chosen_j_commutes_with_su2_l": patisalam_all_commute_with_chosen_j(
            su2_l_generators_from_spin04(),
        ),
        "chosen_j_commutes_with_su2_r": patisalam_all_commute_with_chosen_j(
            su2_r_generators_from_spin04(),
        ),
        "simple_spin04_bivector_j": spin04_simple_bivector_j_audit(),
    }


def patisalam_audit_payload() -> dict[str, object]:
    plus, minus = patisalam_chirality_projectors()
    dimensions = patisalam_spin_span_dimensions()
    return {
        "factorization": "Cl(0,10) = Cl(0,6) tensor Cl(0,4)",
        "cl06_relations_pass": cl06_relations_pass(),
        "cl04_relations_pass": cl04_relations_pass(),
        "cl010_relations_pass": patisalam_cl010_relations_pass(),
        "chirality_ranks": (plus.rank(), minus.rank()),
        "spin06_su4_dimension": dimensions["spin06_su4"],
        "spin04_dimension": dimensions["spin04"],
        "su2_l_dimension": dimensions["su2_l"],
        "su2_r_dimension": dimensions["su2_r"],
        "su2_l_su2_r_commute": patisalam_su2_factors_commute(),
        "j_compatibility": patisalam_j_compatibility_audit(),
        "sm_breaking_note": "SU(4) -> SU(3)c x U(1)_{B-L} and Y = T3_R + (B-L)/2 are left to Session 19.",
    }


def independent_basis_for_generators(generators: Sequence[sp.Matrix]) -> MatrixTuple:
    return _matrix_span_basis(tuple(generators))
