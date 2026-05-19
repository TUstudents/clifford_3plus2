"""Tests for ``j3o_algebra.py`` — Phase 1 J_3(O) construction."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.exceptional.j3o_algebra import (
    J3O,
    j3o_add,
    j3o_basis,
    j3o_basis_octonion,
    j3o_basis_real,
    j3o_bilinear_form,
    j3o_cubic_norm,
    j3o_diagonal,
    j3o_dimension,
    j3o_jordan_product,
    j3o_scalar_multiply,
    j3o_to_real_vector,
    j3o_trace,
    j3o_zero,
    octonion_conjugate,
    real_as_octonion,
)


def test_j3o_dimension_is_twenty_seven() -> None:
    assert j3o_dimension() == 27


def test_j3o_basis_count() -> None:
    basis = j3o_basis()
    assert len(basis) == 27


def test_basis_is_linearly_independent() -> None:
    basis = j3o_basis()
    columns = [j3o_to_real_vector(element) for element in basis]
    matrix = sp.Matrix.hstack(*columns)
    assert matrix.rank() == 27


def test_octonion_conjugate_flips_imaginary_only() -> None:
    octonion = sp.Matrix([1, 2, -3, 4, -5, 6, -7, 8])
    conjugate = octonion_conjugate(octonion)
    expected = sp.Matrix([1, -2, 3, -4, 5, -6, 7, -8])
    assert conjugate == expected


def test_real_as_octonion_embedding() -> None:
    octonion = real_as_octonion(sp.Rational(3, 7))
    assert octonion.shape == (8, 1)
    assert octonion[0, 0] == sp.Rational(3, 7)
    for index in range(1, 8):
        assert octonion[index, 0] == 0


def test_jordan_product_is_symmetric() -> None:
    a = j3o_diagonal(sp.Integer(1), sp.Integer(2), sp.Integer(3))
    b_off_0 = sp.Matrix([0, 1, 2, 3, 0, 0, 0, 0])
    b_off_1 = sp.Matrix([0, 0, 0, 0, 1, 2, 3, 0])
    b_off_2 = sp.Matrix([0, 0, 0, 0, 0, 0, 0, 1])
    b = J3O(
        diagonal=(sp.Integer(4), sp.Integer(5), sp.Integer(6)),
        off_diagonal=(b_off_0, b_off_1, b_off_2),
    )
    ab = j3o_jordan_product(a, b)
    ba = j3o_jordan_product(b, a)
    # Jordan product is symmetric by definition
    assert ab.diagonal == ba.diagonal
    for index in range(3):
        diff = (ab.off_diagonal[index] - ba.off_diagonal[index]).applyfunc(sp.simplify)
        assert diff == sp.zeros(8, 1)


def test_jordan_product_diagonal_against_diagonal() -> None:
    # For diagonal a, b: (a · b)_ii = a_i * b_i.
    a = j3o_diagonal(sp.Integer(2), sp.Integer(3), sp.Integer(5))
    b = j3o_diagonal(sp.Integer(7), sp.Integer(11), sp.Integer(13))
    ab = j3o_jordan_product(a, b)
    assert ab.diagonal == (
        sp.Integer(2 * 7),
        sp.Integer(3 * 11),
        sp.Integer(5 * 13),
    )
    for index in range(3):
        assert ab.off_diagonal[index] == sp.zeros(8, 1)


def test_jordan_power_associativity() -> None:
    # M · (M · M) = (M · M) · M for any M ∈ J_3(O).
    a_off_0 = sp.Matrix([sp.Rational(1, 2), 1, 0, sp.Rational(-1, 3), 0, 1, 0, 0])
    a_off_1 = sp.Matrix([0, sp.Rational(1, 5), 0, 0, 0, 0, 1, 0])
    a_off_2 = sp.Matrix([1, 0, 0, 0, 1, 0, 0, 1])
    a = J3O(
        diagonal=(sp.Rational(1, 7), sp.Integer(2), sp.Rational(-3, 11)),
        off_diagonal=(a_off_0, a_off_1, a_off_2),
    )
    a_squared = j3o_jordan_product(a, a)
    left = j3o_jordan_product(a, a_squared)
    right = j3o_jordan_product(a_squared, a)
    assert left.diagonal == right.diagonal
    for index in range(3):
        diff = (left.off_diagonal[index] - right.off_diagonal[index]).applyfunc(sp.simplify)
        assert diff == sp.zeros(8, 1)


def test_trace_is_linear() -> None:
    a = j3o_diagonal(sp.Integer(1), sp.Integer(2), sp.Integer(3))
    b = j3o_diagonal(sp.Integer(4), sp.Integer(5), sp.Integer(6))
    sum_ab = j3o_add(a, b)
    assert j3o_trace(sum_ab) == j3o_trace(a) + j3o_trace(b)
    scaled = j3o_scalar_multiply(sp.Rational(2, 7), a)
    assert j3o_trace(scaled) == sp.Rational(2, 7) * j3o_trace(a)


def test_bilinear_form_symmetric() -> None:
    a = j3o_basis_real(0)
    b = j3o_basis_octonion(0, 0)
    assert sp.simplify(j3o_bilinear_form(a, b) - j3o_bilinear_form(b, a)) == 0


def test_cubic_norm_of_diagonal_is_product() -> None:
    matrix = j3o_diagonal(sp.Integer(2), sp.Integer(3), sp.Integer(5))
    assert j3o_cubic_norm(matrix) == 2 * 3 * 5


def test_cubic_norm_is_cubic_under_scaling() -> None:
    matrix = j3o_diagonal(sp.Integer(2), sp.Integer(3), sp.Integer(5))
    scaled = j3o_scalar_multiply(sp.Rational(2, 1), matrix)
    norm_scaled = j3o_cubic_norm(scaled)
    norm_original = j3o_cubic_norm(matrix)
    # N(λ M) = λ³ N(M)
    assert norm_scaled == 8 * norm_original


def test_zero_element_trace_is_zero() -> None:
    assert j3o_trace(j3o_zero()) == 0
