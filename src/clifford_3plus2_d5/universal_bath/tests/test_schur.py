"""Tests for Schur and continued-fraction helpers."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.schur import (
    finite_head_continued_fraction,
    jacobi_matrix,
    schur_boundary_response,
)
from clifford_3plus2_d5.universal_bath.tail import period_one_tail, silver_selected_z


def test_finite_head_continued_fraction_matches_schur() -> None:
    z = sp.Symbol("z")
    a_values = (sp.Rational(3, 2), sp.Rational(1, 2))
    b_values = (sp.sqrt(3) / 2,)
    jacobi = jacobi_matrix(a_values, b_values)
    continued_fraction = finite_head_continued_fraction(z, a_values, b_values)
    assert sp.simplify(continued_fraction - schur_boundary_response(jacobi, z)) == 0


def test_terminated_head_depends_on_tail() -> None:
    z = sp.Symbol("z")
    a_values = (sp.Integer(0),)
    b_values = (sp.Integer(1),)
    silver_response = finite_head_continued_fraction(
        z,
        a_values,
        b_values,
        terminator=period_one_tail(z),
    )
    constant_response = finite_head_continued_fraction(
        z,
        a_values,
        b_values,
        terminator=sp.Rational(1, 3),
    )
    probe = silver_selected_z()
    assert sp.simplify(silver_response.subs(z, probe) - constant_response.subs(z, probe)) != 0


def test_invalid_link_counts_are_rejected() -> None:
    z = sp.Symbol("z")
    try:
        finite_head_continued_fraction(z, (sp.Integer(0), sp.Integer(1)), ())
    except ValueError:
        pass
    else:
        raise AssertionError("invalid finite head was accepted")

