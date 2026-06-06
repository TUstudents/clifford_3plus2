"""Tests for moment-to-Jacobi helpers."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.audit import toy_measure
from clifford_3plus2_d5.universal_bath.jacobi import (
    DiscreteMeasure,
    jacobi_from_measure,
    measure_is_positive,
    measure_moment,
    moment_round_trip,
    response_round_trip,
    stieltjes_response,
)


def test_toy_measure_is_positive() -> None:
    measure = toy_measure()
    assert measure_is_positive(measure)
    assert measure_moment(measure, 0) == 1
    assert measure_moment(measure, 1) == sp.Rational(5, 2)


def test_two_point_measure_jacobi_coefficients() -> None:
    measure = DiscreteMeasure(
        "two_point",
        (sp.Integer(0), sp.Integer(2)),
        (sp.Rational(1, 4), sp.Rational(3, 4)),
    )
    jacobi = jacobi_from_measure(measure)
    assert jacobi == sp.Matrix(
        [
            [sp.Rational(3, 2), sp.sqrt(3) / 2],
            [sp.sqrt(3) / 2, sp.Rational(1, 2)],
        ]
    )


def test_measure_round_trips_through_jacobi() -> None:
    measure = toy_measure()
    assert moment_round_trip(measure)
    assert response_round_trip(measure)


def test_stieltjes_response_for_toy_measure() -> None:
    z = sp.Symbol("z")
    measure = toy_measure()
    expected = (z - sp.Rational(3, 2)) / ((z - 1) * (z - 3))
    assert sp.simplify(stieltjes_response(measure, z) - expected) == 0
