"""Exact SM embedding on C^3 plus C^2."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class GaugeGenerator:
    name: str
    matrix: sp.Matrix


def matrix_unit(row: int, column: int, *, dimension: int = 5) -> sp.Matrix:
    if not (0 <= row < dimension and 0 <= column < dimension):
        raise ValueError("matrix unit index out of range")

    matrix = sp.zeros(dimension)
    matrix[row, column] = 1
    return matrix


def complex_projector_3() -> sp.Matrix:
    return sp.diag(1, 1, 1, 0, 0)


def complex_projector_2() -> sp.Matrix:
    return sp.diag(0, 0, 0, 1, 1)


def complex_block_scalar(lambda_3: sp.Expr, lambda_2: sp.Expr) -> sp.Matrix:
    return lambda_3 * complex_projector_3() + lambda_2 * complex_projector_2()


def su3_generators() -> tuple[GaugeGenerator, ...]:
    return (
        GaugeGenerator("E_12", matrix_unit(0, 1)),
        GaugeGenerator("E_21", matrix_unit(1, 0)),
        GaugeGenerator("E_23", matrix_unit(1, 2)),
        GaugeGenerator("E_32", matrix_unit(2, 1)),
        GaugeGenerator("H_12", sp.diag(1, -1, 0, 0, 0)),
        GaugeGenerator("H_23", sp.diag(0, 1, -1, 0, 0)),
    )


def su2_generators() -> tuple[GaugeGenerator, ...]:
    return (
        GaugeGenerator("E_45", matrix_unit(3, 4)),
        GaugeGenerator("E_54", matrix_unit(4, 3)),
        GaugeGenerator("H_45", sp.diag(0, 0, 0, 1, -1)),
    )


def hypercharge_generator() -> GaugeGenerator:
    return GaugeGenerator(
        "Y",
        sp.diag(
            sp.Rational(-1, 3),
            sp.Rational(-1, 3),
            sp.Rational(-1, 3),
            sp.Rational(1, 2),
            sp.Rational(1, 2),
        ),
    )


def sm_gauge_generators() -> tuple[GaugeGenerator, ...]:
    return su3_generators() + su2_generators() + (hypercharge_generator(),)
