"""Moment and scalar Jacobi helpers."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.universal_bath.schur import schur_boundary_response


@dataclass(frozen=True)
class DiscreteMeasure:
    """Finite positive measure seen from a source vector."""

    label: str
    poles: tuple[sp.Expr, ...]
    weights: tuple[sp.Expr, ...]


def measure_moment(measure: DiscreteMeasure, power: int) -> sp.Expr:
    """Return sum_i w_i lambda_i^power."""

    if power < 0:
        raise ValueError("power must be nonnegative")
    return sp.simplify(
        sum(
            weight * pole**power
            for pole, weight in zip(measure.poles, measure.weights, strict=True)
        )
    )


def measure_is_normalized(measure: DiscreteMeasure) -> bool:
    """Return whether the weights sum to one."""

    return sp.simplify(sum(measure.weights, sp.Integer(0)) - 1) == 0


def measure_is_positive(measure: DiscreteMeasure) -> bool:
    """Return whether all poles and weights are numerically positive."""

    return measure_is_normalized(measure) and all(
        float(sp.N(value)) > 0 for value in (*measure.poles, *measure.weights)
    )


def stieltjes_response(measure: DiscreteMeasure, z: sp.Symbol | None = None) -> sp.Expr:
    """Return the Stieltjes response sum_i w_i / (z - lambda_i)."""

    z_symbol = sp.Symbol("z") if z is None else z
    return sp.factor(
        sum(
            weight / (z_symbol - pole)
            for pole, weight in zip(measure.poles, measure.weights, strict=True)
        )
    )


def jacobi_from_measure(measure: DiscreteMeasure) -> sp.Matrix:
    """Return the finite Jacobi matrix for a one-, two-, or three-point measure."""

    if not measure_is_normalized(measure):
        raise ValueError("measure weights must sum to one")
    if len(measure.poles) not in {1, 2, 3}:
        raise ValueError("only one-, two-, and three-point measures are supported")

    if len(measure.poles) == 1:
        return sp.Matrix([[measure.poles[0]]])

    moments = [measure_moment(measure, power) for power in range(5)]
    alpha0 = moments[1]
    beta1_squared = sp.factor(moments[2] - alpha0**2)
    beta1 = sp.sqrt(beta1_squared)

    if len(measure.poles) == 2:
        alpha1 = sp.simplify(sum(measure.poles, sp.Integer(0)) - alpha0)
        return sp.Matrix([[alpha0, beta1], [beta1, alpha1]]).applyfunc(sp.factor)

    alpha1 = sp.factor(
        (moments[3] - 2 * alpha0 * moments[2] + alpha0**2 * moments[1])
        / beta1_squared
    )
    x2_q1_squared = sp.factor(
        (moments[4] - 2 * alpha0 * moments[3] + alpha0**2 * moments[2])
        / beta1_squared
    )
    beta2_squared = sp.factor(x2_q1_squared - alpha1**2 - beta1_squared)
    beta2 = sp.sqrt(beta2_squared)
    alpha2 = sp.factor(sum(measure.poles, sp.Integer(0)) - alpha0 - alpha1)
    return sp.Matrix(
        [
            [alpha0, beta1, 0],
            [beta1, alpha1, beta2],
            [0, beta2, alpha2],
        ]
    ).applyfunc(sp.factor)


def jacobi_moment(jacobi: sp.Matrix, power: int) -> sp.Expr:
    """Return <e0|J^power|e0>."""

    if power < 0:
        raise ValueError("power must be nonnegative")
    e0 = sp.zeros(jacobi.rows, 1)
    e0[0, 0] = 1
    if power == 0:
        return sp.Integer(1)
    return sp.simplify((e0.T * (jacobi**power) * e0)[0, 0])


def moment_round_trip(measure: DiscreteMeasure, max_power: int = 5) -> bool:
    """Return whether the Jacobi matrix reproduces finite measure moments."""

    jacobi = jacobi_from_measure(measure)
    return all(
        sp.simplify(measure_moment(measure, power) - jacobi_moment(jacobi, power)) == 0
        for power in range(max_power + 1)
    )


def response_round_trip(measure: DiscreteMeasure) -> bool:
    """Return whether Stieltjes and Jacobi Schur responses agree."""

    z = sp.Symbol("z")
    jacobi = jacobi_from_measure(measure)
    return sp.simplify(stieltjes_response(measure, z) - schur_boundary_response(jacobi, z)) == 0

