"""Kill gate for interpreting the nilpotent Taylor kernel as a literal Yukawa."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def unit_shift_nilpotent() -> sp.Matrix:
    """Return the length-3 unit-shift nilpotent."""

    return sp.Matrix(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )


def nilpotent_exponential_matrix(x: sp.Expr = sp.Integer(1)) -> sp.Matrix:
    """Return ``exp(xN)`` for the length-3 nilpotent flag."""

    x = sp.sympify(x)
    n_flag = unit_shift_nilpotent()
    return (sp.eye(3) + x * n_flag + x**2 * n_flag**2 / 2).applyfunc(sp.simplify)


def singular_values_at_unit_x() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return exact singular values of literal ``exp(N)``."""

    matrix = nilpotent_exponential_matrix()
    eigenvalues = (matrix.T * matrix).eigenvals()
    values = tuple(sp.sqrt(value) for value in eigenvalues)
    return tuple(sorted(values, key=lambda value: float(sp.N(value)), reverse=True))


def normalized_singular_value_ratios_at_unit_x() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return singular values normalized to the largest singular value."""

    values = singular_values_at_unit_x()
    largest = values[0]
    return tuple(sp.simplify(value / largest) for value in values)


def left_rotation_metric_is_diagonal(x: sp.Expr = sp.Integer(1)) -> bool:
    """Return true if the literal kernel has no left-family mixing metric."""

    matrix = nilpotent_exponential_matrix(x)
    metric = (matrix * matrix.T).applyfunc(sp.simplify)
    return all(metric[row, col] == 0 for row in range(3) for col in range(3) if row != col)


@dataclass(frozen=True)
class LiteralNilpotentYukawaPayload:
    """Payload for the literal nilpotent-Yukawa kill gate."""

    final_verdict: str
    unit_x_singular_values: tuple[sp.Expr, sp.Expr, sp.Expr]
    unit_x_normalized_ratios: tuple[sp.Expr, sp.Expr, sp.Expr]
    left_rotation_metric_diagonal: bool
    literal_matrix_can_be_yukawa: bool
    interpretation: str


def literal_nilpotent_yukawa_payload() -> LiteralNilpotentYukawaPayload:
    """Return the verdict for literal ``exp(xN)`` as a Yukawa matrix."""

    singular_values = singular_values_at_unit_x()
    ratios = normalized_singular_value_ratios_at_unit_x()
    metric_diagonal = left_rotation_metric_is_diagonal()
    checks_pass = singular_values == (sp.Integer(2), sp.Integer(1), sp.Rational(1, 2)) and (
        not metric_diagonal
    )

    if checks_pass:
        final_verdict = "LITERAL_NILPOTENT_YUKAWA_KILL"
        interpretation = (
            "Literal exp(N) has order-one singular values (2,1,1/2), not the "
            "quark hierarchy, and its left metric is not diagonal. The "
            "nilpotent can be a scalar-response jet/bookkeeping operator, but "
            "not the family-space Yukawa matrix if CKM is to remain a boundary "
            "charged-current holonomy."
        )
    else:
        final_verdict = "LITERAL_NILPOTENT_YUKAWA_CONTROL_FAILED"
        interpretation = "The literal nilpotent-Yukawa control did not match expectations."

    return LiteralNilpotentYukawaPayload(
        final_verdict=final_verdict,
        unit_x_singular_values=singular_values,
        unit_x_normalized_ratios=ratios,
        left_rotation_metric_diagonal=metric_diagonal,
        literal_matrix_can_be_yukawa=False,
        interpretation=interpretation,
    )
