"""Universal period-one silver tail."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class SilverTailPayload:
    """Exact identities for the universal period-one tail."""

    z: sp.Symbol
    selected_z: sp.Expr
    epsilon: sp.Expr
    tail_value: sp.Expr
    fixed_point_residual: sp.Expr
    selected_value_matches_epsilon: bool
    alternate_tail_changes_value: bool


def silver_epsilon() -> sp.Expr:
    """Return the silver contraction epsilon = sqrt(2) - 1."""

    return sp.sqrt(2) - 1


def silver_selected_z() -> sp.Expr:
    """Return the BB marginal-stability probe z = 2 sqrt(2)."""

    return 2 * sp.sqrt(2)


def period_one_tail(z: sp.Expr) -> sp.Expr:
    """Return the retarded Weyl tail satisfying t = 1 / (z - t)."""

    return sp.Rational(1, 2) * (z - sp.sqrt(z**2 - 4))


def constant_terminator(value: sp.Expr) -> sp.Expr:
    """Return a constant terminator control."""

    return value


def tail_fixed_point_residual(z: sp.Expr) -> sp.Expr:
    """Return t(z) - 1/(z - t(z))."""

    tail = period_one_tail(z)
    return sp.simplify(tail - 1 / (z - tail))


def silver_tail_payload() -> SilverTailPayload:
    """Return exact identities and controls for the universal tail."""

    z = sp.Symbol("z")
    selected_z = silver_selected_z()
    epsilon = silver_epsilon()
    tail_value = sp.simplify(period_one_tail(selected_z))
    fixed_point_residual = tail_fixed_point_residual(z)
    alternate_value = constant_terminator(sp.Rational(1, 2))
    return SilverTailPayload(
        z=z,
        selected_z=selected_z,
        epsilon=epsilon,
        tail_value=tail_value,
        fixed_point_residual=fixed_point_residual,
        selected_value_matches_epsilon=sp.simplify(tail_value - epsilon) == 0,
        alternate_tail_changes_value=sp.simplify(alternate_value - tail_value) != 0,
    )

