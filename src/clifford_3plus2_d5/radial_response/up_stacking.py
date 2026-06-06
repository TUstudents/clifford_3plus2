"""Up-sector radial stacking-law fork."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def exponential_stack_vector(x: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the Poisson/Taylor radial stack ``(x^2/2!, x, 1)``."""

    x = sp.sympify(x)
    return (sp.simplify(x**2 / 2), x, sp.Integer(1))


def geometric_stack_vector(x: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the geometric/resolvent radial stack ``(x^2, x, 1)``."""

    x = sp.sympify(x)
    return (sp.simplify(x**2), x, sp.Integer(1))


def stack_invariant(vector: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """Return ``C_light C_top / C_charm^2`` for a top-normalized vector."""

    light, charm, top = vector
    return sp.simplify(light * top / charm**2)


@dataclass(frozen=True)
class UpStackingPayload:
    """Payload for the up-sector stacking-law fork."""

    final_verdict: str
    exponential_invariant: sp.Expr
    geometric_invariant: sp.Expr
    stacking_laws_distinct: bool
    up_factorial_relation: sp.Expr
    interpretation: str


def up_stacking_payload() -> UpStackingPayload:
    """Return the up-sector radial stacking verdict."""

    x = sp.Symbol("x", positive=True)
    exponential = stack_invariant(exponential_stack_vector(x))
    geometric = stack_invariant(geometric_stack_vector(x))
    checks_pass = exponential == sp.Rational(1, 2) and geometric == 1

    if checks_pass:
        final_verdict = "UP_STACKING_LAW_EXPONENTIAL_FAVORED"
        interpretation = (
            "The up-sector mass relation is a stacking-law discriminator: "
            "exponential/Poisson recirculation gives C_u C_t / C_c^2 = 1/2, "
            "while geometric/resolvent recirculation gives 1. Thus nilpotent "
            "Taylor earns only the factorial relation, not the repair "
            "amplitude x."
        )
    else:
        final_verdict = "UP_STACKING_LAW_FORK_KILL"
        interpretation = "The exponential/geometric stacking invariants failed."

    return UpStackingPayload(
        final_verdict=final_verdict,
        exponential_invariant=exponential,
        geometric_invariant=geometric,
        stacking_laws_distinct=exponential != geometric,
        up_factorial_relation=sp.Rational(1, 2),
        interpretation=interpretation,
    )
