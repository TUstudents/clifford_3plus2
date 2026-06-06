"""OPUC/CMV helpers for the free unitary tail."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class OPUCFreeTailPayload:
    """Exact checks for the CMV/OPUC free-tail criterion."""

    free_tail: tuple[sp.Expr, ...]
    all_verblunsky_zero: bool
    nonzero_head_then_free_tail: tuple[sp.Expr, ...]
    finite_head_count: int
    free_remainder_after_head: bool
    interpretation: str


def free_verblunsky_tail(length: int) -> tuple[sp.Expr, ...]:
    """Return ``length`` zero Verblunsky coefficients for the free CMV tail."""

    if length < 0:
        raise ValueError("length must be nonnegative")
    return tuple(sp.Integer(0) for _ in range(length))


def is_free_verblunsky_tail(coefficients: tuple[sp.Expr, ...]) -> bool:
    """Return whether all Verblunsky coefficients vanish."""

    return all(sp.simplify(coefficient) == 0 for coefficient in coefficients)


def schur_algorithm_step(zeta: sp.Expr, f_n: sp.Expr, gamma_n: sp.Expr) -> sp.Expr:
    """Return one Schur-algorithm step for real or symbolic gamma.

    This helper is intentionally minimal. Sector work may need complex
    conjugation conventions, but Session 01 only fixes the free-tail criterion.
    """

    return sp.factor((f_n - gamma_n) / (zeta * (1 - gamma_n * f_n)))


def opuc_free_tail_payload() -> OPUCFreeTailPayload:
    """Return the CMV/OPUC free-tail payload."""

    free_tail = free_verblunsky_tail(4)
    alpha0, alpha1 = sp.symbols("alpha0 alpha1")
    finite_head_then_tail = (alpha0, alpha1, *free_verblunsky_tail(4))
    head_count = 2
    free_remainder = is_free_verblunsky_tail(finite_head_then_tail[head_count:])
    return OPUCFreeTailPayload(
        free_tail=free_tail,
        all_verblunsky_zero=is_free_verblunsky_tail(free_tail),
        nonzero_head_then_free_tail=finite_head_then_tail,
        finite_head_count=head_count,
        free_remainder_after_head=free_remainder,
        interpretation=(
            "The CMV/OPUC version of the universal tail is alpha_n = 0 after "
            "the finite head. Chiral sector phases should appear in the finite "
            "head Verblunsky coefficients, while the remainder is the free "
            "unitary tail."
        ),
    )

