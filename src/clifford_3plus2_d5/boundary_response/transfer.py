"""Residual ``K_3`` transfer recurrence.

The note's core numerical invariant comes from the decaying branch of

    x_n = 2 x_{n+1} + x_{n+2}.

This module keeps that calculation exact and isolated from any PMNS/CKM
phenomenology.
"""

from __future__ import annotations

import sympy as sp


def transfer_matrix() -> sp.Matrix:
    """Return the transfer matrix for ``(x_n, x_{n+1})`` from the tail."""

    return sp.Matrix([[2, 1], [1, 0]])


def transfer_polynomial(eps: sp.Expr) -> sp.Expr:
    """Return the recurrence residual for ``x_{n+1} = eps x_n``."""

    return sp.expand(eps**2 + 2 * eps - 1)


def epsilon() -> sp.Expr:
    """Return the exact decaying residual transfer factor."""

    return sp.sqrt(2) - 1


def epsilon_squared() -> sp.Expr:
    """Return ``epsilon^2`` in simplified exact form."""

    return sp.simplify(epsilon() ** 2)


def epsilon_fourth() -> sp.Expr:
    """Return ``epsilon^4 = 17 - 12 sqrt(2)`` exactly."""

    return sp.simplify(epsilon() ** 4)


def transfer_verdict() -> str:
    """Return the exact transfer verdict for the residual recurrence."""

    eps = epsilon()
    if sp.simplify(transfer_polynomial(eps)) == 0 and 0 < float(eps) < 1:
        return "TRANSFER_PASS"
    return "TRANSFER_FAIL"
