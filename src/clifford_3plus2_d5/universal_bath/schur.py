"""Schur and continued-fraction helpers for finite heads."""

from __future__ import annotations

import sympy as sp


def finite_head_continued_fraction(
    z: sp.Expr,
    a_values: tuple[sp.Expr, ...],
    b_values: tuple[sp.Expr, ...],
    terminator: sp.Expr | None = None,
) -> sp.Expr:
    """Return the boundary response of a finite Jacobi head.

    ``a_values`` are diagonal Lanczos coefficients. ``b_values`` are the
    off-diagonal coefficients connecting consecutive head sites. If
    ``terminator`` is provided, it is attached after the final head site through
    one additional unit coupling.
    """

    if len(b_values) not in {len(a_values) - 1, len(a_values)}:
        raise ValueError("b_values must describe internal links, optionally plus tail link")
    if terminator is None and len(b_values) != len(a_values) - 1:
        raise ValueError("terminator-free head needs exactly len(a_values)-1 links")
    if terminator is not None and len(b_values) != len(a_values):
        raise ValueError("terminated head needs one tail link after the final site")

    if terminator is None:
        response = 1 / (z - a_values[-1])
        internal_links = b_values
    else:
        response = 1 / (z - a_values[-1] - b_values[-1] ** 2 * terminator)
        internal_links = b_values[:-1]

    for diagonal, link in zip(reversed(a_values[:-1]), reversed(internal_links), strict=True):
        response = 1 / (z - diagonal - link**2 * response)
    return sp.factor(response)


def schur_boundary_response(jacobi: sp.Matrix, z: sp.Expr) -> sp.Expr:
    """Return <e0|(zI-J)^-1|e0> for a finite Jacobi matrix."""

    if jacobi.rows != jacobi.cols:
        raise ValueError("jacobi must be square")
    e0 = sp.zeros(jacobi.rows, 1)
    e0[0, 0] = 1
    return sp.factor((e0.T * (z * sp.eye(jacobi.rows) - jacobi).inv() * e0)[0, 0])


def jacobi_matrix(a_values: tuple[sp.Expr, ...], b_values: tuple[sp.Expr, ...]) -> sp.Matrix:
    """Return a finite symmetric Jacobi matrix."""

    if len(b_values) != len(a_values) - 1:
        raise ValueError("finite Jacobi matrix needs len(a_values)-1 off-diagonal links")
    matrix = sp.zeros(len(a_values))
    for index, value in enumerate(a_values):
        matrix[index, index] = value
    for index, value in enumerate(b_values):
        matrix[index, index + 1] = value
        matrix[index + 1, index] = value
    return matrix

