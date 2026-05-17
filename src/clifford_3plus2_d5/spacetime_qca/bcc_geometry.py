"""BCC lattice geometry helpers.

The BCC nearest-neighbor directions used by the 3D Weyl walk are the eight
body diagonals of the cube:

``(+-1, +-1, +-1) / sqrt(3)``.

This module contains only geometry and sample-point utilities.  It does not
define the Bialynicki-Birula hopping matrices.
"""

from __future__ import annotations

from itertools import product

import sympy as sp


Vector3 = tuple[sp.Expr, sp.Expr, sp.Expr]


def bcc_body_diagonal_directions(*, normalized: bool = True) -> tuple[Vector3, ...]:
    """Return the 8 BCC body-diagonal directions in deterministic order."""

    scale = sp.sqrt(3) if normalized else sp.Integer(1)
    return tuple(
        (sp.Integer(sx) / scale, sp.Integer(sy) / scale, sp.Integer(sz) / scale)
        for sx, sy, sz in product((-1, 1), repeat=3)
    )


def vector_dot(left: Vector3, right: Vector3) -> sp.Expr:
    return sp.simplify(sum(a * b for a, b in zip(left, right, strict=True)))


def squared_norm(vector: Vector3) -> sp.Expr:
    return vector_dot(vector, vector)


def hypercube_bz_corners(epsilon: sp.Symbol | sp.Expr) -> tuple[Vector3, ...]:
    """Return the 8 naive cubic Brillouin-zone corners.

    These are the relevant sample points for the hypercubic doubling control:
    each component is either ``0`` or ``pi / epsilon``.
    """

    high = sp.pi / epsilon
    return tuple(
        (sp.simplify(kx), sp.simplify(ky), sp.simplify(kz))
        for kx, ky, kz in product((sp.Integer(0), high), repeat=3)
    )


def origin3() -> Vector3:
    return sp.Integer(0), sp.Integer(0), sp.Integer(0)
