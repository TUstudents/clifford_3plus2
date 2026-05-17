"""Infrastructure for a pinned BCC Weyl walk.

This module deliberately does not invent the Bialynicki-Birula hopping
matrices.  It provides shape validation and Bloch-symbol assembly for a
supplied convention.  Session 20 must fill in the exact hop matrices from a
reliable source before any physics verdict is claimed.

Primary source to pin for implementation:

    I. Bialynicki-Birula,
    "Weyl, Dirac, and Maxwell equations on a lattice as unitary cellular
    automata," Phys. Rev. D 49, 6920 (1994), Section II, equations (8)-(11).

If a Meyer or D'Ariano-Perinotti equivalent convention is used instead, the
source and basis/sign mapping must be documented here before the matrices are
exported.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_geometry import (
    Vector3,
    bcc_body_diagonal_directions,
    vector_dot,
)


def validate_weyl_hops(hops: tuple[sp.Matrix, ...]) -> None:
    if len(hops) != 8:
        raise ValueError("a BCC Weyl convention must provide exactly 8 hop matrices")
    for hop in hops:
        if hop.shape != (2, 2):
            raise ValueError("each BCC Weyl hop matrix must be 2x2")


def weyl_bloch_symbol_from_hops(
    epsilon: sp.Symbol | sp.Expr,
    momentum: Vector3,
    hops: tuple[sp.Matrix, ...],
    *,
    directions: tuple[Vector3, ...] | None = None,
) -> sp.Matrix:
    """Assemble ``sum_v exp(i epsilon k.v) H_v`` for supplied hops."""

    validate_weyl_hops(hops)
    bcc_directions = directions or bcc_body_diagonal_directions()
    if len(bcc_directions) != 8:
        raise ValueError("a BCC Weyl convention must provide exactly 8 directions")

    symbol = sp.zeros(2)
    for direction, hop in zip(bcc_directions, hops, strict=True):
        phase = sp.exp(sp.I * epsilon * vector_dot(momentum, direction))
        symbol += phase * hop
    return symbol.applyfunc(sp.simplify)
