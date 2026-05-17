"""Bialynicki-Birula BCC Weyl walk.

This module pins the 8-hop Weyl automaton from:

    I. Bialynicki-Birula,
    "Weyl, Dirac, and Maxwell equations on a lattice as unitary cellular
    automata," Phys. Rev. D 49, 6920 (1994), Section II.

The convention implemented here follows the paper's ``W_{+++}``,
``W_{++-}``, ..., ``W_{---}`` ordering with

``q_+ = (1+i)/4``, ``q_- = (1-i)/4``, and the four rank-one matrices
``P_1`` through ``P_4``.  The Bloch phase convention is
``exp(-i epsilon k.h)``, so the right-handed first-order Hamiltonian is
``sigma . k``.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_geometry import (
    Vector3,
    bcc_reciprocal_origin_equivalent,
    bcc_body_diagonal_directions,
    hypercube_bz_corners,
    vector_dot,
)
from clifford_3plus2_d5.spacetime_qca.continuum import (
    effective_generator_from_floquet,
    hamiltonian_from_generator,
)
from clifford_3plus2_d5.spacetime_qca.dirac import block_diag
from clifford_3plus2_d5.spacetime_qca.pauli import pauli_matrices, same_matrix


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
    """Assemble ``sum_v exp(-i epsilon k.v) H_v`` for supplied hops."""

    validate_weyl_hops(hops)
    bcc_directions = directions or bcc_body_diagonal_directions()
    if len(bcc_directions) != 8:
        raise ValueError("a BCC Weyl convention must provide exactly 8 directions")

    symbol = sp.zeros(2)
    for direction, hop in zip(bcc_directions, hops, strict=True):
        phase = sp.exp(-sp.I * epsilon * vector_dot(momentum, direction))
        symbol += phase * hop
    return symbol.applyfunc(sp.simplify)


def bialynicki_birula_directions() -> tuple[Vector3, ...]:
    """Return the source convention directions ``+++``, ``++-``, ..., ``---``."""

    return bcc_body_diagonal_directions(normalized=False)


def bialynicki_birula_projectors() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    p1 = sp.Matrix([[1, 0], [1, 0]])
    p2 = sp.Matrix([[0, 1], [0, 1]])
    p3 = sp.Matrix([[1, 0], [-1, 0]])
    p4 = sp.Matrix([[0, -1], [0, 1]])
    return p1, p2, p3, p4


def bialynicki_birula_hops() -> tuple[sp.Matrix, ...]:
    """Return the eight source-convention Weyl hop matrices."""

    q_plus = (1 + sp.I) / 4
    q_minus = (1 - sp.I) / 4
    p1, p2, p3, p4 = bialynicki_birula_projectors()
    return (
        q_plus * p1,
        q_minus * p2,
        q_minus * p1,
        q_plus * p2,
        q_minus * p3,
        q_plus * p4,
        q_plus * p3,
        q_minus * p4,
    )


def opposite_helicity_hops(hops: tuple[sp.Matrix, ...] | None = None) -> tuple[sp.Matrix, ...]:
    """Return ``W(-h)`` in the same direction order."""

    base_hops = hops or bialynicki_birula_hops()
    directions = bialynicki_birula_directions()
    hop_by_direction = dict(zip(directions, base_hops, strict=True))
    return tuple(hop_by_direction[tuple(-component for component in direction)] for direction in directions)


def bialynicki_birula_s_matrices(
    hops: tuple[sp.Matrix, ...] | None = None,
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return the first-order source matrices ``s_i = sum_h h_i W(h)``."""

    base_hops = hops or bialynicki_birula_hops()
    directions = bialynicki_birula_directions()
    return tuple(
        sum((sp.Integer(direction[axis]) * hop for direction, hop in zip(directions, base_hops, strict=True)), sp.zeros(2))
        .applyfunc(sp.simplify)
        for axis in range(3)
    )


def bcc_weyl_symbol(
    epsilon: sp.Symbol | sp.Expr,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    helicity: str = "right",
) -> sp.Matrix:
    if helicity == "right":
        hops = bialynicki_birula_hops()
    elif helicity == "left":
        hops = opposite_helicity_hops()
    else:
        raise ValueError("helicity must be 'right' or 'left'")
    return weyl_bloch_symbol_from_hops(
        epsilon,
        (kx, ky, kz),
        hops,
        directions=bialynicki_birula_directions(),
    )


def bcc_weyl_effective_generator(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    helicity: str = "right",
) -> sp.Matrix:
    return effective_generator_from_floquet(
        bcc_weyl_symbol(epsilon, kx, ky, kz, helicity=helicity),
        epsilon=epsilon,
        convention="generator",
    )


def bcc_weyl_effective_hamiltonian(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    helicity: str = "right",
) -> sp.Matrix:
    return hamiltonian_from_generator(
        bcc_weyl_effective_generator(epsilon, kx, ky, kz, helicity=helicity),
    )


def expected_weyl_hamiltonian(kx: sp.Expr, ky: sp.Expr, kz: sp.Expr, *, helicity: str = "right") -> sp.Matrix:
    sx, sy, sz = pauli_matrices()
    sign = 1 if helicity == "right" else -1
    if helicity not in {"right", "left"}:
        raise ValueError("helicity must be 'right' or 'left'")
    return (sign * (kx * sx + ky * sy + kz * sz)).applyfunc(sp.simplify)


def bcc_dirac_symbol(epsilon: sp.Symbol | sp.Expr, kx: sp.Expr, ky: sp.Expr, kz: sp.Expr) -> sp.Matrix:
    return block_diag(
        bcc_weyl_symbol(epsilon, kx, ky, kz, helicity="right"),
        bcc_weyl_symbol(epsilon, kx, ky, kz, helicity="left"),
    )


def bcc_dirac_effective_generator(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
) -> sp.Matrix:
    return effective_generator_from_floquet(
        bcc_dirac_symbol(epsilon, kx, ky, kz),
        epsilon=epsilon,
        convention="generator",
    )


def bcc_dirac_effective_hamiltonian(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
) -> sp.Matrix:
    return hamiltonian_from_generator(bcc_dirac_effective_generator(epsilon, kx, ky, kz))


def bcc_weyl_has_gapless_eigenvalue_at(
    epsilon: sp.Symbol | sp.Expr,
    momentum: Vector3,
    *,
    helicity: str = "right",
) -> bool:
    symbol = bcc_weyl_symbol(epsilon, *momentum, helicity=helicity)
    return any(sp.simplify(value - 1) == 0 for value in symbol.eigenvals())


def bcc_cubic_corner_gapless_samples(epsilon: sp.Symbol | sp.Expr) -> tuple[Vector3, ...]:
    return tuple(corner for corner in hypercube_bz_corners(epsilon) if bcc_weyl_has_gapless_eigenvalue_at(epsilon, corner))


def bcc_cubic_corner_gapless_samples_are_reciprocal_origins(epsilon: sp.Symbol | sp.Expr) -> bool:
    return all(bcc_reciprocal_origin_equivalent(corner, epsilon) for corner in bcc_cubic_corner_gapless_samples(epsilon))


def bcc_symbol_unitary_at(
    epsilon: sp.Symbol | sp.Expr,
    momentum: Vector3,
    *,
    helicity: str = "right",
) -> bool:
    symbol = bcc_weyl_symbol(epsilon, *momentum, helicity=helicity)
    return same_matrix(symbol.H * symbol, sp.eye(2))
