"""Phase D-1: BCC body-diagonal Z_3 rotation and Dirac spinor lift.

The body-diagonal 3-fold rotation of the BCC lattice acts as

    sigma : (x, y, z) -> (y, z, x)

This is a Z_3 element of O_h ⊂ SO(3).  Under sigma, the 8 BCC body
diagonals permute with cycle structure (1, 1, 3, 3):

- (+++) and (---) are fixed (axis points).
- Two disjoint 3-cycles on the remaining 6 directions.

The Dirac spinor lift U_3 ∈ SU(2) is the spin-1/2 representation of
sigma:

    U_3 = exp(- i (2 pi / 3) (n . sigma_vec) / 2)
    where n = (1, 1, 1) / sqrt(3).

Note: U_3^3 = -I (not +I), since SO(3) 2*pi rotation lifts to SU(2) -I.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.topology.reuse import (
    bialynicki_birula_directions,
    block_diag,
    pauli_matrices,
)


@lru_cache(maxsize=1)
def body_diagonal_rotation_matrix() -> sp.Matrix:
    """Return the 3x3 cyclic rotation R: (x, y, z) -> (y, z, x).

    R^3 = I, R^T R = I, det(R) = +1.
    """

    return sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


def apply_rotation_to_direction(
    direction: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``R * direction`` as a 3-tuple."""

    rotation = body_diagonal_rotation_matrix()
    vector = sp.Matrix([direction[0], direction[1], direction[2]])
    rotated = rotation * vector
    return (rotated[0, 0], rotated[1, 0], rotated[2, 0])


@lru_cache(maxsize=1)
def bcc_direction_permutation() -> tuple[int, ...]:
    """Return the permutation array ``perm`` where ``directions[perm[i]] = R(directions[i])``.

    The permutation has cycle structure (1, 1, 3, 3): two fixed points
    (the (+,+,+) and (-,-,-) directions on the rotation axis) and two
    3-cycles on the remaining six directions.
    """

    directions = bialynicki_birula_directions()
    permutation = []
    for direction in directions:
        rotated = apply_rotation_to_direction(direction)
        permutation.append(directions.index(rotated))
    return tuple(permutation)


def permutation_cycles() -> tuple[tuple[int, ...], ...]:
    """Return the cycles of ``bcc_direction_permutation()`` as nested tuples."""

    perm = bcc_direction_permutation()
    visited: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in range(len(perm)):
        if start in visited:
            continue
        cycle: list[int] = [start]
        visited.add(start)
        current = perm[start]
        while current != start:
            cycle.append(current)
            visited.add(current)
            current = perm[current]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def cycle_lengths() -> tuple[int, ...]:
    return tuple(sorted((len(c) for c in permutation_cycles()), reverse=True))


@lru_cache(maxsize=1)
def dirac_spinor_lift() -> sp.Matrix:
    """Return ``U_3 ∈ SU(2)`` lifting the body-diagonal Z_3 rotation.

    U_3 = exp(- i (2 pi / 3) (n . sigma_vec) / 2)
    with n = (1, 1, 1) / sqrt(3).

    Computed as cos(pi/3) I - i sin(pi/3) (n . sigma_vec).
    """

    sigma_x, sigma_y, sigma_z = pauli_matrices()
    # n . sigma_vec normalized
    n_dot_sigma = (sigma_x + sigma_y + sigma_z) / sp.sqrt(3)
    angle = sp.Rational(1, 3) * sp.pi  # half of the SO(3) angle (2*pi/3) for spin-1/2
    return (sp.cos(angle) * sp.eye(2) - sp.I * sp.sin(angle) * n_dot_sigma).applyfunc(sp.simplify)


@lru_cache(maxsize=1)
def dirac_spinor_lift_4d() -> sp.Matrix:
    """Return the Dirac 4-spinor lift in chiral basis.

    In chiral basis, an axial rotation acts the same on both 2-component
    chiral blocks (Weyl_R and Weyl_L transform identically under spatial
    rotations).
    """

    u3 = dirac_spinor_lift()
    return block_diag(u3, u3)


def dirac_spinor_lift_cube() -> sp.Matrix:
    """Return ``(U_3)^3``.  Should equal ``-I`` for spin-1/2 lift of order-3 SO(3) rotation."""

    u3 = dirac_spinor_lift()
    return (u3 * u3 * u3).applyfunc(sp.simplify)


def dirac_spinor_lift_is_su2() -> bool:
    """Return whether U_3 is special unitary: U_3^dagger U_3 = I and det = 1."""

    u3 = dirac_spinor_lift()
    identity_check = (u3.H * u3 - sp.eye(2)).applyfunc(sp.simplify) == sp.zeros(2)
    det_check = sp.simplify(u3.det() - 1) == 0
    return identity_check and det_check


def dirac_spinor_lift_cubes_to_minus_identity() -> bool:
    """Return whether ``U_3^3 = -I`` (the SU(2) lift of an order-3 SO(3) element)."""

    cube = dirac_spinor_lift_cube()
    return (cube + sp.eye(2)).applyfunc(sp.simplify) == sp.zeros(2)
