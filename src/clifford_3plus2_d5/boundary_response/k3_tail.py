"""Explicit finite residual ``K_3`` tail candidates.

The V1 candidate keeps the residual shell fully ``S_3`` symmetric: each shell
has a ``K_3`` adjacency and neighboring shells couple by the identity on the
three residual ports.  This model is intentionally conservative.  If it cannot
split the residual doublet, PMNS/CKM phenomenology should remain parked until
an explicit framed, symmetry-breaking boundary model is supplied.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    commutes_with_all,
    permutation_matrix,
)
from clifford_3plus2_d5.boundary_response.schur import self_energy


def k3_adjacency() -> sp.Matrix:
    """Return the adjacency matrix of the residual complete graph ``K_3``."""

    return sp.Matrix(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )


def finite_k3_tail_hamiltonian(
    shells: int,
    *,
    intra_weight: sp.Expr = sp.Integer(1),
    inter_weight: sp.Expr = sp.Integer(1),
) -> sp.Matrix:
    """Return an ``S_3``-equivariant finite residual-tail Hamiltonian.

    The basis is shell-major: ``(shell_0 e1,e2,e3, shell_1 e1,e2,e3, ...)``.
    """

    if shells < 1:
        raise ValueError("shells must be positive")

    block = intra_weight * k3_adjacency()
    h_q = sp.zeros(3 * shells, 3 * shells)
    for shell in range(shells):
        start = 3 * shell
        h_q[start : start + 3, start : start + 3] = block
    for shell in range(shells - 1):
        left = 3 * shell
        right = 3 * (shell + 1)
        h_q[left : left + 3, right : right + 3] = inter_weight * sp.eye(3)
        h_q[right : right + 3, left : left + 3] = inter_weight * sp.eye(3)
    return h_q


def tail_boundary_coupling(
    shells: int,
    *,
    coupling_weight: sp.Expr = sp.Integer(1),
) -> sp.Matrix:
    """Return ``V: P -> Q`` coupling projected channels to shell zero."""

    if shells < 1:
        raise ValueError("shells must be positive")

    coupling = sp.zeros(3 * shells, 3)
    coupling[0:3, 0:3] = coupling_weight * sp.eye(3)
    return coupling


def shell_s3_action(shells: int, perm: tuple[int, int, int]) -> sp.Matrix:
    """Return the residual ``S_3`` action lifted to every tail shell."""

    if shells < 1:
        raise ValueError("shells must be positive")
    return sp.kronecker_product(sp.eye(shells), permutation_matrix(perm))


def tail_s3_actions(shells: int) -> tuple[sp.Matrix, ...]:
    """Return all lifted residual ``S_3`` actions on the finite tail."""

    return tuple(
        shell_s3_action(shells, perm)
        for perm in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    )


def tail_is_s3_equivariant(h_q: sp.Matrix, shells: int) -> bool:
    """Return true when the finite tail Hamiltonian has residual ``S_3``."""

    return commutes_with_all(h_q, tail_s3_actions(shells))


def k3_tail_self_energy(z: sp.Expr, shells: int = 1) -> sp.Matrix:
    """Return the finite ``K_3`` tail self-energy on projected channels."""

    return self_energy(z, finite_k3_tail_hamiltonian(shells), tail_boundary_coupling(shells))
