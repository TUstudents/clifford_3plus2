"""Naive hypercubic control for the BCC doubling audit."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_geometry import Vector3, hypercube_bz_corners
from clifford_3plus2_d5.spacetime_qca.dirac import dirac_hamiltonian
from clifford_3plus2_d5.spacetime_qca.pauli import same_matrix


def naive_hypercube_hamiltonian(
    epsilon: sp.Symbol | sp.Expr,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
) -> sp.Matrix:
    """Return the naive central-difference Dirac Hamiltonian on a cubic lattice."""

    return dirac_hamiltonian(
        sp.sin(kx * epsilon) / epsilon,
        sp.sin(ky * epsilon) / epsilon,
        sp.sin(kz * epsilon) / epsilon,
    )


def hypercube_continuum_target(kx: sp.Expr, ky: sp.Expr, kz: sp.Expr) -> sp.Matrix:
    return dirac_hamiltonian(kx, ky, kz)


def hypercube_is_gapless_at(
    epsilon: sp.Symbol | sp.Expr,
    momentum: Vector3,
) -> bool:
    hamiltonian = naive_hypercube_hamiltonian(epsilon, *momentum)
    return same_matrix(hamiltonian, sp.zeros(4))


def hypercube_gapless_corners(epsilon: sp.Symbol | sp.Expr) -> tuple[Vector3, ...]:
    return tuple(
        corner for corner in hypercube_bz_corners(epsilon) if hypercube_is_gapless_at(epsilon, corner)
    )


def hypercube_corner_eigenvalues(
    epsilon: sp.Symbol | sp.Expr,
    momentum: Vector3,
) -> dict[sp.Expr, int]:
    """Return exact eigenvalues of the naive cubic Hamiltonian at a sample point."""

    return naive_hypercube_hamiltonian(epsilon, *momentum).eigenvals()
