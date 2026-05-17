"""Chiral-basis Dirac matrices for spacetime QCA audits.

Conventions:

* The spinor is ordered as ``psi = (psi_R, psi_L)^T``.
* The Hamiltonian alpha matrices are
  ``alpha_i = diag(sigma_i, -sigma_i)``.
* ``gamma^0`` is off-diagonal.
* ``gamma^i = gamma^0 alpha_i`` gives signature ``(+---)``.

The Bloch Hamiltonian audit compares against ``H(k) = alpha . k``.
"""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.pauli import pauli_matrices


def block_diag(*blocks: sp.Matrix) -> sp.Matrix:
    if not blocks:
        return sp.zeros(0, 0)

    rows = sum(block.rows for block in blocks)
    cols = sum(block.cols for block in blocks)
    out = sp.zeros(rows, cols)
    row_offset = 0
    col_offset = 0
    for block in blocks:
        out[row_offset : row_offset + block.rows, col_offset : col_offset + block.cols] = block
        row_offset += block.rows
        col_offset += block.cols
    return out


def gamma0() -> sp.Matrix:
    identity = sp.eye(2)
    return sp.Matrix.vstack(
        sp.Matrix.hstack(sp.zeros(2), identity),
        sp.Matrix.hstack(identity, sp.zeros(2)),
    )


def alpha_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``alpha_x, alpha_y, alpha_z`` in chiral basis."""

    return tuple(block_diag(sigma, -sigma) for sigma in pauli_matrices())


def gamma_spatial_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    g0 = gamma0()
    return tuple((g0 * alpha).applyfunc(sp.simplify) for alpha in alpha_matrices())


def gamma_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    return (gamma0(), *gamma_spatial_matrices())


def gamma5() -> sp.Matrix:
    """Return chirality matrix with right-handed block first."""

    return block_diag(sp.eye(2), -sp.eye(2))


def dirac_hamiltonian(kx: sp.Expr, ky: sp.Expr, kz: sp.Expr) -> sp.Matrix:
    ax, ay, az = alpha_matrices()
    return (kx * ax + ky * ay + kz * az).applyfunc(sp.simplify)


def tensor_internal(matrix: sp.Matrix, internal_dim: int) -> sp.Matrix:
    return sp.kronecker_product(matrix, sp.eye(internal_dim))
