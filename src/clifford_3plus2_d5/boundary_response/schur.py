"""Exact Schur-complement helpers for projected boundary response."""

from __future__ import annotations

import sympy as sp


def matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices are equal after exact simplification."""

    if left.shape != right.shape:
        return False
    return all(sp.simplify(entry) == 0 for entry in left - right)


def self_energy(z: sp.Expr, h_q: sp.Matrix, coupling_v: sp.Matrix) -> sp.Matrix:
    """Return ``Sigma(z) = V.T (z I - H_Q)^-1 V``.

    ``coupling_v`` is the map from projected channels into the unresolved
    sector, so its shape is ``dim(Q) x dim(P)``.  The sidecar's first models
    are exact real Hamiltonian/resolvent models, so transpose is the correct
    adjoint.
    """

    q_dim, p_dim = coupling_v.shape
    if h_q.shape != (q_dim, q_dim):
        raise ValueError("h_q shape must match the unresolved-sector dimension")

    resolvent = (z * sp.eye(q_dim) - h_q).inv()
    sigma = coupling_v.T * resolvent * coupling_v
    if sigma.shape != (p_dim, p_dim):
        raise AssertionError("internal shape error while computing self-energy")
    return sigma.applyfunc(sp.simplify)


def projected_resolvent_denominator(
    z: sp.Expr,
    h_p: sp.Matrix,
    h_q: sp.Matrix,
    coupling_v: sp.Matrix,
) -> sp.Matrix:
    """Return ``z I_P - H_P - Sigma(z)``."""

    p_dim = h_p.rows
    if h_p.shape != (p_dim, p_dim):
        raise ValueError("h_p must be square")
    if coupling_v.cols != p_dim:
        raise ValueError("coupling_v must have dim(P) columns")
    sigma = self_energy(z, h_q, coupling_v)
    return (z * sp.eye(p_dim) - h_p - sigma).applyfunc(sp.simplify)
