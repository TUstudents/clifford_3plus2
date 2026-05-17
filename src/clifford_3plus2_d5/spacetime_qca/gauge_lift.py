"""Tensor-lift helpers for spacetime/internal background gauge audits.

Session 20 uses constant background links.  For a BCC hop along direction
``v``, the convention is:

``psi(x, t + eps) = sum_v (H_v x U_link) psi(x + eps v, t)``.

With ``U_link = I + eps A + O(eps^2)`` and real-skew internal ``A``, the
first-order Hamiltonian target is:

``H_eff(k, A) = H_space(k) x I_internal + I_space x iA``.

Position-dependent links and site-local gauge transforms are deferred to the
finite real-space QCA session.
"""

from __future__ import annotations

import sympy as sp


def lift_spacetime_operator(spacetime_operator: sp.Matrix, internal_dim: int) -> sp.Matrix:
    return sp.kronecker_product(spacetime_operator, sp.eye(internal_dim))


def lift_internal_operator(spacetime_dim: int, internal_operator: sp.Matrix) -> sp.Matrix:
    return sp.kronecker_product(sp.eye(spacetime_dim), internal_operator)


def background_gauge_hamiltonian(
    spacetime_hamiltonian: sp.Matrix,
    internal_generator: sp.Matrix,
) -> sp.Matrix:
    """Return ``H_space x I + I x iA`` for real-skew internal ``A``."""

    return (
        lift_spacetime_operator(spacetime_hamiltonian, internal_generator.rows)
        + lift_internal_operator(spacetime_hamiltonian.rows, sp.I * internal_generator)
    ).applyfunc(sp.simplify)
