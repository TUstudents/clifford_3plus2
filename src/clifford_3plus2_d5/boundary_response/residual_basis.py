"""Residual family basis and symmetry projectors.

After vacuum framing, the proposed residual family basis is

    u = (1, 1, 1) / sqrt(3)
    a = (2, -1, -1) / sqrt(6)
    b = (0, 1, -1) / sqrt(2)

The key diagnostic is representation-theoretic: full residual ``S_3`` leaves
only singlet-plus-doublet operators invariant, while the proposed
``K_nu = epsilon^2 P_u + P_b`` splits the doublet.  Therefore an unbroken
``S_3`` residual tail cannot produce ``K_nu`` by Schur complement.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable

import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import epsilon_squared


def standard_basis() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``(e1, e2, e3)`` as exact column vectors."""

    return (
        sp.Matrix([1, 0, 0]),
        sp.Matrix([0, 1, 0]),
        sp.Matrix([0, 0, 1]),
    )


def residual_vectors() -> dict[str, sp.Matrix]:
    """Return the exact residual basis vectors ``u``, ``a``, and ``b``."""

    return {
        "u": sp.Matrix([1, 1, 1]) / sp.sqrt(3),
        "a": sp.Matrix([2, -1, -1]) / sp.sqrt(6),
        "b": sp.Matrix([0, 1, -1]) / sp.sqrt(2),
    }


def residual_basis_matrix(order: tuple[str, ...] = ("a", "u", "b")) -> sp.Matrix:
    """Return the residual basis matrix with named columns."""

    vectors = residual_vectors()
    return sp.Matrix.hstack(*(vectors[name] for name in order))


def projector(vector: sp.Matrix) -> sp.Matrix:
    """Return the real rank-one projector onto a normalized column vector."""

    return sp.simplify(vector * vector.T)


def residual_projectors() -> dict[str, sp.Matrix]:
    """Return projectors onto ``u``, ``a``, and ``b``."""

    vectors = residual_vectors()
    return {name: projector(vec) for name, vec in vectors.items()}


def k_nu_operator() -> sp.Matrix:
    """Return the proposed neutrino core operator ``epsilon^2 P_u + P_b``."""

    projectors = residual_projectors()
    return sp.simplify(epsilon_squared() * projectors["u"] + projectors["b"])


def permutation_matrix(perm: tuple[int, int, int]) -> sp.Matrix:
    """Return the permutation matrix mapping ``e_i`` to ``e_perm[i]``.

    The convention makes each column the image of the corresponding basis
    vector.
    """

    mat = sp.zeros(3, 3)
    for src, dst in enumerate(perm):
        mat[dst, src] = 1
    return mat


def s3_permutation_matrices() -> tuple[sp.Matrix, ...]:
    """Return all six ``S_3`` permutation matrices on residual channels."""

    return tuple(permutation_matrix(tuple(perm)) for perm in permutations((0, 1, 2)))


def selected_port_s2_matrices() -> tuple[sp.Matrix, ...]:
    """Return the ``S_2`` subgroup preserving the first selected port."""

    return (sp.eye(3), permutation_matrix((0, 2, 1)))


def commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    """Return ``[left, right]`` with entrywise simplification."""

    return (left * right - right * left).applyfunc(sp.simplify)


def is_zero_matrix(matrix: sp.Matrix) -> bool:
    """Return true when every entry simplifies to zero."""

    return all(sp.simplify(entry) == 0 for entry in matrix)


def commutes_with_all(matrix: sp.Matrix, operators: Iterable[sp.Matrix]) -> bool:
    """Return true when ``matrix`` commutes with every operator."""

    return all(is_zero_matrix(commutator(matrix, op)) for op in operators)


def is_s3_invariant(matrix: sp.Matrix) -> bool:
    """Return true when ``matrix`` commutes with full residual ``S_3``."""

    return commutes_with_all(matrix, s3_permutation_matrices())


def is_selected_s2_invariant(matrix: sp.Matrix) -> bool:
    """Return true when ``matrix`` commutes with the selected-port ``S_2``."""

    return commutes_with_all(matrix, selected_port_s2_matrices())


def s3_centralizer_template(alpha: sp.Expr, beta: sp.Expr) -> sp.Matrix:
    """Return the full residual ``S_3`` centralizer template.

    On the natural three-dimensional permutation representation,
    ``3 = 1 + 2``.  Hence every full ``S_3``-invariant operator has the
    form ``alpha P_u + beta(P_a + P_b)``.
    """

    projectors = residual_projectors()
    return sp.simplify(alpha * projectors["u"] + beta * (projectors["a"] + projectors["b"]))
