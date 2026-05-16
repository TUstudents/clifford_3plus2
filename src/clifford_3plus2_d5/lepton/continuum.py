"""Symbolic continuum-limit utilities for lepton QCA walks.

Conventions used by Session 16:

* ``K`` is the infinitesimal QCA generator in ``U = I + epsilon K + O(epsilon^2)``.
* ``H = i K`` is the Hermitian Bloch Hamiltonian used for Dirac comparison.
* In complex Hamiltonian convention, ``U = I - i epsilon H + O(epsilon^2)``.

For a background real-skew ``su(3)`` link generator ``A`` and dimensionful
continuum momentum ``k``, the right-moving block is

``U_R(k, epsilon) = exp(-i k epsilon) exp(epsilon A)``.

To first order,

``K_R = A - i k I`` and ``H_R = i A + k I``.
"""

from __future__ import annotations

from typing import Literal

import sympy as sp


def matrix_zero(matrix: sp.Matrix) -> bool:
    return matrix.applyfunc(sp.simplify) == sp.zeros(matrix.rows, matrix.cols)


def first_order_in_epsilon(matrix: sp.Matrix, epsilon: sp.Symbol) -> sp.Matrix:
    """Return the coefficient of ``epsilon`` in a matrix expansion."""

    expanded = matrix.applyfunc(
        lambda value: sp.series(value, epsilon, 0, 2).removeO(),
    )
    return expanded.applyfunc(lambda value: sp.expand(value).coeff(epsilon, 1)).applyfunc(sp.simplify)


def effective_generator_from_floquet(
    floquet: sp.Matrix,
    *,
    epsilon: sp.Symbol,
    convention: Literal["real_skew", "complex_hamiltonian"] = "real_skew",
) -> sp.Matrix:
    """Extract the leading generator from a parameterized Floquet matrix."""

    coefficient = first_order_in_epsilon(floquet, epsilon)
    if convention == "real_skew":
        return coefficient
    if convention == "complex_hamiltonian":
        return (coefficient / (-sp.I)).applyfunc(sp.simplify)
    raise ValueError(f"unknown convention: {convention}")


def hamiltonian_from_real_skew_generator(generator: sp.Matrix) -> sp.Matrix:
    """Convert the QCA generator ``K`` to the Bloch Hamiltonian ``H = iK``."""

    return (sp.I * generator).applyfunc(sp.simplify)
