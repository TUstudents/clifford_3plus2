"""Continuum-expansion utilities for spacetime QCA Bloch symbols."""

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
    convention: Literal["generator", "complex_hamiltonian"] = "generator",
) -> sp.Matrix:
    """Extract the leading generator from ``U = I + epsilon K + O(epsilon^2)``.

    With ``convention="complex_hamiltonian"``, the input is read as
    ``U = I - i epsilon H + O(epsilon^2)`` and the Hermitian Hamiltonian
    coefficient is returned.
    """

    coefficient = first_order_in_epsilon(floquet, epsilon)
    if convention == "generator":
        return coefficient
    if convention == "complex_hamiltonian":
        return (coefficient / (-sp.I)).applyfunc(sp.simplify)
    raise ValueError(f"unknown convention: {convention}")


def hamiltonian_from_generator(generator: sp.Matrix) -> sp.Matrix:
    """Convert skew generator ``K`` to Hamiltonian convention ``H = iK``."""

    return (sp.I * generator).applyfunc(sp.simplify)
