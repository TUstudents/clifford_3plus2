"""Cubic-harmonic decomposition for degree-2 momentum polynomials.

Scope: degree-2 polynomials in ``(k_x, k_y, k_z)``.  The 6-dimensional space
of degree-2 monomials decomposes under the cubic point group ``O_h`` as

    span(k_x², k_y², k_z², k_y k_z, k_z k_x, k_x k_y)
        =  A_{1g}  ⊕  E_g  ⊕  T_{2g}
            (1)        (2)       (3)

with explicit basis decomposition:

- ``A_{1g}`` (1-dim): ``k² = k_x² + k_y² + k_z²``.
- ``E_g`` (2-dim): traceless diagonal, spanned by
  ``{2 k_z² - k_x² - k_y²,  k_x² - k_y²}``.
- ``T_{2g}`` (3-dim): off-diagonal symmetric, spanned by
  ``{k_y k_z,  k_z k_x,  k_x k_y}``.

The three projectors are constructed directly from this decomposition
rather than via the 48-element character formula — they are mathematically
equivalent and avoid hand-coding ``O_h`` group elements.

Monomial basis ordering (fixed):

    m_0 = k_x²
    m_1 = k_y²
    m_2 = k_z²
    m_3 = k_y k_z
    m_4 = k_z k_x
    m_5 = k_x k_y

A 6-dim coefficient vector ``c = (c_0, ..., c_5)`` represents the
polynomial ``Σ c_i m_i``.

Public API:

- ``monomial_basis()`` — the 6 monomial polynomials in fixed order.
- ``polynomial_to_coefficient_vector(polynomial, k_symbols)`` — extract.
- ``coefficient_vector_to_polynomial(vector, k_symbols)`` — reconstruct.
- ``projector_A1g()``, ``projector_Eg()``, ``projector_T2g()`` — the
  three 6×6 SymPy rational matrices.
- ``project_polynomial(polynomial, irrep, k_symbols)`` — convenience.
- ``decompose_matrix_of_polynomials(matrix_of_polys, k_symbols)`` —
  apply elementwise to a matrix whose entries are degree-≤2 polynomials.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import sympy as sp


IrrepName = Literal["A1g", "Eg", "T2g"]
IRREP_NAMES: tuple[IrrepName, ...] = ("A1g", "Eg", "T2g")


def monomial_basis(
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> tuple[sp.Expr, ...]:
    """Return the 6 degree-2 monomials in fixed order."""

    kx, ky, kz = k_symbols
    return (kx**2, ky**2, kz**2, ky * kz, kz * kx, kx * ky)


@lru_cache(maxsize=1)
def projector_A1g() -> sp.Matrix:
    """6×6 projector onto the ``A_{1g}`` irrep (the trace ``k²``)."""

    third = sp.Rational(1, 3)
    matrix = sp.zeros(6, 6)
    for row in range(3):
        for col in range(3):
            matrix[row, col] = third
    return matrix


@lru_cache(maxsize=1)
def projector_Eg() -> sp.Matrix:
    """6×6 projector onto the ``E_g`` irrep (traceless symmetric diagonal)."""

    diag = sp.Rational(2, 3)
    off = sp.Rational(-1, 3)
    matrix = sp.zeros(6, 6)
    for row in range(3):
        for col in range(3):
            matrix[row, col] = diag if row == col else off
    return matrix


@lru_cache(maxsize=1)
def projector_T2g() -> sp.Matrix:
    """6×6 projector onto the ``T_{2g}`` irrep (off-diagonal symmetric)."""

    matrix = sp.zeros(6, 6)
    for index in range(3, 6):
        matrix[index, index] = sp.Integer(1)
    return matrix


def projector(irrep: IrrepName) -> sp.Matrix:
    if irrep == "A1g":
        return projector_A1g()
    if irrep == "Eg":
        return projector_Eg()
    if irrep == "T2g":
        return projector_T2g()
    raise ValueError(f"unknown irrep: {irrep}")


def polynomial_to_coefficient_vector(
    polynomial: sp.Expr,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Matrix:
    """Return the 6×1 coefficient column of a degree-2 polynomial."""

    expanded = sp.expand(polynomial)
    basis = monomial_basis(k_symbols)
    vector = sp.zeros(6, 1)
    for index, monomial in enumerate(basis):
        if index < 3:
            # k_i² monomial; extract coefficient
            k = k_symbols[index]
            other = [s for s in k_symbols if s is not k]
            value = expanded.coeff(k, 2)
            for s in other:
                value = value.coeff(s, 0)
            vector[index, 0] = sp.simplify(value)
        else:
            # k_i k_j cross-term
            tuple_indices = ((1, 2), (2, 0), (0, 1))[index - 3]
            ki = k_symbols[tuple_indices[0]]
            kj = k_symbols[tuple_indices[1]]
            other = [s for s in k_symbols if s is not ki and s is not kj]
            value = expanded.coeff(ki, 1).coeff(kj, 1)
            for s in other:
                value = value.coeff(s, 0)
            vector[index, 0] = sp.simplify(value)
    return vector


def coefficient_vector_to_polynomial(
    vector: sp.Matrix,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Expr:
    """Inverse of ``polynomial_to_coefficient_vector``."""

    basis = monomial_basis(k_symbols)
    total = sp.Integer(0)
    for index, monomial in enumerate(basis):
        total = total + vector[index, 0] * monomial
    return sp.expand(total)


def project_polynomial(
    polynomial: sp.Expr,
    irrep: IrrepName,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Expr:
    vector = polynomial_to_coefficient_vector(polynomial, k_symbols)
    projected = projector(irrep) * vector
    return coefficient_vector_to_polynomial(projected.applyfunc(sp.simplify), k_symbols)


def decompose_matrix_of_polynomials(
    matrix: sp.Matrix,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> dict[IrrepName, sp.Matrix]:
    """Apply each O_h projector elementwise to a matrix of polynomials.

    Returns a dict ``{irrep_name: matrix_of_projected_polynomials}``.
    """

    result: dict[IrrepName, sp.Matrix] = {}
    for irrep in IRREP_NAMES:
        projected = sp.zeros(matrix.rows, matrix.cols)
        for row in range(matrix.rows):
            for col in range(matrix.cols):
                projected[row, col] = project_polynomial(
                    matrix[row, col],
                    irrep,
                    k_symbols,
                )
        result[irrep] = projected.applyfunc(sp.simplify)
    return result


def projectors_satisfy_idempotent_orthogonal_complete() -> bool:
    """Sanity check: P_i² = P_i, P_i · P_j = 0 for i ≠ j, Σ P_i = I_6."""

    p_a = projector_A1g()
    p_e = projector_Eg()
    p_t = projector_T2g()
    identity = sp.eye(6)

    if (p_a * p_a - p_a).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if (p_e * p_e - p_e).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if (p_t * p_t - p_t).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if (p_a * p_e).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if (p_a * p_t).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if (p_e * p_t).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    if ((p_a + p_e + p_t) - identity).applyfunc(sp.simplify) != sp.zeros(6, 6):
        return False
    return True
