"""Phase SC-1: degree-3 cubic-harmonic decomposition under O_h.

Extends cp/cubic_harmonics.py from degree 2 to degree 3.  The 10-
dimensional space of degree-3 monomials in (k_x, k_y, k_z)
decomposes under the cubic point group O_h as

    A_{2u}  ⊕  T_{1u}  ⊕  T_{2u}
     (1)        (6)        (3)        = 10  ✓

with:

- ``A_{2u}`` (1-dim): basis ``k_x k_y k_z`` — the totally-antisymmetric
  pseudo-scalar.  This is the θ_QCD-term momentum-shape direction;
  the load-bearing irrep for the Strong-CP audit.
- ``T_{2u}`` (3-dim): basis ``{k_x (k_y² − k_z²), k_y (k_z² − k_x²),
  k_z (k_x² − k_y²)}`` — vector × traceless E_g pair.
- ``T_{1u}`` (6-dim, two copies): one copy carries the "radial"
  L=1 piece ``k_i · (k_x² + k_y² + k_z²)``; the other carries the
  L=3 "pure cubic" piece.  For the audit's purposes we treat
  T_{1u} as a single 6-dim block (the two copies don't separate
  invariantly under O_h).

The five degree-3-absent irreps (A_{1g}, A_{2g}, E_g, T_{1g}, T_{2g},
A_{1u}, E_u) have zero projectors at degree 3; for API symmetry
we expose them returning 10×10 zeros (with a "not present at
degree 3" docstring note).

Monomial basis ordering (fixed):

    m_0 = k_x³
    m_1 = k_y³
    m_2 = k_z³
    m_3 = k_x² k_y
    m_4 = k_x² k_z
    m_5 = k_y² k_x
    m_6 = k_y² k_z
    m_7 = k_z² k_x
    m_8 = k_z² k_y
    m_9 = k_x k_y k_z
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import sympy as sp


Degree3IrrepName = Literal["A2u", "T2u", "T1u"]
DEGREE3_IRREP_NAMES: tuple[Degree3IrrepName, ...] = ("A2u", "T2u", "T1u")


def degree3_monomial_basis(
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> tuple[sp.Expr, ...]:
    """Return the 10 degree-3 monomials in fixed order (see module docstring)."""

    kx, ky, kz = k_symbols
    return (
        kx**3,
        ky**3,
        kz**3,
        kx**2 * ky,
        kx**2 * kz,
        ky**2 * kx,
        ky**2 * kz,
        kz**2 * kx,
        kz**2 * ky,
        kx * ky * kz,
    )


@lru_cache(maxsize=1)
def projector_A2u_deg3() -> sp.Matrix:
    """10×10 projector onto the A_{2u} irrep at degree 3.

    A_{2u} is 1-dim with basis ``k_x k_y k_z`` (= m_9).  The projector
    is rank-1: only the (9, 9) entry is non-zero.
    """

    matrix = sp.zeros(10, 10)
    matrix[9, 9] = sp.Integer(1)
    return matrix


@lru_cache(maxsize=1)
def projector_T2u_deg3() -> sp.Matrix:
    """10×10 projector onto the T_{2u} irrep at degree 3.

    T_{2u} is 3-dim with orthogonal basis:

        t_x = (m_5 - m_7) / sqrt(2)   ∝ k_x (k_y² - k_z²)
        t_y = (m_8 - m_3) / sqrt(2)   ∝ k_y (k_z² - k_x²)
        t_z = (m_4 - m_6) / sqrt(2)   ∝ k_z (k_x² - k_y²)

    The projector is the sum of three rank-1 outer products.  In the
    monomial basis, each 2×2 block at indices (a, b) for the relevant
    monomial pairs contributes the pattern ``[[1/2, -1/2], [-1/2, 1/2]]``.
    """

    matrix = sp.zeros(10, 10)
    half = sp.Rational(1, 2)
    # t_x at indices 5, 7
    matrix[5, 5] = half
    matrix[5, 7] = -half
    matrix[7, 5] = -half
    matrix[7, 7] = half
    # t_y at indices 8, 3
    matrix[8, 8] = half
    matrix[8, 3] = -half
    matrix[3, 8] = -half
    matrix[3, 3] = half
    # t_z at indices 4, 6
    matrix[4, 4] = half
    matrix[4, 6] = -half
    matrix[6, 4] = -half
    matrix[6, 6] = half
    return matrix


@lru_cache(maxsize=1)
def projector_T1u_deg3() -> sp.Matrix:
    """10×10 projector onto the T_{1u} irrep at degree 3 (two copies, 6-dim).

    Computed as the complement: ``I_{10} - P_{A2u} - P_{T2u}``.

    T_{1u} is 6-dim because the degree-3 polynomial space contains
    two distinct copies of T_{1u} (a "radial" L=1 ``k_i · r²`` copy
    and a "pure cubic" L=3 copy).  The two copies do not separate
    invariantly under O_h alone; for the Strong-CP audit we treat
    T_{1u} as a single 6-dim block (sufficient for the selection
    rule).
    """

    return sp.eye(10) - projector_A2u_deg3() - projector_T2u_deg3()


def _zero_10x10() -> sp.Matrix:
    return sp.zeros(10, 10)


@lru_cache(maxsize=1)
def projector_A1g_deg3() -> sp.Matrix:
    """A_{1g} does not appear at degree 3 (parity selection)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_A2g_deg3() -> sp.Matrix:
    """A_{2g} does not appear at degree 3 (parity selection)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_Eg_deg3() -> sp.Matrix:
    """E_g does not appear at degree 3 (parity selection)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_T1g_deg3() -> sp.Matrix:
    """T_{1g} does not appear at degree 3 (parity selection)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_T2g_deg3() -> sp.Matrix:
    """T_{2g} does not appear at degree 3 (parity selection)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_A1u_deg3() -> sp.Matrix:
    """A_{1u} does not appear at degree 3 (no totally-symmetric pseudo-scalar)."""

    return _zero_10x10()


@lru_cache(maxsize=1)
def projector_Eu_deg3() -> sp.Matrix:
    """E_u does not appear at degree 3 (appears first at degree ≥ 5)."""

    return _zero_10x10()


def projector_deg3(irrep: Degree3IrrepName) -> sp.Matrix:
    """Return the degree-3 projector for one of {A2u, T2u, T1u}."""

    if irrep == "A2u":
        return projector_A2u_deg3()
    if irrep == "T2u":
        return projector_T2u_deg3()
    if irrep == "T1u":
        return projector_T1u_deg3()
    raise ValueError(f"unknown degree-3 irrep: {irrep}")


def polynomial_to_coefficient_vector_deg3(
    polynomial: sp.Expr,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Matrix:
    """Return the 10×1 coefficient column of a degree-3 polynomial.

    Extracts the coefficient of each of the 10 degree-3 monomials
    in fixed order (see module docstring).  Lower-degree terms are
    ignored; higher-degree terms cause a ValueError.
    """

    kx, ky, kz = k_symbols
    expanded = sp.expand(polynomial)
    basis = degree3_monomial_basis(k_symbols)
    vector = sp.zeros(10, 1)

    def _coeff(expr: sp.Expr, exps: tuple[int, int, int]) -> sp.Expr:
        ex, ey, ez = exps
        return sp.expand(expr.coeff(kx, ex).coeff(ky, ey).coeff(kz, ez))

    exponent_tuples: tuple[tuple[int, int, int], ...] = (
        (3, 0, 0),  # m_0
        (0, 3, 0),  # m_1
        (0, 0, 3),  # m_2
        (2, 1, 0),  # m_3
        (2, 0, 1),  # m_4
        (1, 2, 0),  # m_5
        (0, 2, 1),  # m_6
        (1, 0, 2),  # m_7
        (0, 1, 2),  # m_8
        (1, 1, 1),  # m_9
    )

    for index, exps in enumerate(exponent_tuples):
        vector[index, 0] = _coeff(expanded, exps)

    return vector


def coefficient_vector_to_polynomial_deg3(
    vector: sp.Matrix,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Expr:
    """Inverse of ``polynomial_to_coefficient_vector_deg3``."""

    basis = degree3_monomial_basis(k_symbols)
    total: sp.Expr = sp.Integer(0)
    for index, monomial in enumerate(basis):
        total = total + vector[index, 0] * monomial
    return sp.expand(total)


def project_polynomial_deg3(
    polynomial: sp.Expr,
    irrep: Degree3IrrepName,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> sp.Expr:
    """Project a degree-3 polynomial onto the named O_h irrep."""

    vector = polynomial_to_coefficient_vector_deg3(polynomial, k_symbols)
    projected = projector_deg3(irrep) * vector
    return coefficient_vector_to_polynomial_deg3(
        projected.applyfunc(sp.simplify), k_symbols
    )


def decompose_degree3_polynomial(
    polynomial: sp.Expr,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> dict[Degree3IrrepName, sp.Expr]:
    """Return ``{irrep_name: projected polynomial}`` for the three non-empty irreps."""

    return {
        irrep: project_polynomial_deg3(polynomial, irrep, k_symbols)
        for irrep in DEGREE3_IRREP_NAMES
    }


def decompose_degree3_matrix_of_polynomials(
    matrix: sp.Matrix,
    k_symbols: tuple[sp.Symbol, sp.Symbol, sp.Symbol],
) -> dict[Degree3IrrepName, sp.Matrix]:
    """Elementwise decomposition of a matrix of degree-3 polynomials."""

    result: dict[Degree3IrrepName, sp.Matrix] = {}
    for irrep in DEGREE3_IRREP_NAMES:
        projected = sp.zeros(matrix.rows, matrix.cols)
        for row in range(matrix.rows):
            for col in range(matrix.cols):
                projected[row, col] = project_polynomial_deg3(
                    matrix[row, col], irrep, k_symbols
                )
        result[irrep] = projected.applyfunc(sp.simplify)
    return result


def deg3_projectors_satisfy_idempotent_orthogonal_complete() -> bool:
    """Sanity check: P_i² = P_i, P_i · P_j = 0 (i ≠ j), Σ P_i = I_{10}."""

    p_a = projector_A2u_deg3()
    p_t2 = projector_T2u_deg3()
    p_t1 = projector_T1u_deg3()
    identity = sp.eye(10)

    for proj in (p_a, p_t2, p_t1):
        if (proj * proj - proj).applyfunc(sp.simplify) != sp.zeros(10, 10):
            return False
    pairs = ((p_a, p_t2), (p_a, p_t1), (p_t2, p_t1))
    for left, right in pairs:
        if (left * right).applyfunc(sp.simplify) != sp.zeros(10, 10):
            return False
    if ((p_a + p_t2 + p_t1) - identity).applyfunc(sp.simplify) != sp.zeros(10, 10):
        return False
    return True
