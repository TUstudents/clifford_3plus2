"""Tests for ``cubic_harmonics.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.cp.cubic_harmonics import (
    IRREP_NAMES,
    coefficient_vector_to_polynomial,
    decompose_matrix_of_polynomials,
    monomial_basis,
    polynomial_to_coefficient_vector,
    project_polynomial,
    projector,
    projector_A1g,
    projector_Eg,
    projector_T2g,
    projectors_satisfy_idempotent_orthogonal_complete,
)


def _k_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    return kx, ky, kz


def test_projectors_satisfy_algebra() -> None:
    assert projectors_satisfy_idempotent_orthogonal_complete()


def test_projector_traces_match_irrep_dimensions() -> None:
    assert sp.trace(projector_A1g()) == 1
    assert sp.trace(projector_Eg()) == 2
    assert sp.trace(projector_T2g()) == 3


def test_monomial_round_trip_for_k_squared() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = kx**2 + ky**2 + kz**2  # pure A_{1g}
    vector = polynomial_to_coefficient_vector(polynomial, k)
    reconstructed = coefficient_vector_to_polynomial(vector, k)
    assert sp.simplify(reconstructed - polynomial) == 0


def test_monomial_round_trip_for_cross_terms() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = sp.Rational(2, 3) * kx * ky - sp.Rational(5, 7) * ky * kz
    vector = polynomial_to_coefficient_vector(polynomial, k)
    reconstructed = coefficient_vector_to_polynomial(vector, k)
    assert sp.simplify(reconstructed - polynomial) == 0


def test_k_squared_projects_to_pure_A1g() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = kx**2 + ky**2 + kz**2
    assert sp.simplify(project_polynomial(polynomial, "A1g", k) - polynomial) == 0
    assert sp.simplify(project_polynomial(polynomial, "Eg", k)) == 0
    assert sp.simplify(project_polynomial(polynomial, "T2g", k)) == 0


def test_traceless_diagonal_projects_to_pure_Eg() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = 2 * kz**2 - kx**2 - ky**2  # pure E_g
    assert sp.simplify(project_polynomial(polynomial, "A1g", k)) == 0
    assert sp.simplify(project_polynomial(polynomial, "Eg", k) - polynomial) == 0
    assert sp.simplify(project_polynomial(polynomial, "T2g", k)) == 0


def test_cross_term_projects_to_pure_T2g() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = kx * ky  # pure T_{2g}
    assert sp.simplify(project_polynomial(polynomial, "A1g", k)) == 0
    assert sp.simplify(project_polynomial(polynomial, "Eg", k)) == 0
    assert sp.simplify(project_polynomial(polynomial, "T2g", k) - polynomial) == 0


def test_arbitrary_degree_two_polynomial_reconstructs() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    polynomial = (
        3 * kx**2
        - 2 * ky**2
        + sp.Rational(1, 5) * kz**2
        + 7 * ky * kz
        - sp.Rational(3, 4) * kz * kx
        + sp.Rational(11, 2) * kx * ky
    )
    parts = {
        irrep: project_polynomial(polynomial, irrep, k)
        for irrep in IRREP_NAMES
    }
    total = parts["A1g"] + parts["Eg"] + parts["T2g"]
    assert sp.simplify(total - polynomial) == 0


def test_matrix_decomposition_reconstructs() -> None:
    k = _k_symbols()
    kx, ky, kz = k
    matrix = sp.Matrix(
        [
            [kx**2 + kz**2, kx * ky],
            [sp.Rational(1, 2) * ky * kz, ky**2 - kz**2],
        ],
    )
    decomposed = decompose_matrix_of_polynomials(matrix, k)
    reconstructed = sum(decomposed.values(), sp.zeros(matrix.rows, matrix.cols))
    assert (reconstructed - matrix).applyfunc(sp.simplify) == sp.zeros(
        matrix.rows,
        matrix.cols,
    )


def test_projector_lookup() -> None:
    assert projector("A1g") == projector_A1g()
    assert projector("Eg") == projector_Eg()
    assert projector("T2g") == projector_T2g()
