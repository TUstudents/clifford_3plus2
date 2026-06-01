"""Tests for the weighted K3 scar variant."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_LAPLACIAN_SPECTRUM,
    permuted_path_laplacians,
    spectrum_dict,
    weighted_scar_controls_pass,
    weighted_triangle_eigenvalue_formula,
    weighted_triangle_laplacian,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_weighted_triangle_spectrum_formula() -> None:
    x, y = sp.symbols("x y")
    eigenvalues = set(weighted_triangle_eigenvalue_formula())
    assert eigenvalues == {sp.Integer(0), 3 * x, x + 2 * y}


def test_path_limit_and_strong_bond_scar_give_0_1_3() -> None:
    path_limit = weighted_triangle_laplacian(sp.Integer(1), sp.Integer(0))
    strong_bond = weighted_triangle_laplacian(sp.Rational(1, 3), sp.Rational(4, 3))
    symmetric = weighted_triangle_laplacian(sp.Integer(1), sp.Integer(1))

    assert _spectrum_tuple(path_limit) == EXPECTED_LAPLACIAN_SPECTRUM
    assert _spectrum_tuple(strong_bond) == EXPECTED_LAPLACIAN_SPECTRUM
    assert spectrum_dict(symmetric) == {sp.Integer(0): 1, sp.Integer(3): 2}
    assert weighted_scar_controls_pass()


def test_all_three_missing_edge_scars_are_spectrum_equivalent() -> None:
    for laplacian in permuted_path_laplacians():
        assert _spectrum_tuple(laplacian) == EXPECTED_LAPLACIAN_SPECTRUM

