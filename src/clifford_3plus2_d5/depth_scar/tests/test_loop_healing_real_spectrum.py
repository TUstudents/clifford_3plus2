"""Tests for the real loop-healing spectrum deformation."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.loop_healing import (
    real_healed_depth_formula_pass,
    real_healed_depth_spectrum_formula,
    real_healed_laplacian,
    real_healed_laplacian_spectrum_formula,
    real_healed_spectrum_formula_pass,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    k3_laplacian,
    path_laplacian,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_real_loop_healing_has_exact_spectrum_formula() -> None:
    delta = sp.symbols("delta")
    assert real_healed_laplacian_spectrum_formula(delta) == (0, 1 + 2 * delta, 3)
    assert real_healed_depth_spectrum_formula(delta) == (0, 2 + 4 * delta, 6)
    assert real_healed_spectrum_formula_pass()
    assert real_healed_depth_formula_pass()


def test_real_loop_healing_interpolates_path_to_unbroken_triangle() -> None:
    path_limit = real_healed_laplacian(sp.Integer(0))
    unbroken_limit = real_healed_laplacian(sp.Integer(1))

    assert path_limit == path_laplacian()
    assert unbroken_limit == k3_laplacian()
    assert _spectrum_tuple(path_limit) == EXPECTED_LAPLACIAN_SPECTRUM
    assert _spectrum_tuple(2 * path_limit) == EXPECTED_DEPTH_SPECTRUM
    assert _spectrum_tuple(unbroken_limit) == (0, 3, 3)
    assert _spectrum_tuple(2 * unbroken_limit) == (0, 6, 6)
