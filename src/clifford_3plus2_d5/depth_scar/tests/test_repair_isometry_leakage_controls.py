"""V10 leakage controls for weighted repair paths."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_isometry import (
    symmetric_leakage_preserves_ratio_but_rescales,
    unequal_leakage_breaks_ratio,
)
from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    weighted_path_laplacian,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import EXPECTED_LAPLACIAN_SPECTRUM


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_symmetric_leakage_preserves_ratio_but_rescales_depths() -> None:
    leakage = sp.Rational(1, 2)
    weight = 1 - leakage**2

    assert _spectrum_tuple(weighted_path_laplacian(weight, weight)) == (
        sp.Integer(0),
        weight,
        3 * weight,
    )
    assert _spectrum_tuple(weighted_path_laplacian(weight, weight)) != EXPECTED_LAPLACIAN_SPECTRUM
    assert symmetric_leakage_preserves_ratio_but_rescales()


def test_unequal_leakage_breaks_one_to_three_ratio() -> None:
    spectrum = _spectrum_tuple(weighted_path_laplacian(sp.Rational(3, 4), sp.Rational(1, 2)))

    assert sp.simplify(spectrum[2] - 3 * spectrum[1]) != 0
    assert unequal_leakage_breaks_ratio()
