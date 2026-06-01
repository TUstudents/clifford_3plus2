"""V9 tests separating support locality from normalization saturation."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    partial_isometry_forces_unit_magnitudes,
)
from clifford_3plus2_d5.depth_scar.microscopic_locality import (
    generic_local_repair_nilpotent_pass,
    generic_local_repair_operator,
    partial_isometry_saturation_conditional_pass,
    target_spectrum_forces_unit_weights,
    weighted_path_laplacian,
    weighted_path_spectrum_formula,
    weighted_path_spectrum_formula_pass,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_generic_local_repair_is_length_three_nilpotent_when_both_links_active() -> None:
    alpha, beta = sp.symbols("alpha beta", nonzero=True)
    operator = generic_local_repair_operator(alpha, beta)

    assert operator.rank() == 2
    assert operator**2 == sp.Matrix([[0, 0, alpha * beta], [0, 0, 0], [0, 0, 0]])
    assert operator**3 == sp.zeros(3, 3)
    assert generic_local_repair_nilpotent_pass()


def test_weighted_path_formula_matches_exact_spectrum() -> None:
    w1, w2 = sp.symbols("w1 w2", positive=True)

    assert _spectrum_tuple(weighted_path_laplacian(w1, w2)) == weighted_path_spectrum_formula(
        w1,
        w2,
    )
    assert weighted_path_spectrum_formula_pass()


def test_target_spectrum_requires_unit_weights() -> None:
    assert _spectrum_tuple(weighted_path_laplacian(sp.Integer(1), sp.Integer(1))) == (
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(3),
    )
    assert target_spectrum_forces_unit_weights()


def test_partial_isometry_saturation_supplies_conditional_normalization() -> None:
    assert partial_isometry_forces_unit_magnitudes()
    assert partial_isometry_saturation_conditional_pass()
