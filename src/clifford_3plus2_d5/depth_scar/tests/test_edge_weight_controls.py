"""Controls for the V3 edge-weight scar potential."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    defect_term_is_load_bearing,
    non_scar_zero_of_potential_without_defect_term,
    path_scar_weight_vectors,
    scar_potential,
    scar_potential_without_defect_term,
    symmetric_point_weights,
    symmetric_triangle_control_rejected,
    weighted_triangle_laplacian_general,
)


def test_s3_term_is_load_bearing_against_non_scar_zero() -> None:
    weights = non_scar_zero_of_potential_without_defect_term()
    assert weights not in set(path_scar_weight_vectors())
    assert sp.simplify(scar_potential_without_defect_term(weights)) == 0
    assert sp.simplify(scar_potential(weights)) == sp.Rational(4, 27)
    assert defect_term_is_load_bearing()


def test_unbroken_symmetric_triangle_control_is_rejected() -> None:
    weights = symmetric_point_weights()
    assert sp.simplify(scar_potential(weights)) > 0
    assert weighted_triangle_laplacian_general(weights).eigenvals() == {
        sp.Integer(0): 1,
        sp.Integer(3): 2,
    }
    assert symmetric_triangle_control_rejected()

