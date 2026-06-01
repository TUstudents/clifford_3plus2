"""Tests for the exact zero set of the V3 scar potential."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    path_scar_minima_pass,
    path_scar_weight_vectors,
    potential_zero_set_solutions,
    scar_potential,
    symmetric_edge_invariants,
    symmetric_point_weights,
)


def test_path_scars_are_exact_zeroes() -> None:
    for weights in path_scar_weight_vectors():
        assert sp.simplify(scar_potential(weights)) == 0


def test_zero_set_solutions_are_exactly_three_path_scars() -> None:
    solutions = potential_zero_set_solutions()
    assert set(solutions) == set(path_scar_weight_vectors())
    assert len(solutions) == 3
    assert path_scar_minima_pass()


def test_zero_set_logic_has_s1_2_s2_1_s3_0_at_each_scar() -> None:
    for weights in potential_zero_set_solutions():
        assert symmetric_edge_invariants(weights) == (sp.Integer(2), sp.Integer(1), sp.Integer(0))


def test_symmetric_triangle_is_not_a_minimum() -> None:
    assert sp.simplify(scar_potential(symmetric_point_weights())) == 6

