"""Tests for path phase removal and healed-loop holonomy."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.loop_healing import (
    gauge_transformed_edge_phase,
    healed_triangle_cycle_rank,
    loop_phase_is_gauge_invariant,
    path_cycle_rank,
    path_phase_gauge_solution,
    path_phase_removal_residuals,
    path_phases_are_removable,
    triangle_loop_phase,
    triangle_loop_phase_after_gauge,
)


def test_path_phases_are_gauge_removable_because_tree_has_no_cycle() -> None:
    theta_ua, theta_ab = sp.symbols("theta_ua theta_ab")
    alpha_u, alpha_a, alpha_b = path_phase_gauge_solution(theta_ua, theta_ab)

    assert (alpha_u, alpha_a, alpha_b) == (0, theta_ua, theta_ua + theta_ab)
    assert path_phase_removal_residuals(theta_ua, theta_ab) == (0, 0)
    assert path_phases_are_removable()
    assert path_cycle_rank() == 0


def test_healed_triangle_has_one_gauge_invariant_loop_phase() -> None:
    theta_ua, theta_ab, theta_bu = sp.symbols("theta_ua theta_ab theta_bu")
    alpha_u, alpha_a, alpha_b = sp.symbols("alpha_u alpha_a alpha_b")

    before = triangle_loop_phase(theta_ua, theta_ab, theta_bu)
    after = triangle_loop_phase_after_gauge(
        theta_ua, theta_ab, theta_bu, alpha_u, alpha_a, alpha_b
    )

    assert healed_triangle_cycle_rank() == 1
    assert sp.simplify(after - before) == 0
    assert loop_phase_is_gauge_invariant()


def test_edge_phase_transform_rule_preserves_loop_sum() -> None:
    theta_ua, theta_ab, theta_bu = sp.symbols("theta_ua theta_ab theta_bu")
    alpha_u, alpha_a, alpha_b = sp.symbols("alpha_u alpha_a alpha_b")

    transformed_sum = sp.simplify(
        gauge_transformed_edge_phase(theta_ua, alpha_u, alpha_a)
        + gauge_transformed_edge_phase(theta_ab, alpha_a, alpha_b)
        + gauge_transformed_edge_phase(theta_bu, alpha_b, alpha_u)
    )

    assert sp.simplify(transformed_sum - (theta_ua + theta_ab + theta_bu)) == 0
