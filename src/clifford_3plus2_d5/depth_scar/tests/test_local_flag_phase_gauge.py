"""Tests for removing tree phases from the local repair flag."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    complex_local_flag_operator,
    flag_phase_gauge_solution,
    flag_phase_removal_residuals,
    flag_phases_are_removable,
    phased_unit_flag_gauge_equivalent_to_canonical,
    rephase_operator,
)
from clifford_3plus2_d5.depth_scar.nilpotent_flag import nilpotent_flag_operator


def test_flag_phase_gauge_solution_removes_both_tree_phases() -> None:
    alpha, beta = sp.symbols("alpha beta")

    assert flag_phase_gauge_solution(alpha, beta) == (0, alpha, alpha + beta)
    assert flag_phase_removal_residuals(alpha, beta) == (0, 0)
    assert flag_phases_are_removable()


def test_unit_magnitude_phased_flag_is_gauge_equivalent_to_canonical() -> None:
    alpha, beta = sp.symbols("alpha beta", real=True)
    operator = complex_local_flag_operator(sp.Integer(1), sp.Integer(1), alpha, beta)
    rephased = rephase_operator(operator, *flag_phase_gauge_solution(alpha, beta))

    assert sp.simplify(rephased - nilpotent_flag_operator()) == sp.zeros(3, 3)
    assert phased_unit_flag_gauge_equivalent_to_canonical()
