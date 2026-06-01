"""Tests for local partial-isometry normalization of the repair flag."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    complex_local_flag_operator,
    final_projection,
    initial_projection,
    nonnegative_partial_isometry_solution,
    partial_isometry_forces_unit_magnitudes,
    partial_isometry_magnitude_solutions,
    partial_isometry_residuals,
    projection_formula_pass,
)


def test_generic_complex_flag_has_diagonal_initial_and_final_projections() -> None:
    r, s, alpha, beta = sp.symbols("r s alpha beta", real=True)
    operator = complex_local_flag_operator(r, s, alpha, beta)

    assert sp.simplify(initial_projection(operator) - sp.diag(0, r**2, s**2)) == sp.zeros(3, 3)
    assert sp.simplify(final_projection(operator) - sp.diag(r**2, s**2, 0)) == sp.zeros(3, 3)
    assert projection_formula_pass()


def test_partial_isometry_idempotence_forces_unit_nonzero_magnitudes() -> None:
    r, s = sp.symbols("r s", real=True)

    assert partial_isometry_residuals(r, s) == (r**4 - r**2, s**4 - s**2)
    assert set(partial_isometry_magnitude_solutions()) == {
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    }
    assert nonnegative_partial_isometry_solution() == (1, 1)
    assert partial_isometry_forces_unit_magnitudes()
