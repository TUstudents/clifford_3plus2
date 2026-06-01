"""Negative controls for the V5 nilpotent-flag scar origin."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    cyclic_closure_control_rejected,
    flag_laplacian_from_nilpotent,
    nonnegative_weighted_flag_target_solution,
    rank_one_nilpotent_control_operator,
    rank_one_nilpotent_control_rejected,
    weighted_flag_target_conditions_pass,
    weighted_flag_target_solutions,
    weighted_nilpotent_flag_operator,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import EXPECTED_LAPLACIAN_SPECTRUM


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_rank_one_nilpotent_control_misses_path_spectrum() -> None:
    control = rank_one_nilpotent_control_operator()
    laplacian = flag_laplacian_from_nilpotent(control)

    assert control**2 == sp.zeros(3, 3)
    assert _spectrum_tuple(laplacian) == (0, 0, 2)
    assert _spectrum_tuple(laplacian) != EXPECTED_LAPLACIAN_SPECTRUM
    assert rank_one_nilpotent_control_rejected()


def test_cyclic_closure_control_is_unbroken_k3_not_path_scar() -> None:
    assert cyclic_closure_control_rejected()


def test_weighted_target_conditions_force_unit_adjacent_steps_up_to_sign() -> None:
    assert set(weighted_flag_target_solutions()) == {
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    }
    assert nonnegative_weighted_flag_target_solution() == (1, 1)
    assert weighted_flag_target_conditions_pass()


def test_unequal_weighted_flag_control_misses_target_spectrum() -> None:
    laplacian = flag_laplacian_from_nilpotent(
        weighted_nilpotent_flag_operator(sp.Integer(2), sp.Integer(1))
    )

    assert _spectrum_tuple(laplacian) != EXPECTED_LAPLACIAN_SPECTRUM
