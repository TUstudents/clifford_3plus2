"""Tests for rank-one, CP, and mass-exponent consequences."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.prediction_ledger import (
    graph_cycle_rank,
    leading_kernel_is_democratic_rank_one,
    leading_transfer_kernel,
    pure_path_has_no_intrinsic_cp_holonomy,
    restored_triangle_has_one_loop,
    two_sided_lambda_power_semigroup,
    two_sided_mass_depth_semigroup,
)


def test_leading_kernel_is_democratic_rank_one() -> None:
    kernel = leading_transfer_kernel()
    assert kernel == sp.ones(3, 3) / 3
    assert kernel.rank() == 1
    assert leading_kernel_is_democratic_rank_one()


def test_pure_path_has_no_cycle_for_intrinsic_cp_holonomy() -> None:
    assert graph_cycle_rank(((0, 1), (1, 2))) == 0
    assert pure_path_has_no_intrinsic_cp_holonomy()


def test_restored_triangle_has_one_loop_for_possible_holonomy() -> None:
    assert graph_cycle_rank(((0, 1), (1, 2), (0, 2))) == 1
    assert restored_triangle_has_one_loop()


def test_two_sided_mass_exponent_semigroup_is_fixed_but_conditional() -> None:
    assert two_sided_mass_depth_semigroup() == (0, 2, 4, 6, 8, 12)
    assert two_sided_lambda_power_semigroup() == (0, 1, 2, 3, 4, 6)

