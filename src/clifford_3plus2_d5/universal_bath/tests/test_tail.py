"""Tests for the universal silver tail."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.tail import (
    period_one_tail,
    silver_epsilon,
    silver_selected_z,
    silver_tail_payload,
    tail_fixed_point_residual,
)


def test_period_one_tail_satisfies_weyl_fixed_point() -> None:
    z = sp.Symbol("z")
    assert tail_fixed_point_residual(z) == 0


def test_tail_at_bb_marginal_point_is_silver() -> None:
    assert sp.simplify(period_one_tail(silver_selected_z()) - silver_epsilon()) == 0


def test_silver_tail_payload_controls_alternate_tail() -> None:
    payload = silver_tail_payload()
    assert payload.fixed_point_residual == 0
    assert payload.selected_value_matches_epsilon
    assert payload.alternate_tail_changes_value

