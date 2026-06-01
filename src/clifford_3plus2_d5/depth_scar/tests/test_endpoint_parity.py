"""Tests for the surviving endpoint parity selection rule."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.prediction_ledger import (
    endpoint_parity_blocks_even_odd,
    endpoint_reflection_matrix,
    generic_endpoint_symmetric_operator,
    mode_endpoint_parities,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import defect_modes


def test_endpoint_reflection_is_an_involution() -> None:
    reflection = endpoint_reflection_matrix()
    assert reflection * reflection == sp.eye(3)


def test_depth_modes_have_endpoint_parities_even_odd_even() -> None:
    assert mode_endpoint_parities() == (1, -1, 1)


def test_endpoint_symmetric_operators_block_even_odd_mixing() -> None:
    operator = generic_endpoint_symmetric_operator()
    reflection = endpoint_reflection_matrix()
    mode0, mode2, mode6 = defect_modes()

    assert sp.simplify(reflection * operator - operator * reflection) == sp.zeros(3, 3)
    assert sp.simplify((mode0.T * operator * mode2)[0]) == 0
    assert sp.simplify((mode6.T * operator * mode2)[0]) == 0
    assert endpoint_parity_blocks_even_odd()

