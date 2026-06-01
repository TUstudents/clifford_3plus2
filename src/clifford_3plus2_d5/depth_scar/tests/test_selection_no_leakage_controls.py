"""V11 tests for consequences and approximate leakage bounds."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_isometry import repair_isometry_saturation_pass
from clifford_3plus2_d5.depth_scar.selection_no_leakage import (
    approximate_leakage_weight_bounds,
    small_leakage_bounds_weights,
    unique_successor_no_leakage_pass,
)


def test_unique_successor_no_leakage_theorem_passes() -> None:
    assert repair_isometry_saturation_pass()
    assert unique_successor_no_leakage_pass()


def test_small_leakage_bounds_active_repair_weights() -> None:
    eta = sp.symbols("eta", nonnegative=True)

    assert approximate_leakage_weight_bounds(eta) == (1 - eta**2, 1)
    assert small_leakage_bounds_weights()
