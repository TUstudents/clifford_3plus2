"""Tests for V3 edge-weight symmetric invariants."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    repair_weight_symbols,
    scar_potential,
    scar_potential_is_symmetric_under_permutations,
    symmetric_edge_invariants,
)


def test_symmetric_edge_invariants_are_elementary_polynomials() -> None:
    w1, w2, w3 = repair_weight_symbols()
    s1, s2, s3 = symmetric_edge_invariants((w1, w2, w3))

    assert s1 == w1 + w2 + w3
    assert s2 == w1 * w2 + w2 * w3 + w3 * w1
    assert s3 == w1 * w2 * w3


def test_scar_potential_has_expected_form() -> None:
    w1, w2, w3 = repair_weight_symbols()
    s1, s2, s3 = symmetric_edge_invariants((w1, w2, w3))

    assert sp.simplify(
        scar_potential((w1, w2, w3)) - ((s1 - 2) ** 2 + (s2 - 1) ** 2 + s3)
    ) == 0


def test_scar_potential_is_symmetric_under_edge_permutations() -> None:
    assert scar_potential_is_symmetric_under_permutations()
