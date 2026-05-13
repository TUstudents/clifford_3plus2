from __future__ import annotations

import sympy as sp
import pytest

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.search.forced_j import find_gate_word_for_standard_j
from clifford_3plus2_d5.search.gate_words import (
    matrix_product,
    rank_one_pair_rotation,
    rank_one_pair_rotations,
    scan_gate_words,
    standard_period_four_primitives,
)


def test_global_clock_primitive_generates_j_at_depth_one() -> None:
    primitives = standard_period_four_primitives()
    match = find_gate_word_for_standard_j(primitives, max_depth=1)

    assert match is not None
    word, matrix = match
    assert word == ("global_clock_tick",)
    assert matrix == standard_real_carrier().complex_structure


def test_gate_word_scanner_is_deterministic_and_bounded() -> None:
    primitives = standard_period_four_primitives()
    results = list(scan_gate_words(primitives, max_depth=3))

    assert [word for word, _ in results] == [
        ("global_clock_tick",),
        ("global_clock_tick", "global_clock_tick"),
        ("global_clock_tick", "global_clock_tick", "global_clock_tick"),
    ]


def test_period_four_products_have_expected_powers() -> None:
    j = standard_period_four_primitives()[0].matrix

    assert matrix_product([j]) == j
    assert matrix_product([j, j]) == -identity(10)
    assert matrix_product([j, j, j, j]) == identity(10)


def test_rank_one_pair_rotation_is_independently_addressable() -> None:
    gates = rank_one_pair_rotations()

    assert len(gates) == 5
    assert all(gate.independently_addressable for gate in gates)
    assert gates[0].matrix != standard_real_carrier().complex_structure


def test_rank_one_pair_rotation_acts_only_on_one_pair() -> None:
    rotation = rank_one_pair_rotation(0)
    expected = identity(10)
    expected[0, 0] = 0
    expected[0, 5] = -1
    expected[5, 0] = 1
    expected[5, 5] = 0

    assert rotation == expected
    assert rotation.T * rotation == identity(10)


def test_invalid_rank_one_pair_rotation_raises() -> None:
    with pytest.raises(ValueError):
        rank_one_pair_rotation(5)


def test_empty_or_zero_depth_scan_yields_nothing() -> None:
    assert list(scan_gate_words(standard_period_four_primitives(), max_depth=0)) == []
    assert list(scan_gate_words((), max_depth=3)) == []


def test_matrix_product_defaults_to_identity() -> None:
    assert matrix_product([]) == sp.eye(10)
