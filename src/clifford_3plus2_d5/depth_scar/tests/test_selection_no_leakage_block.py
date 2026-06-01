"""V11 tests for unique-successor repair blocks."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.microscopic_locality import PATH_REPAIR_EDGES
from clifford_3plus2_d5.depth_scar.selection_no_leakage import (
    successor_unitary_repair_block,
    unique_successor_block_is_isometry,
    unique_successor_block_matches_path_support,
)


def test_successor_unitary_repair_block_is_active_isometry() -> None:
    theta_a, theta_b = sp.symbols("theta_a theta_b", real=True)
    block = successor_unitary_repair_block(theta_a, theta_b)

    assert block == sp.Matrix(
        [
            [sp.exp(sp.I * theta_a), 0],
            [0, sp.exp(sp.I * theta_b)],
            [0, 0],
        ]
    )
    assert sp.simplify(block.H * block - sp.eye(2)) == sp.zeros(2, 2)
    assert unique_successor_block_is_isometry()


def test_successor_unitary_repair_block_has_path_support() -> None:
    theta_a, theta_b = sp.symbols("theta_a theta_b", real=True)
    block = successor_unitary_repair_block(theta_a, theta_b)
    support = tuple(
        (row, col + 1)
        for row in range(3)
        for col in range(2)
        if block[row, col] != 0
    )

    assert support == PATH_REPAIR_EDGES
    assert unique_successor_block_matches_path_support()
