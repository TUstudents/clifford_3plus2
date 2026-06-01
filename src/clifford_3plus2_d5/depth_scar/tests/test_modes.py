"""Tests for the defect normal modes."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    defect_eigenbasis_matrix,
    defect_mode_depths,
    defect_modes,
    depth_scar_operator,
)


def test_defect_modes_are_orthonormal() -> None:
    basis = defect_eigenbasis_matrix()
    assert sp.simplify(basis.T * basis) == sp.eye(3)


def test_defect_modes_have_depths_0_2_6() -> None:
    operator = depth_scar_operator()
    for mode, expected in zip(defect_modes(), EXPECTED_DEPTH_SPECTRUM, strict=True):
        assert sp.simplify(operator * mode - expected * mode) == sp.zeros(3, 1)
    assert defect_mode_depths() == EXPECTED_DEPTH_SPECTRUM

