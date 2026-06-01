"""Tests for the exact path-defect repair Laplacian."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    depth_scar_operator,
    path_incidence_matrix,
    path_laplacian,
    path_spectrum_passes,
    spectrum_dict,
)


def test_path_incidence_gives_p3_laplacian() -> None:
    incidence = path_incidence_matrix()
    assert incidence == sp.Matrix([[1, -1, 0], [0, 1, -1]])
    assert path_laplacian() == incidence.T * incidence
    assert path_laplacian() == sp.Matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])


def test_depth_scar_operator_is_positive_semidefinite_with_target_spectrum() -> None:
    spectrum = spectrum_dict(depth_scar_operator())
    assert spectrum[sp.Integer(0)] == 1
    assert spectrum[sp.Integer(2)] == 1
    assert spectrum[sp.Integer(6)] == 1
    assert tuple(sorted(spectrum, key=sp.default_sort_key)) == EXPECTED_DEPTH_SPECTRUM
    assert spectrum_dict(path_laplacian()) == {
        sp.Integer(0): 1,
        sp.Integer(1): 1,
        sp.Integer(3): 1,
    }
    assert EXPECTED_LAPLACIAN_SPECTRUM == (sp.Integer(0), sp.Integer(1), sp.Integer(3))
    assert path_spectrum_passes()

