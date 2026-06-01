"""Tests for spectra at V3 edge-weight scar minima."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.edge_weight_potential import (
    path_scar_depth_spectra,
    path_scar_laplacian_spectra,
    path_scar_spectra_pass,
    path_scar_weight_vectors,
    weighted_triangle_laplacian_general,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_each_path_scar_minimum_has_target_laplacian_spectrum() -> None:
    for weights in path_scar_weight_vectors():
        laplacian = weighted_triangle_laplacian_general(weights)
        assert _spectrum_tuple(laplacian) == EXPECTED_LAPLACIAN_SPECTRUM
    assert all(spectrum == EXPECTED_LAPLACIAN_SPECTRUM for spectrum in path_scar_laplacian_spectra())


def test_each_path_scar_minimum_has_target_doubled_depth_spectrum() -> None:
    for weights in path_scar_weight_vectors():
        laplacian = weighted_triangle_laplacian_general(weights)
        assert _spectrum_tuple(2 * laplacian) == EXPECTED_DEPTH_SPECTRUM
    assert all(spectrum == EXPECTED_DEPTH_SPECTRUM for spectrum in path_scar_depth_spectra())
    assert path_scar_spectra_pass()

