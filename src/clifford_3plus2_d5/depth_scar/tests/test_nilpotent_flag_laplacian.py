"""Tests for the nilpotent flag induced Laplacian and transfer."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_adjacency_operator,
    flag_degree_operator,
    flag_laplacian_from_nilpotent,
    nilpotent_flag_laplacian_matches_path,
    nilpotent_flag_spectrum_pass,
    nilpotent_flag_transfer_matches_v1,
    nilpotent_flag_transfer_operator,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
    path_laplacian,
    transfer_operator_port_basis,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_nilpotent_flag_induces_path_adjacency_degree_and_laplacian() -> None:
    assert flag_adjacency_operator() == sp.Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert flag_degree_operator() == sp.diag(1, 2, 1)
    assert flag_laplacian_from_nilpotent() == path_laplacian()
    assert nilpotent_flag_laplacian_matches_path()


def test_nilpotent_flag_has_target_laplacian_and_depth_spectra() -> None:
    laplacian = flag_laplacian_from_nilpotent()

    assert _spectrum_tuple(laplacian) == EXPECTED_LAPLACIAN_SPECTRUM
    assert _spectrum_tuple(2 * laplacian) == EXPECTED_DEPTH_SPECTRUM
    assert nilpotent_flag_spectrum_pass()


def test_nilpotent_flag_transfer_matches_v1_transfer_kernel() -> None:
    assert sp.simplify(
        nilpotent_flag_transfer_operator() - transfer_operator_port_basis()
    ) == sp.zeros(3, 3)
    assert nilpotent_flag_transfer_matches_v1()
