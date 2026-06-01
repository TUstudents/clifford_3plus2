"""Tests for the minimal nilpotent support classification."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import nilpotent_flag_operator
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    accepted_minimal_nilpotent_supports,
    accepted_minimal_orbit_keys,
    all_minimal_supports_equivalent_to_flag,
    flag_laplacian_from_nilpotent,
    is_length_three_nilpotent,
    minimal_support_spectra_pass,
    permutation_equivalent,
    support_edge_count,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_minimal_length_three_nilpotent_supports_form_one_s3_orbit() -> None:
    supports = accepted_minimal_nilpotent_supports()

    assert len(supports) == 6
    assert len(accepted_minimal_orbit_keys()) == 1
    assert all(support_edge_count(support) == 2 for support in supports)
    assert all(support.rank() == 2 for support in supports)
    assert all(is_length_three_nilpotent(support) for support in supports)
    assert all(permutation_equivalent(support, nilpotent_flag_operator()) for support in supports)
    assert all_minimal_supports_equivalent_to_flag()


def test_each_accepted_minimal_support_induces_target_spectra() -> None:
    for support in accepted_minimal_nilpotent_supports():
        laplacian = flag_laplacian_from_nilpotent(support)
        assert _spectrum_tuple(laplacian) == EXPECTED_LAPLACIAN_SPECTRUM
        assert _spectrum_tuple(2 * laplacian) == EXPECTED_DEPTH_SPECTRUM
    assert minimal_support_spectra_pass()
