"""Minimizer tests for V8 minimal causal repair."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.minimal_repair_variational import (
    causal_repair_cost,
    minimal_causal_repair_cost,
    minimal_causal_repair_minimizers,
    minimizer_orbit_keys,
    minimizer_spectra_pass,
    minimizers_equivalent_to_flag,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    EXPECTED_DEPTH_SPECTRUM,
    EXPECTED_LAPLACIAN_SPECTRUM,
)
from clifford_3plus2_d5.depth_scar.support_classification import (
    accepted_minimal_nilpotent_supports,
    accepted_minimal_orbit_keys,
    flag_laplacian_from_nilpotent,
    support_key,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def _support_key_set(supports):
    return {support_key(support) for support in supports}


def test_minimal_causal_repair_cost_selects_v7_flag_orbit() -> None:
    minimizers = minimal_causal_repair_minimizers()

    assert minimal_causal_repair_cost() == 2
    assert len(minimizers) == 6
    assert all(causal_repair_cost(support) == 2 for support in minimizers)
    assert _support_key_set(minimizers) == _support_key_set(accepted_minimal_nilpotent_supports())
    assert minimizer_orbit_keys() == accepted_minimal_orbit_keys()
    assert minimizers_equivalent_to_flag()


def test_minimal_causal_repair_minimizers_induce_target_spectra() -> None:
    for support in minimal_causal_repair_minimizers():
        laplacian = flag_laplacian_from_nilpotent(support)
        assert _spectrum_tuple(laplacian) == EXPECTED_LAPLACIAN_SPECTRUM
        assert _spectrum_tuple(2 * laplacian) == EXPECTED_DEPTH_SPECTRUM

    assert minimizer_spectra_pass()
