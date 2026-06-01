"""Negative controls for the V6 local-flag unitarity gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.local_flag_unitarity import (
    canonical_flag_laplacian_and_transfer_pass,
    complex_local_flag_operator,
    contractive_magnitude_control_rejected,
    cyclic_unitary_closure_operator,
    cyclic_unitary_closure_rejected,
    final_projection,
    flag_laplacian_from_nilpotent,
    initial_projection,
    rank_one_partial_isometry_control_rejected,
    unequal_magnitude_control_rejected,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import k3_laplacian


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_rank_one_partial_isometry_is_rejected_as_short_support() -> None:
    control = complex_local_flag_operator(1, 0, 0, 0)

    assert initial_projection(control) ** 2 == initial_projection(control)
    assert final_projection(control) ** 2 == final_projection(control)
    assert control**2 == sp.zeros(3, 3)
    assert rank_one_partial_isometry_control_rejected()


def test_contractive_and_unequal_magnitudes_fail_partial_isometry() -> None:
    contractive = complex_local_flag_operator(sp.Rational(1, 2), sp.Rational(1, 2), 0, 0)
    unequal = complex_local_flag_operator(1, 2, 0, 0)

    assert initial_projection(contractive) ** 2 != initial_projection(contractive)
    assert final_projection(contractive) ** 2 != final_projection(contractive)
    assert initial_projection(unequal) ** 2 != initial_projection(unequal)
    assert contractive_magnitude_control_rejected()
    assert unequal_magnitude_control_rejected()


def test_cyclic_unitary_closure_is_not_nilpotent_and_returns_k3() -> None:
    control = cyclic_unitary_closure_operator()
    laplacian = flag_laplacian_from_nilpotent(control)

    assert control**3 == sp.eye(3)
    assert laplacian == k3_laplacian()
    assert _spectrum_tuple(laplacian) == (0, 3, 3)
    assert cyclic_unitary_closure_rejected()


def test_canonical_local_flag_matches_v5_laplacian_and_v1_transfer() -> None:
    assert canonical_flag_laplacian_and_transfer_pass()
