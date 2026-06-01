"""Controls separating the scar from the unbroken K3 graph and diagonal spurion."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.audit import depth_scar_operator_trace_spurion
from clifford_3plus2_d5.depth_scar.repair_graphs import (
    doubled_k3_laplacian,
    hand_written_diagonal_depth_operator,
    hand_written_diagonal_is_not_graph_native,
    k3_laplacian,
    spectrum_dict,
    unbroken_k3_control_fails_hierarchy,
)


def test_unbroken_k3_stays_degenerate() -> None:
    assert spectrum_dict(k3_laplacian()) == {sp.Integer(0): 1, sp.Integer(3): 2}
    assert spectrum_dict(doubled_k3_laplacian()) == {sp.Integer(0): 1, sp.Integer(6): 2}
    assert unbroken_k3_control_fails_hierarchy()


def test_hand_written_diagonal_matches_spectrum_but_is_not_graph_native() -> None:
    diagonal = hand_written_diagonal_depth_operator()
    assert diagonal == sp.diag(0, 2, 6)
    assert hand_written_diagonal_is_not_graph_native()


def test_path_scar_traceless_part_is_nonzero_s3_breaking_spurion() -> None:
    spurion = depth_scar_operator_trace_spurion()
    assert sp.simplify(spurion.trace()) == 0
    assert spurion == sp.Matrix(
        [
            [sp.Rational(-2, 3), -2, 0],
            [-2, sp.Rational(4, 3), -2],
            [0, -2, sp.Rational(-2, 3)],
        ]
    )

