"""Tests for the exact depth-scar projector ledger."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.depth_scar.prediction_ledger import (
    family_projectors,
    projectors_resolve_identity,
)


def test_family_projectors_are_rank_one_orthogonal_resolution() -> None:
    p0, p2, p6 = family_projectors()
    projectors = (p0, p2, p6)

    assert all(projector.rank() == 1 for projector in projectors)
    assert all(
        sp.simplify(projector * projector - projector) == sp.zeros(3, 3)
        for projector in projectors
    )
    assert sp.simplify(p0 * p2) == sp.zeros(3, 3)
    assert sp.simplify(p0 * p6) == sp.zeros(3, 3)
    assert sp.simplify(p2 * p6) == sp.zeros(3, 3)
    assert sp.simplify(p0 + p2 + p6) == sp.eye(3)
    assert projectors_resolve_identity()


def test_projectors_have_expected_port_forms() -> None:
    p0, p2, p6 = family_projectors()
    assert p0 == sp.ones(3, 3) / 3
    assert p2 == sp.Matrix([[sp.Rational(1, 2), 0, sp.Rational(-1, 2)], [0, 0, 0],
                            [sp.Rational(-1, 2), 0, sp.Rational(1, 2)]])
    assert p6 == sp.Matrix(
        [
            [sp.Rational(1, 6), sp.Rational(-1, 3), sp.Rational(1, 6)],
            [sp.Rational(-1, 3), sp.Rational(2, 3), sp.Rational(-1, 3)],
            [sp.Rational(1, 6), sp.Rational(-1, 3), sp.Rational(1, 6)],
        ]
    )

