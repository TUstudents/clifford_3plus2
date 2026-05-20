"""Tests for ``bcc_z3_on_flavor.py``."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.koide.bcc_z3_on_flavor import (
    bcc_z3_on_flavor_payload,
    generation_to_sigma_axis,
    koide_condition_on_pdg_vector,
    sigma_axis_to_generation_label,
    trace_projector_commutes_with_z3,
    traceless_projector_commutes_with_z3,
    z3_rotation_3d,
    z3_rotation_fixes_diagonal_direction,
    z3_rotation_order_three,
    z3_rotation_orthogonal_det_one,
)


def test_z3_rotation_is_3x3() -> None:
    R = z3_rotation_3d()
    assert R.shape == (3, 3)


def test_z3_rotation_matches_expected_cyclic() -> None:
    R = z3_rotation_3d()
    expected = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert R == expected


def test_z3_rotation_fixes_diagonal_direction() -> None:
    assert z3_rotation_fixes_diagonal_direction()


def test_z3_rotation_order_three() -> None:
    assert z3_rotation_order_three()


def test_z3_rotation_orthogonal_det_one() -> None:
    assert z3_rotation_orthogonal_det_one()


def test_trace_projector_commutes_with_z3() -> None:
    assert trace_projector_commutes_with_z3()


def test_traceless_projector_commutes_with_z3() -> None:
    assert traceless_projector_commutes_with_z3()


def test_sigma_to_generation_convention() -> None:
    assert sigma_axis_to_generation_label("x") == "e"
    assert sigma_axis_to_generation_label("y") == "μ"
    assert sigma_axis_to_generation_label("z") == "τ"


def test_generation_to_sigma_axis_inverse() -> None:
    assert generation_to_sigma_axis("e") == "x"
    assert generation_to_sigma_axis("μ") == "y"
    assert generation_to_sigma_axis("τ") == "z"


def test_sigma_axis_validation() -> None:
    with pytest.raises(ValueError):
        sigma_axis_to_generation_label("w")
    with pytest.raises(ValueError):
        generation_to_sigma_axis("strange")


def test_pdg_koide_condition_holds_under_identification() -> None:
    assert koide_condition_on_pdg_vector()


def test_payload_all_consistent() -> None:
    p = bcc_z3_on_flavor_payload()
    assert p.z3_fixes_diagonal
    assert p.z3_is_order_three
    assert p.z3_orthogonal_det_one
    assert p.trace_projector_z3_invariant
    assert p.traceless_projector_z3_invariant
    assert p.pdg_koide_condition_holds
    assert "CONSISTENT" in p.verdict
