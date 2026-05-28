"""Tests for the V15 quark coin rigidity theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_coin,
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    flatness_ratio_for_phase,
    gamma_sum_square_is_five,
    isotropic_coin_is_unitary,
    isotropic_quark_coin,
    isotropic_quark_phase_angle,
    quark_coin_rigidity_audit_payload,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_gamma_sum_square_is_five_i() -> None:
    assert gamma_sum_square_is_five()


def test_isotropic_coin_is_symbolically_unitary_for_real_ratio() -> None:
    ratio = sp.Symbol("r", real=True)
    assert isotropic_coin_is_unitary(ratio)


def test_isotropic_phase_is_atan_r_sqrt5() -> None:
    ratio = sp.Symbol("r", real=True)
    expected = sp.atan(ratio * sp.sqrt(5))
    assert sp.simplify(isotropic_quark_phase_angle(ratio) - expected) == 0


def test_v11_flat_coin_is_r_equals_one_specialization() -> None:
    _assert_matrix_equal(isotropic_quark_coin(sp.Integer(1)), quark_boundary_coin())
    assert sp.simplify(isotropic_quark_phase_angle(1) - quark_boundary_phase_angle()) == 0


def test_flat_phase_requires_unit_even_odd_ratio() -> None:
    ratio = flatness_ratio_for_phase(quark_boundary_phase_angle())
    assert sp.simplify(ratio - 1) == 0


def test_nonflat_controls_are_unitary_but_change_phase() -> None:
    for ratio in (sp.Rational(1, 2), sp.Integer(2)):
        assert isotropic_coin_is_unitary(ratio)
        assert sp.simplify(isotropic_quark_phase_angle(ratio) - quark_boundary_phase_angle()) != 0


def test_quark_coin_rigidity_payload_reports_theorem_pass() -> None:
    payload = quark_coin_rigidity_audit_payload()
    assert payload.final_verdict == "QUARK_COIN_RIGIDITY_THEOREM_PASS"
    assert payload.gamma_sum_square_matches
    assert payload.symbolic_coin_unitary
    assert payload.v11_coin_is_flat_specialization
    assert payload.nonflat_controls_unitary
    assert payload.nonflat_controls_change_phase
    assert payload.flat_ergodicity_required
    assert not payload.ckm_phase_forced_by_unitarity_alone
