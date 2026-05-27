"""Tests for the V8 conditional leptonic phase-word audit."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_audit_payload,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    coxeter_spin_lift_angle,
    leptonic_phase_word_angle,
    leptonic_phase_word_audit_payload,
    leptonic_phase_word_phase,
    phase_from_pi_angle,
    principal_pi_angle,
    raw_leptonic_phase_word_angle,
    second_order_schur_angle,
)


def test_coxeter_spin_lift_angles_are_exact() -> None:
    assert coxeter_spin_lift_angle("A3") == sp.Rational(1, 4)
    assert coxeter_spin_lift_angle("A2") == sp.Rational(1, 3)


def test_unknown_coxeter_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Coxeter"):
        coxeter_spin_lift_angle("A1")


def test_second_order_schur_sign_is_pi() -> None:
    assert second_order_schur_angle() == 1
    assert phase_from_pi_angle(second_order_schur_angle()) == sp.exp(sp.I * sp.pi)


def test_principal_angle_maps_raw_full_word_to_minus_five_twelfths() -> None:
    assert principal_pi_angle(sp.Rational(19, 12)) == -sp.Rational(5, 12)


def test_full_phase_word_angles_are_exact() -> None:
    assert raw_leptonic_phase_word_angle() == sp.Rational(19, 12)
    assert leptonic_phase_word_angle() == -sp.Rational(5, 12)


def test_full_phase_word_matches_expected_phase() -> None:
    assert leptonic_phase_word_phase() == sp.exp(-sp.I * sp.pi * sp.Rational(5, 12))


def test_no_schur_and_subword_controls_do_not_match_full_phase() -> None:
    target = -sp.Rational(5, 12)
    assert leptonic_phase_word_angle(include_schur=False) != target
    assert leptonic_phase_word_angle(word=("A3",)) != target
    assert leptonic_phase_word_angle(word=("A2",)) != target


def test_cp_conjugate_branch_is_plus_five_twelfths() -> None:
    assert principal_pi_angle(-leptonic_phase_word_angle()) == sp.Rational(5, 12)


def test_phase_word_audit_reports_conditional_pass_but_parks_pmns() -> None:
    payload = leptonic_phase_word_audit_payload()
    assert payload.final_verdict == "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
    assert payload.raw_full_angle == sp.Rational(19, 12)
    assert payload.principal_full_angle == -sp.Rational(5, 12)
    assert payload.cp_conjugate_angle == sp.Rational(5, 12)
    assert not payload.no_schur_matches
    assert not payload.a3_only_matches
    assert not payload.a2_only_matches
    assert not payload.word_selection_derived
    assert payload.pmns_ckm_parked


def test_v7_leakage_pass_regression() -> None:
    assert charged_lepton_leakage_audit_payload().final_verdict == "CHARGED_LEPTON_LEAKAGE_PASS"
