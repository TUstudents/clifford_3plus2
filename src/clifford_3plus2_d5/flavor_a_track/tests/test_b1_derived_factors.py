"""Tests for B1 — derived-factor ledger."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.b1_derived_factors import (
    bcc_clebsch_check,
    charged_lepton_clebsch,
    coin_dimension_check,
    color_return_check,
    derived_factors_audit_payload,
    derived_factors_verdict,
    leptonic_phase_word_check,
)


def test_color_return_is_su3_casimir() -> None:
    assert color_return_check()


def test_coin_base_five_is_two_bcc_plus_three_color() -> None:
    assert coin_dimension_check()


def test_bcc_clebsches() -> None:
    assert bcc_clebsch_check()


def test_charged_lepton_clebsch_is_sqrt_3_2() -> None:
    assert sp.simplify(charged_lepton_clebsch() - sp.sqrt(sp.Rational(3, 2))) == 0


def test_leptonic_phase_word_derived() -> None:
    assert leptonic_phase_word_check()


def test_payload_all_checks_pass() -> None:
    payload = derived_factors_audit_payload()
    assert payload.final_verdict == "DERIVED_FACTORS_CATALOGUED"
    assert payload.all_checks_pass
    assert all(factor.check_passes for factor in payload.derived_factors)


def test_verdict_kills_when_a_factor_check_fails() -> None:
    # Decisive negative control: if any factor's source check fails (e.g. the
    # external V10 holonomy returns LEPTONIC_PHASE_WORD_DERIVED_KILL, flipping
    # leptonic_phase_word_check to False), the catalogue must KILL.
    assert derived_factors_verdict(True) == "DERIVED_FACTORS_CATALOGUED"
    assert derived_factors_verdict(False) == "DERIVED_FACTORS_CATALOGUED_KILL"
