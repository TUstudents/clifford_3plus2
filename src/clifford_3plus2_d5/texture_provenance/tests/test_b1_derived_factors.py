"""Tests for B1 — derived-factor ledger."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.texture_provenance.b1_derived_factors import (
    bcc_clebsch_check,
    charged_lepton_clebsch,
    coin_dimension_check,
    color_return_check,
    derived_factors_audit_payload,
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
