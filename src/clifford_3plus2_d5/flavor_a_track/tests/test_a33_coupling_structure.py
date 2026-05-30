"""Tests for A3-3 — sector structure lives in the coupling V_f."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.a33_coupling_structure import (
    bcc_factors_are_sqrt2_and_inverse,
    coin_phase_is_atan_sqrt5,
    color_factor_is_four_thirds,
    color_return_is_casimir_identity,
    coupling_structure_audit_payload,
    coupling_structure_verdict,
    lepton_is_color_singlet_quark_triplet,
)


def test_color_return_is_four_thirds_identity() -> None:
    assert color_return_is_casimir_identity()
    assert color_factor_is_four_thirds()


def test_bcc_and_coin_factors() -> None:
    assert bcc_factors_are_sqrt2_and_inverse()
    assert coin_phase_is_atan_sqrt5()


def test_lepton_singlet_quark_triplet() -> None:
    assert lepton_is_color_singlet_quark_triplet()


def test_payload_pass() -> None:
    payload = coupling_structure_audit_payload()
    assert payload.final_verdict == "SECTOR_STRUCTURE_IN_COUPLINGS"
    assert sp.simplify(payload.color_factor_value - sp.Rational(4, 3)) == 0
    assert payload.color_return_is_casimir
    assert payload.lepton_singlet_quark_triplet


def test_verdict_kills_when_any_check_fails() -> None:
    # Decisive negative control: any one of the five V_f checks failing forces
    # the KILL string.
    assert coupling_structure_verdict(True, True, True, True, True) == "SECTOR_STRUCTURE_IN_COUPLINGS"
    kill = "SECTOR_STRUCTURE_IN_COUPLINGS_KILL"
    assert coupling_structure_verdict(False, True, True, True, True) == kill
    assert coupling_structure_verdict(True, False, True, True, True) == kill
    assert coupling_structure_verdict(True, True, False, True, True) == kill
    assert coupling_structure_verdict(True, True, True, False, True) == kill
    assert coupling_structure_verdict(True, True, True, True, False) == kill
