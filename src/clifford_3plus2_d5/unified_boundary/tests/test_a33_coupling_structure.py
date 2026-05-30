"""Tests for A3-3 — sector structure lives in the coupling V_f."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.unified_boundary.a33_coupling_structure import (
    bcc_factors_are_sqrt2_and_inverse,
    coin_phase_is_atan_sqrt5,
    color_factor_is_four_thirds,
    color_return_is_casimir_identity,
    coupling_structure_audit_payload,
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
