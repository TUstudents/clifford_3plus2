"""Tests for ``walk_symmetries.py`` — alpha-2 and alpha-3 audits.

Slow: each test triggers symbolic Bloch-symbol P/T/C/CP/CPT
conjugation checks at O(ε²); 9 tests at ~13-27s each ≈ 2 minutes
for the module.
"""

from __future__ import annotations

import pytest

from clifford_3plus2_d5.cp.walk_symmetries import (
    massless_audit_payload,
    yukawa_audit_payload,
)

pytestmark = pytest.mark.slow


def test_alpha2_cpt_exact() -> None:
    payload = massless_audit_payload()
    assert payload.cpt_exact


def test_alpha2_p_exact() -> None:
    payload = massless_audit_payload()
    assert payload.p_exact


def test_alpha2_cp_broken() -> None:
    payload = massless_audit_payload()
    assert not payload.cp_exact


def test_alpha2_t_and_c_broken() -> None:
    payload = massless_audit_payload()
    assert not payload.t_exact
    assert not payload.c_exact


def test_alpha2_full_verdict_table() -> None:
    payload = massless_audit_payload()
    expected = {
        "P": True,
        "T": False,
        "C": False,
        "PT": False,
        "CP": False,
        "CT": True,
        "CPT": True,
    }
    for verdict in payload.verdicts:
        assert verdict.exact == expected[verdict.name], (
            f"unexpected verdict for {verdict.name}"
        )


def test_alpha2_pass_interpretation() -> None:
    payload = massless_audit_payload()
    assert "ALPHA PASS" in payload.interpretation


def test_alpha3_matches_alpha2_under_trivial_internal_action() -> None:
    yukawa = yukawa_audit_payload()
    massless = massless_audit_payload()
    for y_verdict, m_verdict in zip(yukawa.verdicts, massless.verdicts, strict=True):
        assert y_verdict.name == m_verdict.name
        # Under trivial internal action, combined verdict equals kinetic verdict
        # AND beta-preservation.
        if y_verdict.kinetic_exact and y_verdict.yukawa_beta_preserved:
            assert y_verdict.combined_exact
        else:
            assert not y_verdict.combined_exact


def test_alpha3_p_cpt_ct_exact_combined() -> None:
    payload = yukawa_audit_payload()
    assert payload.cpt_exact
    assert payload.p_exact


def test_alpha3_cp_broken_combined() -> None:
    payload = yukawa_audit_payload()
    assert not payload.cp_exact
