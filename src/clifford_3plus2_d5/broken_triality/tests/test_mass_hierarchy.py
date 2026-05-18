"""Tests for ``mass_hierarchy.py`` — BT-2 kill test."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.broken_triality.mass_hierarchy import (
    FAIL_THRESHOLD,
    PASS_THRESHOLD,
    bt2_audit_payload,
    nonzero_eigenvalue_count,
    nonzero_eigenvalue_ratio,
    ratio_below_fail_threshold,
    ratio_passes_pass_threshold,
)


def test_default_yukawa_has_two_nonzero_eigenvalues() -> None:
    assert nonzero_eigenvalue_count() == 2


def test_nonzero_eigenvalue_ratio_is_below_two() -> None:
    ratio = nonzero_eigenvalue_ratio()
    assert ratio is not None
    # (5/7) / (31/72) = 360/217
    assert ratio == sp.Rational(360, 217)
    assert float(ratio) < 2


def test_ratio_below_fail_threshold() -> None:
    assert ratio_below_fail_threshold()
    assert not ratio_passes_pass_threshold()


def test_bt2_audit_reports_fail() -> None:
    payload = bt2_audit_payload()
    assert not payload.passes
    assert "FAIL" in payload.verdict
    assert payload.nonzero_eigenvalue_count == 2
    assert payload.nonzero_ratio == sp.Rational(360, 217)
    assert payload.nonzero_ratio_float is not None
    assert payload.nonzero_ratio_float < 2
    assert payload.pass_threshold == PASS_THRESHOLD
    assert payload.fail_threshold == FAIL_THRESHOLD
    assert "flat" in payload.interpretation or "essentially flat" in payload.interpretation


def test_thresholds_are_correctly_set() -> None:
    assert PASS_THRESHOLD == 100
    assert FAIL_THRESHOLD == 10
