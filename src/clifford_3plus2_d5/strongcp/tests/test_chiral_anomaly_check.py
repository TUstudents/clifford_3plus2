"""Tests for ``chiral_anomaly_check.py``.

Slow: depends on H^(2) BCH extraction via ``higher_order_parity``,
which is SymPy-heavy (~10s per test).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from clifford_3plus2_d5.strongcp.chiral_anomaly_check import (
    chiral_anomaly_check_payload,
    chiral_cross_trace_h1_h2,
    chiral_trace_h1,
    chiral_trace_h1_squared,
    chiral_trace_h2,
    h1_is_purely_vector,
    h2_is_purely_vector,
    no_direct_theta_bar_shift_at_leading_order,
)


def test_chiral_trace_h1_vanishes() -> None:
    assert chiral_trace_h1() == 0


def test_chiral_trace_h2_vanishes() -> None:
    assert chiral_trace_h2() == 0


def test_chiral_trace_h1_squared_vanishes() -> None:
    assert chiral_trace_h1_squared() == 0


def test_chiral_cross_trace_h1_h2_is_nonzero() -> None:
    # This is an empirical finding: the cross term is non-zero at O(ε³).
    # Recorded so a future regression catches the change.
    assert chiral_cross_trace_h1_h2() != 0


def test_h1_is_purely_vector() -> None:
    assert h1_is_purely_vector()


def test_h2_is_purely_vector() -> None:
    assert h2_is_purely_vector()


def test_no_direct_theta_bar_shift_at_leading_order() -> None:
    assert no_direct_theta_bar_shift_at_leading_order()


def test_payload_verdict_is_vector() -> None:
    payload = chiral_anomaly_check_payload()
    assert payload.no_direct_theta_shift_at_o_eps_eps2
    assert "VECTOR" in payload.verdict
    assert "no direct θ̄ shift" in payload.verdict


def test_payload_records_cross_term() -> None:
    payload = chiral_anomaly_check_payload()
    assert payload.cross_term_nonzero
    assert payload.tr_g5_h1 == 0
    assert payload.tr_g5_h2 == 0
    assert payload.tr_g5_h1_h2_cross != 0
