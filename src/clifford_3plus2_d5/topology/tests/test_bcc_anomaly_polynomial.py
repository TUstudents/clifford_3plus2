"""Tests for ``anomaly_cancellation/bcc_anomaly_polynomial.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.topology.anomaly_cancellation.bcc_anomaly_polynomial import (
    all_anomalies_cancel,
    anomaly_polynomial_payload,
    gravitational_anomaly,
    su2_squared_u1_anomaly,
    su3_squared_u1_anomaly,
    u1_cubed_anomaly,
)


def test_gravitational_anomaly_cancels_per_generation() -> None:
    assert gravitational_anomaly(1) == 0


def test_u1_cubed_anomaly_cancels_per_generation() -> None:
    assert u1_cubed_anomaly(1) == 0


def test_su2_squared_u1_anomaly_cancels_per_generation() -> None:
    assert su2_squared_u1_anomaly(1) == 0


def test_su3_squared_u1_anomaly_cancels_per_generation() -> None:
    assert su3_squared_u1_anomaly(1) == 0


def test_all_anomalies_cancel_for_one_generation() -> None:
    assert all_anomalies_cancel(1)


def test_all_anomalies_cancel_for_three_generations() -> None:
    # Cancellation is per-generation, so it scales linearly with N.
    assert all_anomalies_cancel(3)


def test_all_anomalies_cancel_for_arbitrary_N() -> None:
    # Standard SM result: cancellation holds for any N ≥ 0.
    for N in (0, 1, 2, 3, 5, 17):
        assert all_anomalies_cancel(N)


def test_anomaly_polynomial_payload_reports_cancellation() -> None:
    payload = anomaly_polynomial_payload(3)
    assert payload.generations == 3
    assert payload.gravitational == 0
    assert payload.u1_cubed == 0
    assert payload.su2_squared_u1 == 0
    assert payload.su3_squared_u1 == 0
    assert payload.all_cancel
    assert "cancel" in payload.interpretation


def test_per_generation_y_squared_is_sympy_expression() -> None:
    # Sanity check: returned objects are sympy expressions
    grav = gravitational_anomaly(1)
    assert isinstance(grav, sp.Expr)
