"""Tests for ``anomaly_cancellation/anomaly_audit.py``."""

from __future__ import annotations

from clifford_3plus2_d5.topology.anomaly_cancellation.anomaly_audit import (
    anomaly_audit_payload,
)


def test_audit_three_generations_passes() -> None:
    payload = anomaly_audit_payload(3)
    assert payload.generations == 3
    assert payload.polynomial.all_cancel
    assert payload.global_check.witten_anomaly_free
    assert payload.discrete_constraint.per_generation_continuous_cancels
    assert payload.discrete_constraint.per_generation_global_anomaly_free
    assert payload.all_cancellations_satisfied
    assert not payload.forces_three_generations
    assert "ANOMALY KILL" in payload.verdict


def test_audit_any_N_passes() -> None:
    for N in (1, 2, 3, 5, 10):
        payload = anomaly_audit_payload(N)
        assert payload.all_cancellations_satisfied
        assert not payload.forces_three_generations


def test_audit_links_to_subphase_payloads() -> None:
    payload = anomaly_audit_payload(3)
    # Sub-payload identity
    assert payload.polynomial.generations == 3
    assert payload.global_check.generations == 3
    assert payload.discrete_constraint.admissible_N_up_to_twenty == tuple(range(21))


def test_audit_interpretation_mentions_anomaly_cancellation() -> None:
    payload = anomaly_audit_payload(3)
    text = payload.interpretation.lower()
    assert "anomal" in text
    assert "n = 3" in text or "n=3" in text or "three" in text
