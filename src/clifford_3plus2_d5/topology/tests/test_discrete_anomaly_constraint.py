"""Tests for ``anomaly_cancellation/discrete_anomaly_constraint.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.topology.anomaly_cancellation.discrete_anomaly_constraint import (
    admissible_generation_counts,
    anomalies_force_N_equals_three,
    constraint_on_N,
    discrete_anomaly_constraint_payload,
    per_generation_anomaly_polynomial_is_zero,
    per_generation_global_anomaly_is_anomaly_free,
)


def test_per_generation_continuous_anomalies_vanish() -> None:
    assert per_generation_anomaly_polynomial_is_zero()


def test_per_generation_global_anomaly_satisfied() -> None:
    assert per_generation_global_anomaly_is_anomaly_free()


def test_constraint_is_trivially_zero() -> None:
    assert constraint_on_N() == sp.Integer(0)


def test_anomalies_do_not_force_three_generations() -> None:
    assert not anomalies_force_N_equals_three()


def test_admissible_generations_is_full_range() -> None:
    admissible = admissible_generation_counts(20)
    assert admissible == tuple(range(21))


def test_three_is_admissible_but_not_unique() -> None:
    admissible = admissible_generation_counts(20)
    assert 3 in admissible
    # Other values are also admissible — i.e. 3 is not the unique answer.
    assert 1 in admissible
    assert 5 in admissible
    assert 7 in admissible


def test_payload_kill_verdict() -> None:
    payload = discrete_anomaly_constraint_payload()
    assert payload.per_generation_continuous_cancels
    assert payload.per_generation_global_anomaly_free
    assert payload.constraint_symbolic == 0
    assert not payload.forces_N_equals_three
    assert payload.admissible_N_up_to_twenty == tuple(range(21))
    assert "ANOMALY KILL" in payload.verdict
