"""Tests for ``anomaly_cancellation/global_anomaly_check.py``."""

from __future__ import annotations

from clifford_3plus2_d5.topology.anomaly_cancellation.global_anomaly_check import (
    global_anomaly_check_payload,
    mod2_lattice_anomaly,
    su2_doublet_count,
    su2_doublet_count_per_generation,
    witten_anomaly_free,
    witten_constrains_generation_count,
)


def test_doublet_count_per_generation_is_four() -> None:
    # 3 (Q_L colors) + 1 (L_L) = 4
    assert su2_doublet_count_per_generation() == 4


def test_doublet_count_scales_linearly() -> None:
    for N in (1, 2, 3, 7):
        assert su2_doublet_count(N) == 4 * N


def test_witten_anomaly_free_for_any_N() -> None:
    for N in (0, 1, 2, 3, 5, 10, 17):
        assert witten_anomaly_free(N)


def test_mod2_lattice_anomaly_is_zero_for_any_N() -> None:
    for N in (0, 1, 2, 3, 5, 10):
        assert mod2_lattice_anomaly(N) == 0


def test_witten_does_not_constrain_N() -> None:
    # 4 is even, so 4N is always even.
    assert not witten_constrains_generation_count()


def test_payload_reports_satisfied_for_three_generations() -> None:
    payload = global_anomaly_check_payload(3)
    assert payload.generations == 3
    assert payload.doublets_per_generation == 4
    assert payload.total_doublets == 12
    assert payload.witten_anomaly_free
    assert payload.mod2_lattice_anomaly == 0
    assert not payload.witten_constrains_N
    assert "every N" in payload.interpretation
