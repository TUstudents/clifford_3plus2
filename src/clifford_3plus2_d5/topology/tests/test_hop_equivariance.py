"""Tests for ``hop_equivariance.py``."""

from __future__ import annotations

from clifford_3plus2_d5.topology.hop_equivariance import (
    all_hops_equivariant,
    equivariance_failures,
    equivariance_residual_for_index,
    hop_equivariance_audit_payload,
)


def test_equivariance_residual_shape() -> None:
    residual = equivariance_residual_for_index(0)
    assert residual.shape == (2, 2)


def test_audit_payload_runs() -> None:
    payload = hop_equivariance_audit_payload()
    assert payload.direction_count == 8
    assert payload.cycle_lengths == (3, 3, 1, 1)
    assert len(payload.permutation) == 8


def test_audit_records_pass_or_fail() -> None:
    payload = hop_equivariance_audit_payload()
    if payload.all_equivariant:
        assert "EQUIVARIANT" in payload.verdict
        assert payload.failure_count == 0
    else:
        assert "NOT EQUIVARIANT" in payload.verdict
        assert payload.failure_count > 0


def test_failures_match_equivariance_check() -> None:
    failures = equivariance_failures()
    if all_hops_equivariant():
        assert len(failures) == 0
    else:
        assert len(failures) > 0
