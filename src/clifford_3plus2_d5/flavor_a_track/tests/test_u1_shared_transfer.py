"""Tests for U1 — shared transfer invariant."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.u1_shared_transfer import (
    residual_graph_root,
    sector_actual_values,
    sector_epsilon_powers,
    sectors_match_root,
    shared_transfer_audit_payload,
)


def test_k3_root_is_silver_ratio() -> None:
    assert sp.simplify(residual_graph_root(3) - (sp.sqrt(2) - 1)) == 0


def test_every_sector_matches_the_k3_root() -> None:
    assert sectors_match_root(residual_graph_root(3))


def test_sector_values_are_silver_ratio_powers() -> None:
    root = residual_graph_root(3)
    actual = sector_actual_values()
    powers = sector_epsilon_powers()
    for name, value in actual.items():
        assert sp.simplify(value - root ** powers[name]) == 0


def test_negative_control_k4_root_does_not_match() -> None:
    # The real sectors are the K3 root, NOT the K4 root: the gate is sensitive.
    assert not sectors_match_root(residual_graph_root(4))


def test_payload_pass_and_controls() -> None:
    payload = shared_transfer_audit_payload()
    assert payload.final_verdict == "SHARED_TRANSFER_INVARIANT"
    assert payload.sectors_match_k3_root
    assert payload.graph_roots_distinct
    assert payload.independent_epsilon_detected
    assert sp.simplify(payload.residual_graph_root - (sp.sqrt(2) - 1)) == 0
