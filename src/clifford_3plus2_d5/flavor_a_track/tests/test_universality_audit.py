"""Tests for U4 — combined universality verdict + the A3 deferral."""

from __future__ import annotations

from clifford_3plus2_d5.flavor_a_track.u1_shared_transfer import (
    residual_graph_root,
    sectors_match_root,
)
from clifford_3plus2_d5.flavor_a_track.universality_audit import (
    REMAINING_DECLARED_INPUTS,
    universality_audit_payload,
)


def test_combined_necessary_conditions_pass() -> None:
    payload = universality_audit_payload()
    assert payload.final_verdict == "UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS"
    assert payload.necessary_conditions_pass


def test_subpayloads_pass() -> None:
    payload = universality_audit_payload()
    assert payload.shared_transfer.final_verdict == "SHARED_TRANSFER_INVARIANT"
    assert payload.color_label_partition.final_verdict == "SECTOR_DIFFERENCE_IS_COLOR_LABEL"
    assert payload.coupling_catalog.final_verdict == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"


def test_full_reproduction_is_deferred_to_a3() -> None:
    payload = universality_audit_payload()
    assert payload.full_reproduction_confirmed is False
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
    assert payload.remaining_declared_inputs == (
        "unified_H_Q_on_chiral16_reproduces_all_Sigma_f",
    )


def test_gate_can_actually_fail_decisive_negative_control() -> None:
    # U1 is sensitive to a sector-dependent epsilon: the real sectors match the
    # K3 root but NOT the K4 root. If a sector used a different graph root, U1
    # would return INDEPENDENT_EPSILON_KILL and the combined verdict would flip.
    assert sectors_match_root(residual_graph_root(3))
    assert not sectors_match_root(residual_graph_root(4))
