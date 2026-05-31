"""Tests for W2 — [111]-singlet support (the primary gate) + its verdict taxonomy."""

from __future__ import annotations

from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_support_audit import (
    HELICITY_SPLIT,
    KILL_MISSING_A2U,
    KILL_MISSING_T1U,
    KILL_T2G_PRESENT,
    PASS,
    combined_verdict,
    hop_walsh_support_audit_payload,
    hop_walsh_support_verdict,
)


def test_verdict_taxonomy() -> None:
    # (baseline, t1u, a2u, t2g) -> verdict, with fixed priority so each control
    # hits exactly one reason.
    assert hop_walsh_support_verdict(True, True, True, False) == PASS
    assert hop_walsh_support_verdict(True, True, False, False) == KILL_MISSING_A2U
    assert hop_walsh_support_verdict(True, True, True, True) == KILL_T2G_PRESENT
    assert hop_walsh_support_verdict(True, False, True, False) == KILL_MISSING_T1U


def test_combined_verdict_logic() -> None:
    assert combined_verdict(PASS, PASS) == PASS
    assert combined_verdict(PASS, KILL_T2G_PRESENT) == HELICITY_SPLIT
    assert combined_verdict(KILL_T2G_PRESENT, PASS) == HELICITY_SPLIT
    assert combined_verdict(KILL_T2G_PRESENT, KILL_T2G_PRESENT) == KILL_T2G_PRESENT


def test_real_verdict_is_t2g_present_kill() -> None:
    # The genuine experiment: the BB Weyl hop source carries a nonzero [111] T2g
    # singlet, so the cube/parity {0,2,6} story is killed (both helicities).
    payload = hop_walsh_support_audit_payload()
    assert payload.final_verdict == KILL_T2G_PRESENT
    assert payload.right_handed_verdict == KILL_T2G_PRESENT
    assert payload.left_handed_verdict == KILL_T2G_PRESENT


def test_real_support_flags() -> None:
    payload = hop_walsh_support_audit_payload()
    # A1g, T1u[111], A2u present; the decisive failure is the present T2g[111].
    assert payload.right.baseline_present
    assert payload.right.t1u_singlet_present
    assert payload.right.a2u_present
    assert payload.right.t2g_singlet_present
    assert payload.right.t2g_full_triplet_zero is False
