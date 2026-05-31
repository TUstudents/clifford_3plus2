"""Tests for W3 — effective-Hamiltonian diagnostic (reports only, never kills)."""

from __future__ import annotations

from clifford_3plus2_d5.depth_hop_walsh.effective_hamiltonian_diagnostic import (
    effective_hamiltonian_diagnostic_payload,
)


def test_diagnostic_only_and_never_a_verdict() -> None:
    payload = effective_hamiltonian_diagnostic_payload()
    assert payload.diagnostic_only is True
    assert payload.final_verdict == "DEPTH_EFFECTIVE_HAMILTONIAN_DIAGNOSTIC"
    # It must not carry any depth pass/kill string.
    assert "SUPPORT_PASS" not in payload.final_verdict
    assert "SUPPORT_KILL" not in payload.final_verdict


def test_mirrors_strong_cp_result() -> None:
    payload = effective_hamiltonian_diagnostic_payload()
    assert payload.effective_a2u_is_zero is True  # no effective Lorentz theta term
    assert payload.effective_h1_nonzero is True  # nonzero degree-2 Lorentz correction
