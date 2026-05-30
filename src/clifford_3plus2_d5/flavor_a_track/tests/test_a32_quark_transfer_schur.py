"""Tests for A3-2 — quark transfer is the same-chain Schur structure."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.a32_quark_transfer_schur import (
    finite_shell_green_ratios_converge,
    odd_depth_control_distinct,
    quark_transfer_from_common_chain,
    quark_transfer_matches_common_chain,
    quark_transfer_schur_audit_payload,
    scaled_chain_control_distinct,
)


def test_quark_amplitudes_are_common_chain_powers() -> None:
    predicted = quark_transfer_from_common_chain()
    eps = sp.sqrt(2) - 1
    assert sp.simplify(predicted[(1, 2)] - eps ** 2) == 0
    assert sp.simplify(predicted[(2, 3)] - eps ** 4) == 0
    assert sp.simplify(predicted[(1, 3)] - eps ** 6) == 0
    assert quark_transfer_matches_common_chain()


def test_finite_shell_green_ratios_converge() -> None:
    assert finite_shell_green_ratios_converge()


def test_negative_controls_reject() -> None:
    # The gate must be able to fail: odd depths and a different chain do not match.
    assert odd_depth_control_distinct()
    assert scaled_chain_control_distinct()


def test_payload_pass() -> None:
    payload = quark_transfer_schur_audit_payload()
    assert payload.final_verdict == "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR"
    assert payload.transfer_matches_common_chain
    assert payload.finite_shell_converges
    assert payload.odd_depth_control_distinct
    assert payload.scaled_chain_control_distinct
