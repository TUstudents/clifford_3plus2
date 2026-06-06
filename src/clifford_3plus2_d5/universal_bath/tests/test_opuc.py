"""Tests for CMV/OPUC free-tail helpers."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.opuc import (
    free_verblunsky_tail,
    is_free_verblunsky_tail,
    opuc_free_tail_payload,
    schur_algorithm_step,
)


def test_free_verblunsky_tail_is_zero() -> None:
    tail = free_verblunsky_tail(5)
    assert tail == (0, 0, 0, 0, 0)
    assert is_free_verblunsky_tail(tail)


def test_nonzero_head_can_have_free_remainder() -> None:
    payload = opuc_free_tail_payload()
    assert payload.all_verblunsky_zero
    assert payload.free_remainder_after_head
    assert payload.nonzero_head_then_free_tail[:2] != (0, 0)
    assert is_free_verblunsky_tail(payload.nonzero_head_then_free_tail[2:])


def test_schur_algorithm_step_for_free_zero_function() -> None:
    zeta = sp.Symbol("zeta")
    assert schur_algorithm_step(zeta, sp.Integer(0), sp.Integer(0)) == 0

