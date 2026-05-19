"""Tests for ``internal_triviality.py``."""

from __future__ import annotations

from clifford_3plus2_d5.topology.internal_triviality import (
    chiral16_dimension,
    chiral16_is_z3_trivial,
    chiral16_real_dimension,
    chiral16_z3_decomposition,
    internal_z3_triviality_payload,
    spatial_z3_action_on_internal,
    three_generations_dimensional_check,
)


def test_chiral16_dimensions() -> None:
    assert chiral16_dimension() == 16
    assert chiral16_real_dimension() == 32


def test_spatial_z3_action_is_identity() -> None:
    import sympy as sp

    action = spatial_z3_action_on_internal()
    assert action == sp.eye(32)


def test_chiral16_is_z3_trivial() -> None:
    assert chiral16_is_z3_trivial()


def test_decomposition_is_all_trivial_character() -> None:
    decomposition = chiral16_z3_decomposition()
    assert decomposition["trivial (chi = 1)"] == 16
    assert decomposition["omega (chi = e^{2 pi i / 3})"] == 0
    assert decomposition["omega^2 (chi = e^{-2 pi i / 3})"] == 0


def test_three_generations_does_not_fit_in_trivial_character() -> None:
    check = three_generations_dimensional_check()
    assert check["trivial_char_actual"] == 16
    assert check["omega_char_actual"] == 0
    assert check["omega2_char_actual"] == 0


def test_audit_payload_confirms_kill() -> None:
    payload = internal_z3_triviality_payload()
    assert payload.chiral16_complex_dim == 16
    assert payload.chiral16_real_dim == 32
    assert payload.z3_acts_trivially
    assert not payload.can_yield_three_generations
    assert "TOPOLOGY KILL" in payload.verdict
    assert "Z_3-trivial" in payload.verdict
