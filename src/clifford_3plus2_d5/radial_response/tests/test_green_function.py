"""Tests for the Feshbach recirculation gate."""

import sympy as sp

from clifford_3plus2_d5.radial_response.green_function import (
    boundary_recirculation_series_terms,
    example_boundary_system,
    feshbach_self_energy,
    full_resolvent_p_block,
    green_function_payload,
    p_block_resolvent,
)


def test_feshbach_self_energy_is_boundary_return_amplitude() -> None:
    z, _h_p, h_q, v = example_boundary_system()
    a = sp.Symbol("a")
    g = sp.Symbol("g")
    assert feshbach_self_energy(z, h_q, v) == sp.Matrix([[g**2 / (z - a)]])


def test_schur_p_block_matches_full_resolvent_p_block() -> None:
    z, h_p, h_q, v = example_boundary_system()
    assert sp.simplify(p_block_resolvent(z, h_p, h_q, v) - full_resolvent_p_block(z, h_p, h_q, v)) == sp.zeros(1, 1)


def test_recirculation_series_terms_are_repeated_boundary_returns() -> None:
    z, _h_p, h_q, v = example_boundary_system()
    a = sp.Symbol("a")
    g = sp.Symbol("g")
    assert boundary_recirculation_series_terms(z, h_q, v, 2) == (
        sp.Matrix([[g**2 / z]]),
        sp.Matrix([[a * g**2 / z**2]]),
        sp.Matrix([[a**2 * g**2 / z**3]]),
    )


def test_green_function_payload_passes() -> None:
    payload = green_function_payload()
    assert payload.final_verdict == "MASS_AS_BOUNDARY_RECIRCULATION_PASS"
    assert payload.schur_matches_full_resolvent
