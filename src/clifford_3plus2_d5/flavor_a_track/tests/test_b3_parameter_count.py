"""Tests for B3 — parameter count vs observables."""

from __future__ import annotations

from clifford_3plus2_d5.flavor_a_track.b3_parameter_count import (
    N_TEXTURE_OBSERVABLES,
    parameter_count_audit_payload,
    parameter_count_verdict,
)


def test_real_count_is_predictive() -> None:
    payload = parameter_count_audit_payload()
    assert payload.final_verdict == "TEXTURE_PREDICTIVE"
    assert payload.n_free == 4
    assert payload.n_observables == N_TEXTURE_OBSERVABLES == 8
    assert payload.surplus == 4


def test_verdict_logic_predictive_when_surplus_positive() -> None:
    assert parameter_count_verdict(4, 8) == "TEXTURE_PREDICTIVE"


def test_verdict_kills_when_inputs_match_or_exceed_observables() -> None:
    # Decisive negative control: as many free knobs as data -> numerology.
    assert parameter_count_verdict(8, 8) == "TEXTURE_NUMEROLOGY_KILL"
    assert parameter_count_verdict(9, 8) == "TEXTURE_NUMEROLOGY_KILL"
