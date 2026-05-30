"""Tests for B2 — free-input enumeration."""

from __future__ import annotations

from clifford_3plus2_d5.flavor_a_track.b2_free_inputs import (
    free_inputs,
    free_inputs_audit_payload,
    free_inputs_verdict,
    quark_depths_are_even_and_additive,
)


def test_four_free_inputs_enumerated() -> None:
    inputs = free_inputs()
    names = {f.name for f in inputs}
    assert names == {
        "quark_depth_embedding",
        "charged_lepton_leakage_depth",
        "ergodicity_prior_r_equals_1",
        "cp_phase_branch",
    }


def test_quark_depths_even_and_additive() -> None:
    assert quark_depths_are_even_and_additive()


def test_payload() -> None:
    payload = free_inputs_audit_payload()
    assert payload.final_verdict == "FREE_INPUTS_ENUMERATED"
    assert payload.n_free == 4
    assert payload.quark_depths_even_additive
    assert all(f.source for f in payload.free_inputs)


def test_verdict_kills_when_enumeration_does_not_stand() -> None:
    # Decisive negative control: the headline fix. The enumeration only stands
    # when the depths have the even+additive structure AND the count is the
    # declared four. A violation of either forces the KILL string (previously
    # this gate could never fail).
    assert free_inputs_verdict(True, 4) == "FREE_INPUTS_ENUMERATED"
    assert free_inputs_verdict(False, 4) == "FREE_INPUTS_NOT_ENUMERATED_KILL"
    assert free_inputs_verdict(True, 5) == "FREE_INPUTS_NOT_ENUMERATED_KILL"
    assert free_inputs_verdict(False, 3) == "FREE_INPUTS_NOT_ENUMERATED_KILL"
