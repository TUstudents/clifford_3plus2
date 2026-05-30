"""Tests for B2 — free-input enumeration."""

from __future__ import annotations

from clifford_3plus2_d5.texture_provenance.b2_free_inputs import (
    free_inputs,
    free_inputs_audit_payload,
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
    assert all(f.source for f in payload.free_inputs)
