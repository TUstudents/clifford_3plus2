"""Tests for B4 — combined texture-provenance verdict + generation deferral."""

from __future__ import annotations

from clifford_3plus2_d5.texture_provenance.texture_provenance_audit import (
    REMAINING_DECLARED_INPUTS,
    texture_provenance_audit_payload,
)


def test_combined_verdict() -> None:
    payload = texture_provenance_audit_payload()
    assert payload.final_verdict == "TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT"
    assert payload.texture_is_predictive


def test_subpayloads() -> None:
    payload = texture_provenance_audit_payload()
    assert payload.derived_factors.final_verdict == "DERIVED_FACTORS_CATALOGUED"
    assert payload.free_inputs.final_verdict == "FREE_INPUTS_ENUMERATED"
    assert payload.parameter_count.final_verdict == "TEXTURE_PREDICTIVE"


def test_hierarchy_derivation_deferred() -> None:
    payload = texture_provenance_audit_payload()
    assert payload.full_hierarchy_derived is False
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS
    assert payload.remaining_declared_inputs == ("generation_depth_embedding_derived",)


def test_predictive_means_fewer_inputs_than_observables() -> None:
    payload = texture_provenance_audit_payload()
    assert payload.free_inputs.n_free < payload.parameter_count.n_observables
