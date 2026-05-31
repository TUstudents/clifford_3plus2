"""Tests for the V43 selector-sector closure ledger."""

from __future__ import annotations

from clifford_3plus2_d5.boundary_response.vacuum_selector_closure import (
    INTERMEDIATE_SELECTOR_AXIOMS,
    SELECTOR_CLOSURE_EXPECTED_VERDICTS,
    selector_closure_prerequisite_verdicts,
    selector_closure_prerequisites_pass,
    vacuum_selector_closure_audit_payload,
)


def test_selector_closure_prerequisite_verdicts_pass() -> None:
    assert selector_closure_prerequisite_verdicts() == SELECTOR_CLOSURE_EXPECTED_VERDICTS
    assert selector_closure_prerequisites_pass()


def test_selector_closure_has_exactly_one_intermediate_axiom() -> None:
    assert INTERMEDIATE_SELECTOR_AXIOMS == (
        "positive_quartic_backreaction_bounds_selector_radius",
    )


def test_vacuum_selector_closure_payload_reports_pass() -> None:
    payload = vacuum_selector_closure_audit_payload()

    assert payload.final_verdict == "VACUUM_SELECTOR_CLOSED_WITH_QUARTIC_AXIOM_PASS"
    assert payload.prerequisite_verdicts == SELECTOR_CLOSURE_EXPECTED_VERDICTS
    assert payload.prerequisites_pass
    assert payload.remaining_intermediate_axioms == INTERMEDIATE_SELECTOR_AXIOMS
    assert payload.remaining_intermediate_axioms == (
        "positive_quartic_backreaction_bounds_selector_radius",
    )
    assert not payload.quartic_microscopically_derived
    assert payload.closed_for_polishing
