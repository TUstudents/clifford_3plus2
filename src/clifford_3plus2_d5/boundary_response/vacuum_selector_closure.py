"""V43 closed selector-sector ledger with one explicit quartic axiom.

V35--V42 now derive the selector branch, the radial origin instability, and the
finite-radius closure from a positive quartic backreaction.  V43 records the
thread as closed modulo one named intermediate axiom:

    positive_quartic_backreaction_bounds_selector_radius.

This module is intentionally a ledger.  It does not hide the remaining axiom,
and it does not claim a microscopic derivation of the positive quartic
coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from clifford_3plus2_d5.boundary_response.vacuum_selector_bb_induced_breaking import (
    REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING,
    bb_induced_radial_breaking_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_chiral_bb import (
    chiral_bb_selector_sign_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_filled_band_potential import (
    filled_band_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_microscopic_potential import (
    microscopic_selector_potential_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_no_go import (
    free_bb_radial_stabilization_audit_payload,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_radial_theorem import (
    analytic_radial_breaking_theorem_audit_payload,
)

SELECTOR_CLOSURE_EXPECTED_VERDICTS = (
    ("V35", "CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS"),
    ("V36", "CHIRAL_BB_BRANCH_SELECTION_PASS"),
    ("V37", "MICROSCOPIC_FILLED_BAND_SELECTOR_POTENTIAL_PASS"),
    ("V38", "FREE_BB_RADIAL_STABILIZATION_NO_GO_PASS"),
    ("V41", "BB_INDUCED_RADIAL_BREAKING_PASS"),
    ("V42", "ANALYTIC_RADIAL_BREAKING_THEOREM_PASS"),
)

INTERMEDIATE_SELECTOR_AXIOMS = REMAINING_DECLARED_INPUTS_AFTER_BB_INDUCED_BREAKING


@cache
def selector_closure_prerequisite_verdicts() -> tuple[tuple[str, str], ...]:
    """Return the selector-chain prerequisite verdicts."""

    return (
        ("V35", chiral_bb_selector_sign_audit_payload().final_verdict),
        ("V36", filled_band_selector_potential_audit_payload().final_verdict),
        ("V37", microscopic_selector_potential_audit_payload().final_verdict),
        ("V38", free_bb_radial_stabilization_audit_payload().final_verdict),
        ("V41", bb_induced_radial_breaking_audit_payload().final_verdict),
        ("V42", analytic_radial_breaking_theorem_audit_payload().final_verdict),
    )


def selector_closure_prerequisites_pass() -> bool:
    """Return true when every selector-chain prerequisite has the expected verdict."""

    return selector_closure_prerequisite_verdicts() == SELECTOR_CLOSURE_EXPECTED_VERDICTS


@dataclass(frozen=True)
class VacuumSelectorClosureAuditPayload:
    """Verdict payload for the V43 closed selector-sector ledger."""

    final_verdict: str
    prerequisite_verdicts: tuple[tuple[str, str], ...]
    prerequisites_pass: bool
    remaining_intermediate_axioms: tuple[str, ...]
    quartic_microscopically_derived: bool
    closed_for_polishing: bool
    interpretation: str


@cache
def vacuum_selector_closure_audit_payload() -> VacuumSelectorClosureAuditPayload:
    """Return the V43 selector-sector closure verdict."""

    verdicts = selector_closure_prerequisite_verdicts()
    prerequisites = verdicts == SELECTOR_CLOSURE_EXPECTED_VERDICTS
    quartic_derived = False
    closed = prerequisites and INTERMEDIATE_SELECTOR_AXIOMS == (
        "positive_quartic_backreaction_bounds_selector_radius",
    )

    if closed:
        final_verdict = "VACUUM_SELECTOR_CLOSED_WITH_QUARTIC_AXIOM_PASS"
        interpretation = (
            "The selector branch, radial origin instability, and finite-radius "
            "closure are derived through V42.  The sidecar is closed for "
            "polishing with exactly one intermediate axiom: positive quartic "
            "backreaction bounds the selector radius.  The quartic coefficient "
            "is not yet microscopically derived."
        )
    else:
        final_verdict = "VACUUM_SELECTOR_CLOSURE_KILL"
        interpretation = (
            "The selector closure ledger failed at least one prerequisite "
            "verdict or did not reduce to the intended single quartic axiom."
        )

    return VacuumSelectorClosureAuditPayload(
        final_verdict=final_verdict,
        prerequisite_verdicts=verdicts,
        prerequisites_pass=prerequisites,
        remaining_intermediate_axioms=INTERMEDIATE_SELECTOR_AXIOMS,
        quartic_microscopically_derived=quartic_derived,
        closed_for_polishing=closed,
        interpretation=interpretation,
    )
