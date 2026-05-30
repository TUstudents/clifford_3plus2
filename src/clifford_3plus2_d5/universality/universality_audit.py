"""U4 — combined cross-sector universality verdict (roadmap gate A2).

Aggregates U1 (shared transfer invariant), U2 (sector difference = color label),
and U3 (couplings are SM quantum-number projections) into one verdict.

This gate tests the *necessary conditions* for the strong universality claim and
can return a clean KILL.  It does NOT confirm universality: the full numerical
reproduction of every ``Sigma_f`` from one ``H_Q`` on lepton's chiral-16 carrier
is the flavor program (roadmap A3) and is recorded as the remaining declared
input.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.universality.u1_shared_transfer import (
    SharedTransferAuditPayload,
    shared_transfer_audit_payload,
)
from clifford_3plus2_d5.universality.u2_color_label_partition import (
    ColorLabelPartitionAuditPayload,
    color_label_partition_audit_payload,
)
from clifford_3plus2_d5.universality.u3_coupling_catalog import (
    CouplingCatalogAuditPayload,
    coupling_catalog_audit_payload,
)

REMAINING_DECLARED_INPUTS = ("unified_H_Q_on_chiral16_reproduces_all_Sigma_f",)


@dataclass(frozen=True)
class UniversalityAuditPayload:
    """Combined verdict payload for the cross-sector universality gate."""

    final_verdict: str
    shared_transfer: SharedTransferAuditPayload
    color_label_partition: ColorLabelPartitionAuditPayload
    coupling_catalog: CouplingCatalogAuditPayload
    necessary_conditions_pass: bool
    full_reproduction_confirmed: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def universality_audit_payload() -> UniversalityAuditPayload:
    """Return the combined U1-U3 universality verdict with the A3 deferral."""

    u1 = shared_transfer_audit_payload()
    u2 = color_label_partition_audit_payload()
    u3 = coupling_catalog_audit_payload()

    necessary_pass = (
        u1.final_verdict == "SHARED_TRANSFER_INVARIANT"
        and u2.final_verdict == "SECTOR_DIFFERENCE_IS_COLOR_LABEL"
        and u3.final_verdict == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"
    )

    if necessary_pass:
        final_verdict = "UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS"
        interpretation = (
            "Necessary conditions for one-boundary universality hold: (U1) all "
            "sectors share the single residual K3 transfer root sqrt(2)-1; (U2) "
            "the quark and lepton shells differ by exactly the three color "
            "ports, with the same 1 + 2 non-color core; (U3) the per-sector "
            "couplings V_f are SM quantum-number projections (quark color-"
            "triplet, lepton color-singlet). This does NOT confirm "
            "universality: the full numerical reproduction of every Sigma_f "
            "from one H_Q on lepton's chiral-16 carrier remains the flavor "
            "program (roadmap A3), recorded as the one remaining declared input."
        )
    else:
        final_verdict = "SECTOR_DEPENDENT_BOUNDARY_KILL"
        interpretation = (
            "A necessary condition failed: "
            f"U1={u1.final_verdict}; U2={u2.final_verdict}; "
            f"U3={u3.final_verdict}. The sectors are not projections of one "
            "boundary; the strong universality claim is not supported."
        )

    return UniversalityAuditPayload(
        final_verdict=final_verdict,
        shared_transfer=u1,
        color_label_partition=u2,
        coupling_catalog=u3,
        necessary_conditions_pass=necessary_pass,
        full_reproduction_confirmed=False,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS,
        interpretation=interpretation,
    )
