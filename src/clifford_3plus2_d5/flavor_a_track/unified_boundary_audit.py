"""A3-4 — combined unified-transfer-boundary verdict (roadmap gate A3a).

Aggregates A3-1 (one common H_Q; lepton Sigma recovered), A3-2 (quark transfer is
the same-chain Schur structure), and A3-3 (sector structure lives in V_f).

Scope: A3a unifies the transfer boundary ``H_Q`` and shows the sector-specific
structure is in ``V_f``.  It does NOT derive the quark depths {0,2,6} or the
Clebsch / coin factors from the chiral-16 geometry — those remain inputs whose
derivation is A3b, recorded as the remaining declared inputs.  The gate can KILL
but does not prove the full flavor pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.flavor_a_track.a31_common_hq import (
    CommonHqAuditPayload,
    common_hq_audit_payload,
)
from clifford_3plus2_d5.flavor_a_track.a32_quark_transfer_schur import (
    QuarkTransferSchurAuditPayload,
    quark_transfer_schur_audit_payload,
)
from clifford_3plus2_d5.flavor_a_track.a33_coupling_structure import (
    CouplingStructureAuditPayload,
    coupling_structure_audit_payload,
)

REMAINING_DECLARED_INPUTS = (
    "quark_family_depths_0_2_6_derived",
    "clebsch_and_coin_factors_derived_from_chiral16",
)


@dataclass(frozen=True)
class UnifiedBoundaryAuditPayload:
    """Combined verdict payload for the A3a unified-transfer-boundary gate."""

    final_verdict: str
    common_hq: CommonHqAuditPayload
    quark_transfer_schur: QuarkTransferSchurAuditPayload
    coupling_structure: CouplingStructureAuditPayload
    transfer_boundary_unified: bool
    full_flavor_pattern_proven: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def unified_boundary_audit_payload() -> UnifiedBoundaryAuditPayload:
    """Return the combined A3a verdict with the A3b deferral recorded."""

    a31 = common_hq_audit_payload()
    a32 = quark_transfer_schur_audit_payload()
    a33 = coupling_structure_audit_payload()

    unified = (
        a31.final_verdict == "LEPTON_SIGMA_FROM_COMMON_HQ"
        and a32.final_verdict == "QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR"
        and a33.final_verdict == "SECTOR_STRUCTURE_IN_COUPLINGS"
    )

    if unified:
        final_verdict = "UNIFIED_TRANSFER_BOUNDARY_PASS"
        interpretation = (
            "One common H_Q (the semi-infinite sterile chain) yields the lepton "
            "core Sigma = epsilon^2 P_u + P_b as its Schur complement (A3-1) and "
            "the quark transfer hierarchy epsilon^2, epsilon^4, epsilon^6 as the "
            "same-chain Schur/Green structure at depths {0,2,6} (A3-2); the "
            "sector-specific color (C_F=4/3), BCC (sqrt(2), 1/sqrt(2)), and coin "
            "(atan(sqrt(5))) structure lives in V_f (A3-3). The transfer "
            "boundary is unified. This does NOT prove the full flavor pattern: "
            "deriving the quark depths {0,2,6} and the Clebsch/coin factors from "
            "the chiral-16 geometry remains A3b, recorded as the remaining "
            "declared inputs."
        )
    else:
        final_verdict = "TRANSFER_NOT_UNIFIABLE_KILL"
        interpretation = (
            "A unification condition failed: "
            f"A3-1={a31.final_verdict}; A3-2={a32.final_verdict}; "
            f"A3-3={a33.final_verdict}. The quark and lepton sectors are not "
            "Schur complements of one common H_Q; the transfer boundary does "
            "not unify."
        )

    return UnifiedBoundaryAuditPayload(
        final_verdict=final_verdict,
        common_hq=a31,
        quark_transfer_schur=a32,
        coupling_structure=a33,
        transfer_boundary_unified=unified,
        full_flavor_pattern_proven=False,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS,
        interpretation=interpretation,
    )
