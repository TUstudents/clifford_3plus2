"""Phase SC-6 combined audit: aggregate SC-1..SC-3 + SC-5 (structural argument).

Phase SC-4 (direct lattice topological-charge computation) is the
direct-computation confirmation of the structural verdict; it is
deferred to a follow-up session.  The structural argument
encapsulated by SC-1, SC-2, SC-3, SC-5 is independently load-bearing:

- SC-1: degree-3 cubic-harmonic decomposition includes A_{2u}, the
  θ_QCD-term irrep.
- SC-2: BCC lattice + BB walk are centrosymmetric; the parity
  selection rule applies.
- SC-3: H^(2) is 100% in T_{1u} with zero A_{2u} content.  H^(1)
  was previously verified in cp/ to be 100% in T_{2g} (g-irrep).
- SC-5: tr(γ^5 H^(1)) = 0 and tr(γ^5 H^(2)) = 0 — H^(1), H^(2)
  are individually vector (not axial), inducing no direct chiral
  rotation on the fermion measure → no θ̄ shift at O(ε) or O(ε²).

Verdict:

    STRONG-CP TRIVIAL at O(ε) and O(ε²) by structural lattice
    symmetry.  Any higher-order contribution is ε^n-suppressed
    below the neutron-EDM bound 10⁻¹⁰.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.strongcp.bcc_centrosymmetry import (
    BCCCentrosymmetryAuditPayload,
    bcc_centrosymmetry_payload,
)
from clifford_3plus2_d5.strongcp.chiral_anomaly_check import (
    ChiralAnomalyCheckPayload,
    chiral_anomaly_check_payload,
)
from clifford_3plus2_d5.strongcp.cubic_harmonics_degree3 import (
    deg3_projectors_satisfy_idempotent_orthogonal_complete,
)
from clifford_3plus2_d5.strongcp.higher_order_parity import (
    HigherOrderParityAuditPayload,
    higher_order_parity_payload,
)
from clifford_3plus2_d5.strongcp.theta_bar_constraint import (
    ThetaBarConstraintPayload,
    theta_bar_constraint_payload,
)


@dataclass(frozen=True)
class StrongCPAuditPayload:
    """Final combined verdict (SC-1..SC-3 + SC-5; SC-4 deferred)."""

    sc1_degree3_harmonics_consistent: bool
    sc2_centrosymmetry: BCCCentrosymmetryAuditPayload
    sc3_higher_order_parity: HigherOrderParityAuditPayload
    sc5_chiral_anomaly: ChiralAnomalyCheckPayload
    theta_bar_constraint: ThetaBarConstraintPayload
    sc4_topological_charge_deferred: bool
    structural_argument_complete: bool
    final_verdict: str
    interpretation: str


def strong_cp_audit_payload() -> StrongCPAuditPayload:
    """Run the combined Strong-CP audit aggregating SC-1..SC-3 + SC-5."""

    sc1 = deg3_projectors_satisfy_idempotent_orthogonal_complete()
    sc2 = bcc_centrosymmetry_payload()
    sc3 = higher_order_parity_payload()
    sc5 = chiral_anomaly_check_payload()
    theta = theta_bar_constraint_payload()

    structural_complete = (
        sc1
        and sc2.all_centrosymmetric
        and sc3.selection_rule_applies
        and sc5.no_direct_theta_shift_at_o_eps_eps2
    )

    if structural_complete:
        final = "STRONG-CP TRIVIAL at O(ε) and O(ε²); SAFE at higher orders"
        interpretation = (
            "**Structural argument complete (SC-1, SC-2, SC-3, SC-5).**  "
            "BCC lattice and BB walk are centrosymmetric; H^(1) lives in "
            "T_{2g} (g-irrep, verified in cp/); H^(2) lives in T_{1u} with "
            "ZERO A_{2u} content (the θ_QCD-term irrep).  By the cubic-group "
            "parity selection rule (g × g = g, u × u = g, g × u = u), no "
            "product of H^(1) and H^(2) populates A_{2u} in the effective "
            "action's θ-channel.  Independently, ``tr(γ^5 H^(n)) = 0`` for "
            "n = 1, 2, so no chiral-rotation contribution to θ̄ at these "
            "orders.\n\n"
            f"{theta.interpretation}\n\n"
            "Phase SC-4 (direct lattice topological-charge density "
            "computation Q(x) = (1/32π²) ε^{μνρσ} tr F_{μν} F_{ρσ} on BCC "
            "plaquettes) is the direct-computation confirmation; deferred "
            "to a follow-up session.  The structural verdict here is "
            "independently load-bearing."
        )
    else:
        final = "STRUCTURAL ARGUMENT INCOMPLETE"
        interpretation = (
            f"sc1: {sc1}; sc2: {sc2.all_centrosymmetric}; "
            f"sc3: {sc3.selection_rule_applies}; "
            f"sc5: {sc5.no_direct_theta_shift_at_o_eps_eps2}.  "
            "Investigate before declaring a verdict."
        )

    return StrongCPAuditPayload(
        sc1_degree3_harmonics_consistent=sc1,
        sc2_centrosymmetry=sc2,
        sc3_higher_order_parity=sc3,
        sc5_chiral_anomaly=sc5,
        theta_bar_constraint=theta,
        sc4_topological_charge_deferred=True,
        structural_argument_complete=structural_complete,
        final_verdict=final,
        interpretation=interpretation,
    )
