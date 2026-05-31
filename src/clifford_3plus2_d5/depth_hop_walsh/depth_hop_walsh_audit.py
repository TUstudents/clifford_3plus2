"""Aggregate — the depth-hop-Walsh probe (Claim A).

Final verdict = W2's combined verdict. W1 (Walsh decomposition + covariance) and W3
(effective-Hamiltonian diagnostic) are carried for context; W3 is diagnostic_only
and CANNOT alter the final verdict.

Scope: a combined PASS establishes Claim A only — primitive angular 0,1,3
coefficient-Walsh support in the raw BCC Weyl hop source. It does NOT derive the
depths (the bridge d_radial = 2*Walsh-degree, Claim B, and the
sqrt(5) = sqrt(2_BCC + 3_color) compatibility, Claim C, remain unproven).
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.depth_hop_walsh.covariant_o_decomposition import (
    CovariantODecompositionPayload,
    covariant_o_decomposition_payload,
)
from clifford_3plus2_d5.depth_hop_walsh.effective_hamiltonian_diagnostic import (
    EffectiveHamiltonianDiagnosticPayload,
    effective_hamiltonian_diagnostic_payload,
)
from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_decomposition import (
    HopWalshDecompositionPayload,
    hop_walsh_decomposition_payload,
)
from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_support_audit import (
    HopWalshSupportPayload,
    hop_walsh_support_audit_payload,
)
from clifford_3plus2_d5.depth_hop_walsh.s3_schur_obstruction import (
    S3SchurObstructionPayload,
    s3_schur_obstruction_payload,
)

# Claim A is the only claim this probe can move; the depths are not derived here.
REMAINING_DECLARED_INPUTS = (
    "radial_depth_equals_twice_walsh_degree",  # Claim B (the bridge)
    "cube_geometry_compatible_with_sqrt5_coin",  # Claim C
    "generation_depth_embedding_derived",  # the standing flavor input
)


@dataclass(frozen=True)
class DepthHopWalshAuditPayload:
    """Combined payload.

    W2 (coefficient-Walsh) is the named primary verdict; W4 (covariant
    O-decomposition) is the escape-hatch resolution; W3 is diagnostic-only. Claim A
    is killed iff BOTH W2 and W4 kill — which they do.
    """

    final_verdict: str
    walsh_right: HopWalshDecompositionPayload
    walsh_left: HopWalshDecompositionPayload
    support: HopWalshSupportPayload
    covariant: CovariantODecompositionPayload
    schur_obstruction: S3SchurObstructionPayload
    diagnostic: EffectiveHamiltonianDiagnosticPayload
    establishes_claim_a: bool
    claim_a_killed_both_lenses: bool
    depth_hierarchy_requires_s3_breaking: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def depth_hop_walsh_audit_payload() -> DepthHopWalshAuditPayload:
    """Return the combined depth-hop-Walsh verdict (Claim A probe)."""

    walsh_right = hop_walsh_decomposition_payload("right")
    walsh_left = hop_walsh_decomposition_payload("left")
    support = hop_walsh_support_audit_payload()
    covariant = covariant_o_decomposition_payload()
    schur = s3_schur_obstruction_payload()
    diagnostic = effective_hamiltonian_diagnostic_payload()

    coefficient_pass = support.final_verdict == "DEPTH_HOP_WALSH_SUPPORT_PASS"
    covariant_pass = covariant.final_verdict == "COVARIANT_SUPPORT_PASS"
    establishes_claim_a = coefficient_pass and covariant_pass
    killed_both = (not coefficient_pass) and (not covariant_pass)
    requires_s3_breaking = schur.final_verdict == "DEPTH_HIERARCHY_REQUIRES_S3_BREAKING"

    interpretation = (
        f"Primary (W2, coefficient-Walsh): {support.final_verdict}. "
        f"Escape-hatch (W4, covariant O-decomposition): {covariant.final_verdict} "
        f"(right norms A1/A2/E/T1/T2 = "
        f"{covariant.right_norms['A1']}/{covariant.right_norms['A2']}/"
        f"{covariant.right_norms['E']}/{covariant.right_norms['T1']}/"
        f"{covariant.right_norms['T2']}). The coefficient-Walsh T2g reassembles to "
        "covariant T2=0 (the escape hatch's mechanism is real), but a forbidden E "
        "quadrupole is present and T1 is absent — so Claim A is killed under BOTH "
        f"lenses. W1 covariance check = {walsh_right.covariance_check} (lattice "
        f"symbol not C3-covariant). W3 ({diagnostic.final_verdict}) is "
        f"diagnostic-only. Core obstruction (W5, {schur.final_verdict}): by Schur "
        "on the residual 3 = 1 + 2, any S3-invariant depth operator has spectrum "
        "{alpha, beta, beta} (K3 Laplacian -> {0,3,3} -> doubled {0,6}), so the "
        "three distinct depths {0,2,6} are necessarily an S3-breaking spurion. "
        "Deriving {0,2,6} == deriving the family-symmetry-breaking spurion (the "
        "closed-negative generation problem). The depth embedding remains a "
        "declared free input."
    )

    return DepthHopWalshAuditPayload(
        final_verdict=support.final_verdict,
        walsh_right=walsh_right,
        walsh_left=walsh_left,
        support=support,
        covariant=covariant,
        schur_obstruction=schur,
        diagnostic=diagnostic,
        establishes_claim_a=establishes_claim_a,
        claim_a_killed_both_lenses=killed_both,
        depth_hierarchy_requires_s3_breaking=requires_s3_breaking,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS,
        interpretation=interpretation,
    )
