"""Combined boundary-core audit.

The audit is intentionally conservative.  It confirms the exact transfer
factor, then asks whether an explicit unbroken residual ``S_3`` ``K_3`` tail can
produce

    K_nu = epsilon^2 P_u + P_b.

It cannot: ``K_nu`` splits the residual ``S_3`` doublet, while any unbroken
``S_3`` tail Schur complement remains singlet-plus-degenerate-doublet.  This is
a useful kill result, not a failure of the sidecar.  PMNS/CKM extensions should
remain parked until an explicit framed boundary model supplies the required
``S_3 -> S_2`` breaking dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import (
    finite_k3_tail_hamiltonian,
    k3_tail_self_energy,
    tail_is_s3_equivariant,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_s3_invariant,
    is_selected_s2_invariant,
    k_nu_operator,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    epsilon_fourth,
    transfer_verdict,
)


@dataclass(frozen=True)
class BoundaryCoreAuditPayload:
    """Verdict payload for the boundary-response core sidecar."""

    transfer_verdict: str
    symmetry_verdict: str
    tail_verdict: str
    final_verdict: str
    epsilon_value: sp.Expr
    epsilon_fourth_value: sp.Expr
    target_selected_s2_invariant: bool
    target_s3_invariant: bool
    tail_hamiltonian_s3_equivariant: bool
    tail_self_energy_s3_invariant: bool
    tail_matches_target_at_probe: bool
    pmns_ckm_parked: bool
    interpretation: str


def boundary_core_audit_payload() -> BoundaryCoreAuditPayload:
    """Return the combined transfer/symmetry/Schur-complement verdict."""

    z_probe = sp.Integer(3)
    target = k_nu_operator()
    tail_sigma = k3_tail_self_energy(z_probe, shells=1)
    h_q = finite_k3_tail_hamiltonian(shells=1)

    target_s2 = is_selected_s2_invariant(target)
    target_s3 = is_s3_invariant(target)
    tail_h_s3 = tail_is_s3_equivariant(h_q, shells=1)
    tail_sigma_s3 = is_s3_invariant(tail_sigma)
    tail_matches = matrix_equal(tail_sigma, target)

    if not target_s3 and target_s2:
        symmetry_verdict = "S3_KILL"
    else:
        symmetry_verdict = "SYMMETRY_UNRESOLVED"

    if tail_h_s3 and tail_sigma_s3 and not tail_matches:
        tail_verdict = "K3_TAIL_KILL"
    elif tail_matches:
        tail_verdict = "CORE_PASS"
    else:
        tail_verdict = "TAIL_UNRESOLVED"

    t_verdict = transfer_verdict()
    if t_verdict == "TRANSFER_PASS" and tail_verdict == "K3_TAIL_KILL":
        final_verdict = "BOUNDARY_CORE_KILL_UNBROKEN_K3"
        parked = True
        interpretation = (
            "The residual transfer recurrence gives epsilon = sqrt(2)-1, "
            "but the proposed K_nu = epsilon^2 P_u + P_b is only invariant "
            "under the selected-port S2, not full residual S3.  The explicit "
            "S3-equivariant finite K3 tail has an S3-invariant Schur "
            "self-energy and cannot equal K_nu.  PMNS/CKM boundary textures "
            "stay parked until an explicit framed H_Q,V derives the required "
            "S3 -> S2 doublet splitting."
        )
    elif t_verdict == "TRANSFER_PASS" and tail_verdict == "CORE_PASS":
        final_verdict = "BOUNDARY_CORE_PASS"
        parked = False
        interpretation = (
            "An explicit H_Q,V candidate produced the proposed K_nu target.  "
            "Boundary-scattering phenomenology may be reopened."
        )
    else:
        final_verdict = "BOUNDARY_CORE_UNRESOLVED"
        parked = True
        interpretation = (
            "The transfer, symmetry, or Schur-complement checks did not give a "
            "decisive pass/kill.  PMNS/CKM remain parked."
        )

    return BoundaryCoreAuditPayload(
        transfer_verdict=t_verdict,
        symmetry_verdict=symmetry_verdict,
        tail_verdict=tail_verdict,
        final_verdict=final_verdict,
        epsilon_value=epsilon(),
        epsilon_fourth_value=epsilon_fourth(),
        target_selected_s2_invariant=target_s2,
        target_s3_invariant=target_s3,
        tail_hamiltonian_s3_equivariant=tail_h_s3,
        tail_self_energy_s3_invariant=tail_sigma_s3,
        tail_matches_target_at_probe=tail_matches,
        pmns_ckm_parked=parked,
        interpretation=interpretation,
    )
