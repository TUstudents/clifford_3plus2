"""Phase KO-5 combined audit: aggregate KO-1..KO-4 into a single payload.

The Koide audit assesses whether the BCC body-diagonal Z₃ structure
of the program naturally predicts, permits, or forbids the empirical
Koide K = 2/3 relation for charged-lepton masses.

Final verdict: KOIDE CONSISTENT.

The Z₃-equivariant Yukawa locus L_Z3 = {(3|v_t|², (3/2)|v_o|²,
(3/2)|v_o|²)} intersects the Koide 45° cone at a 1-parameter family
(|v_t|/|v_o| = 3 + 2√2 ≈ 5.83) but does not lie entirely inside the
cone, and PDG charged-lepton masses are not in L_Z3 because of their
all-distinct (no-degeneracy) structure.

The audit's conclusion: the BCC body-diagonal Z₃ structure is
**compatible** with Koide but does not uniquely select PDG.  Three
distinct masses require Z₃-breaking input — most naturally via a
Higgs-VEV alignment that tilts the Yukawa off the equivariant
locus (Bold-B territory).
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.koide.bcc_z3_on_flavor import (
    BCCZ3OnFlavorPayload,
    bcc_z3_on_flavor_payload,
)
from clifford_3plus2_d5.koide.cone_locus_compatibility import (
    ConeLocusCompatibilityPayload,
    cone_locus_compatibility_payload,
)
from clifford_3plus2_d5.koide.koide_geometry import (
    KoideGeometryPayload,
    koide_geometry_payload,
)
from clifford_3plus2_d5.koide.yukawa_eigenvalue_locus import (
    YukawaLocusPayload,
    yukawa_locus_payload,
)


@dataclass(frozen=True)
class KoideAuditPayload:
    """Final combined verdict (KO-1..KO-4)."""

    geometry: KoideGeometryPayload
    bcc_z3_on_flavor: BCCZ3OnFlavorPayload
    yukawa_locus: YukawaLocusPayload
    cone_locus_compat: ConeLocusCompatibilityPayload
    final_verdict: str
    pdg_in_locus: bool
    verdict: str
    interpretation: str


def koide_audit_payload() -> KoideAuditPayload:
    """Run the combined Koide audit aggregating KO-1..KO-4."""

    geom = koide_geometry_payload()
    z3 = bcc_z3_on_flavor_payload()
    locus = yukawa_locus_payload()
    classifier = cone_locus_compatibility_payload()

    final = classifier.final_verdict
    pdg_in_loc = classifier.pdg_in_locus

    if final == "KOIDE CONSISTENT" and not pdg_in_loc:
        verdict = f"KOIDE AUDIT — {final} (PDG NOT IN LOCUS)"
        interpretation = (
            f"Phase KO-1 verified Koide empirically: K_PDG = "
            f"{float(geom.pdg_K):.6f} vs 2/3 = 0.666667, deviation "
            f"{float(geom.K_deviation_from_2_3):.2e} (~10⁻⁵).  Three "
            f"equivalent geometric forms (Koide K, 45° angle, |v_trace|² "
            f"= |v_traceless|² equipartition) agree.\n\n"
            f"Phase KO-2 verified the BCC body-diagonal Z₃ structure: "
            f"R = [[0,1,0],[0,0,1],[1,0,0]] fixes (1,1,1)/√3, commutes "
            f"with both trace and traceless projectors (these are the "
            f"Z₃-irrep projectors), and σ^x↔e / σ^y↔μ / σ^z↔τ is a "
            f"consistent identification convention.\n\n"
            f"Phase KO-3 built the Z₃-equivariant Yukawa from the BCC "
            f"body-diagonal Z₃ orbit (broken_triality-style pattern with "
            f"R replacing triality).  Y_ij = ⟨R^i v_*, R^j v_*⟩ is "
            f"circulant; eigenvalues are (3|v_t|², (3/2)|v_o|², "
            f"(3/2)|v_o|²) — TWO of three are ALWAYS degenerate by Z₃ "
            f"equivariance.\n\n"
            f"Phase KO-4 compared the Yukawa locus L_Z3 against the "
            f"Koide cone C.  L_Z3 ∩ C is non-empty (1-parameter family at "
            f"|v_t|/|v_o| = 3 + 2√2 ≈ 5.83), but L_Z3 ⊄ C.  PDG ∉ L_Z3 "
            f"because PDG has all distinct masses while L_Z3 has 2-fold "
            f"degeneracy.\n\n"
            f"Verdict: KOIDE CONSISTENT.  The BCC body-diagonal Z₃ "
            f"structure admits Koide-satisfying Yukawa solutions but "
            f"does not uniquely select them.  The apparent Koide↔BCC "
            f"coincidence is NOT spurious — the cone is naturally "
            f"associated with the Z₃-trivial direction — but the "
            f"program does not PREDICT K = 2/3 from carrier structure "
            f"alone.  Three distinct PDG masses require Z₃-breaking "
            f"input (Bold-B dynamical Higgs VEV alignment is the "
            f"natural follow-up)."
        )
    else:
        verdict = f"KOIDE AUDIT — {final}"
        interpretation = (
            f"Geometry: {geom.verdict}.  BCC Z₃: {z3.verdict}.  "
            f"Yukawa locus: {locus.verdict}.  Cone-locus: "
            f"{classifier.final_verdict}.  PDG in locus: {pdg_in_loc}."
        )

    return KoideAuditPayload(
        geometry=geom,
        bcc_z3_on_flavor=z3,
        yukawa_locus=locus,
        cone_locus_compat=classifier,
        final_verdict=final,
        pdg_in_locus=pdg_in_loc,
        verdict=verdict,
        interpretation=interpretation,
    )
