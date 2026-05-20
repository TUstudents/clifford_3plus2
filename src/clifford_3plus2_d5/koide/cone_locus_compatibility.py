"""Phase KO-4: cone vs. Z₃-equivariant locus comparison.

Compares the Yukawa-eigenvalue locus from Phase KO-3 against the
Koide 45° cone (the K = 2/3 condition on the mass-vector).

From KO-3, the Z₃-equivariant locus is the 2-parameter family

    L_Z3  =  { ((3 |v_t|²), (3/2)|v_o|², (3/2)|v_o|²)  :  |v_t|, |v_o| ∈ ℝ₊ }

parametrized by (|v_t|, |v_o|).  Every triple in L_Z3 has 2-fold
degeneracy.  The Koide cone

    C  =  { (m_1, m_2, m_3) ∈ ℝ³₊  :  K(m) = 2/3 }

is a 2-dim sub-variety of ℝ³₊.  Their intersection is

    L_Z3 ∩ C  =  { ((3 |v_t|²), (3/2)|v_o|², (3/2)|v_o|²)
                   :  |v_t|/|v_o| = 3 + 2√2 }

a **1-parameter family** on the cone (parametrized by overall scale
|v_o|).  This intersection is non-empty but L_Z3 ⊄ C: most points
in L_Z3 are NOT on the cone.

PDG observation: PDG charged-lepton masses are all DISTINCT
(no degeneracy), so PDG ∉ L_Z3.  Therefore PDG ∉ L_Z3 ∩ C.

Verdict classification:

    KOIDE PREDICTED   ⇔   L_Z3 ⊂ C  AND  PDG ∈ L_Z3.
    KOIDE CONSISTENT  ⇔   L_Z3 ∩ C ≠ ∅.
    KOIDE CONFLICT    ⇔   L_Z3 ∩ C = ∅.

Outcome: **KOIDE CONSISTENT** (L_Z3 ∩ C is non-trivial: a 1-parameter
family).  Not PREDICTED (PDG ∉ L_Z3 because of degeneracy mismatch).
The BCC body-diagonal Z₃ structure is compatible with Koide as a
restricted sub-family but does not uniquely select PDG.

Additional tag: **PDG ∉ L_Z3** — three distinct masses require
Z₃-breaking input outside the equivariant Yukawa construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.koide.koide_geometry import (
    KOIDE_K_TARGET,
    koide_K_from_masses,
)
from clifford_3plus2_d5.koide.yukawa_eigenvalue_locus import (
    koide_special_ratio_symbolic,
    pdg_mass_ratio_matches_special,
)


def locus_intersection_is_non_empty() -> bool:
    """Return whether L_Z3 ∩ C ≠ ∅.

    True: at |v_t|/|v_o| = 3 + 2√2, the Yukawa eigenvalue triple
    satisfies Koide exactly.  Verified by direct symbolic computation.
    """

    r = koide_special_ratio_symbolic()
    eigenvalues = (
        sp.simplify(3 * r ** 2),
        sp.Rational(3, 2),
        sp.Rational(3, 2),
    )
    K = koide_K_from_masses(eigenvalues)
    return sp.simplify(K - KOIDE_K_TARGET) == 0


def locus_is_strictly_inside_cone() -> bool:
    """Return whether L_Z3 ⊂ C (every Z₃-equivariant Yukawa satisfies Koide).

    False: only the special ratio |v_t|/|v_o| = 3 + 2√2 lies on the cone.
    Most (|v_t|, |v_o|) combinations give K ≠ 2/3.
    """

    # Counterexample: |v_t| = |v_o| = 1 (off the special ratio).
    eigenvalues = (
        sp.Integer(3),
        sp.Rational(3, 2),
        sp.Rational(3, 2),
    )
    K = koide_K_from_masses(eigenvalues)
    return sp.simplify(K - KOIDE_K_TARGET) == 0


def pdg_in_locus() -> bool:
    """Return whether PDG charged-lepton masses lie in L_Z3.

    False: PDG masses are all distinct; L_Z3 always has 2-fold
    degeneracy.
    """

    return pdg_mass_ratio_matches_special()


def classify_verdict() -> str:
    """Return one of: KOIDE PREDICTED / KOIDE CONSISTENT / KOIDE CONFLICT.

    Per the analytic characterization:
    - Locus ∩ Cone ≠ ∅ but Locus ⊄ Cone, and PDG ∉ Locus.
    - Verdict: KOIDE CONSISTENT.
    """

    has_intersection = locus_intersection_is_non_empty()
    locus_in_cone = locus_is_strictly_inside_cone()
    if has_intersection and locus_in_cone:
        return "KOIDE PREDICTED"
    if has_intersection:
        return "KOIDE CONSISTENT"
    return "KOIDE CONFLICT"


@dataclass(frozen=True)
class ConeLocusCompatibilityPayload:
    """Result of the Phase KO-4 verdict classifier."""

    locus_intersection_non_empty: bool
    locus_strictly_inside_cone: bool
    pdg_in_locus: bool
    final_verdict: str
    additional_tags: tuple[str, ...]
    interpretation: str


def cone_locus_compatibility_payload() -> ConeLocusCompatibilityPayload:
    """Run the Phase KO-4 audit."""

    intersect_ok = locus_intersection_is_non_empty()
    inside_ok = locus_is_strictly_inside_cone()
    pdg_ok = pdg_in_locus()
    verdict = classify_verdict()

    tags: list[str] = []
    if pdg_ok:
        tags.append("PDG IN LOCUS")
    else:
        tags.append("PDG NOT IN LOCUS")

    if verdict == "KOIDE CONSISTENT":
        interpretation = (
            "The Z₃-equivariant Yukawa locus L_Z3 = {(3|v_t|², (3/2)|v_o|², "
            "(3/2)|v_o|²)} intersects the Koide cone in a 1-parameter "
            "family at the special ratio |v_t|/|v_o| = 3 + 2√2 "
            f"(≈ {float(koide_special_ratio_symbolic()):.4f}).  "
            "L_Z3 ⊄ Cone: most points in L_Z3 are off the cone.  "
            "PDG ∉ L_Z3 because PDG has three distinct masses while L_Z3 "
            "always has 2-fold degeneracy.  "
            f"Verdict: KOIDE CONSISTENT.  The BCC body-diagonal Z₃ "
            "structure admits Koide-satisfying Yukawa solutions but "
            "does not uniquely select them.  Three distinct PDG masses "
            "require Z₃-breaking input — either a Higgs-VEV alignment "
            "that tilts the Yukawa off the equivariant locus, or an "
            "explicit Yukawa perturbation breaking the BCC body-diagonal "
            "symmetry.  Bold-B (dynamical Higgs sector) is the natural "
            "track to investigate VEV-driven cone selection."
        )
    elif verdict == "KOIDE PREDICTED":
        interpretation = (
            "L_Z3 ⊂ Cone: every Z₃-equivariant Yukawa satisfies Koide.  "
            "Major positive — the program's BCC body-diagonal Z₃ structure "
            "FORCES K = 2/3 for any Yukawa construction.  Investigate."
        )
    else:
        interpretation = (
            "L_Z3 ∩ Cone = ∅: the program's Z₃-equivariant Yukawa "
            "construction is INCOMPATIBLE with K = 2/3.  KILL — the "
            "apparent BCC↔Koide coincidence is broken by the program's "
            "structure.  No natural cone-residing Yukawa exists."
        )

    return ConeLocusCompatibilityPayload(
        locus_intersection_non_empty=intersect_ok,
        locus_strictly_inside_cone=inside_ok,
        pdg_in_locus=pdg_ok,
        final_verdict=verdict,
        additional_tags=tuple(tags),
        interpretation=interpretation,
    )
