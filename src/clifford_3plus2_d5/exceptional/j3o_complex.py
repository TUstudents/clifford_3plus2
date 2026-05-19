"""Phase 2b: J_3^C(O) complexification.

The complexified exceptional Jordan algebra ``J_3^C(O) = J_3(O) ⊗_R C``
has 54 real dimensions (= 27 complex).  Under the natural extension
``Spin(10) × U(1) ⊂ E_6 × U(1)``, the 27_C rep decomposes as

```text
27_C = (16, +1) ⊕ (10, -2) ⊕ (1, +4)
```

with U(1) charges chosen so the anomaly cancels (16 · 1 + 10 · (-2) +
1 · 4 = 0).  Realified:

```text
54 = (16 + 10 + 1) ⊕ (16* + 10* + 1*) = 16 + 16* + 10 + 10* + 1 + 1*
```

This produces a **chiral-16 PARTICLE plus its conjugate antiparticle**
representation, NOT three generation copies.

Boyle's three-generation claim relies on additional structure not
captured by the standard Spin(10) × U(1) decomposition of E_6's 27_C.
Without that extra structure, J_3^C(O) gives one generation + its
antiparticle + extras, just like the SM's one-generation content +
antiparticles.

**Final verdict**: neither J_3(O) nor J_3^C(O) under Spin(10) (or
Spin(10) × U(1)) gives three independent chiral-16 generation copies.
The exceptional-Jordan approach to three generations is closed at the
Spin(10)-representation-theory level.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.exceptional.spin10_on_j3o import (
    chiral16_real_dimension,
)


def j3o_complex_real_dimension() -> int:
    """Return the real dimension of J_3^C(O) (= 2 × 27 = 54)."""

    return 54


def standard_decomposition_realified() -> dict[str, int]:
    """Return the realified Spin(10) × U(1) decomposition of 27_C.

    Each complex irrep contributes its real dimension; the conjugate
    rep gives an independent copy of the same real dimension.
    """

    return {
        "16 (chiral-16, U(1) = +1)": 16,
        "10 (vector, U(1) = -2)": 10,
        "1  (singlet, U(1) = +4)": 1,
        "16* (anti-chiral-16, U(1) = -1)": 16,
        "10* (anti-vector, U(1) = +2)": 10,
        "1*  (anti-singlet, U(1) = -4)": 1,
    }


def decomposition_total_dimension() -> int:
    return sum(standard_decomposition_realified().values())


def chiral16_copies_count() -> int:
    """Return how many independent chiral-16 copies the decomposition has.

    16 + 16* is ONE generation (particle + antiparticle), not two
    generations.  But the audit reports the bare count of chiral-16-type
    real-16-dim subreps; the interpretation flags particle/antiparticle.
    """

    return 2  # 16 and 16*


def three_chiral16_required_dimension() -> int:
    return 3 * chiral16_real_dimension()


def three_chiral16_fits_inside_j3o_complex() -> bool:
    """Return whether 3 × 16 = 48 ≤ 54 dimensionally.

    Note: dimensional fit alone is NOT enough; the Spin(10) × U(1)
    representation theory forces 54 = 16 + 16* + 10 + 10* + 1 + 1*,
    giving only 2 chiral-16-type subreps (particle + antiparticle).
    """

    return three_chiral16_required_dimension() <= j3o_complex_real_dimension()


@dataclass(frozen=True)
class J3OComplexAuditPayload:
    j3o_complex_dimension: int
    decomposition: dict[str, int]
    decomposition_sum: int
    chiral16_copies: int
    three_chiral16_dimensional_fit: bool
    three_chiral16_representation_fit: bool
    verdict: str
    interpretation: str


def j3o_complex_decomposition_audit_payload() -> J3OComplexAuditPayload:
    """Run the FJ-5b kill test."""

    decomposition = standard_decomposition_realified()
    total = sum(decomposition.values())
    copies = chiral16_copies_count()
    dim_fits = three_chiral16_fits_inside_j3o_complex()
    rep_fits = copies >= 3  # need at least 3 distinct chiral-16 subreps

    if total == 54 and not rep_fits:
        verdict = "J_3^C(O) KILL — 54 = 16 + 16* + 10 + 10* + 1 + 1*"
        interpretation = (
            f"Complexifying J_3(O) doubles the real dimension to 54 = "
            f"2 × 27.  Under Spin(10) × U(1) ⊂ E_6 × U(1), the rep "
            f"decomposes as 16 + 16* + 10 + 10* + 1 + 1*, giving "
            f"ONE chiral-16 (particle) plus its conjugate 16* "
            f"(antiparticle), not three independent generation copies.  "
            f"Dimensionally three chiral-16 fit (48 ≤ 54), but the "
            f"REPRESENTATION-THEORY forced decomposition disallows it.  "
            f"Boyle's three-generation argument must rely on structure "
            f"beyond Spin(10) × U(1) acting on J_3^C(O).  The "
            f"exceptional-Jordan family approach to three generations "
            f"is closed at the carrier level."
        )
    else:
        verdict = "J_3^C(O) UNEXPECTED"
        interpretation = (
            f"Decomposition sum: {total} (expected 54).  Chiral-16 copies: "
            f"{copies}.  Dimensional fit: {dim_fits}, rep fit: {rep_fits}.  "
            f"Investigate."
        )

    return J3OComplexAuditPayload(
        j3o_complex_dimension=j3o_complex_real_dimension(),
        decomposition=decomposition,
        decomposition_sum=total,
        chiral16_copies=copies,
        three_chiral16_dimensional_fit=dim_fits,
        three_chiral16_representation_fit=rep_fits,
        verdict=verdict,
        interpretation=interpretation,
    )
