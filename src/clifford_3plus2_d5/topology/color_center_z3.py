"""Phase D-2: chiral-16 decomposition under SU(3)_c Cartan Z_3.

The SU(3)_c Cartan subalgebra contains a Z_3 subgroup with generator

    g_3 = exp(2 pi i / 3 . diag(0, 1, -1))
        = diag(1, omega, omega^2)
    where omega = exp(2 pi i / 3).

On a color triplet ``(R, G, B)`` the three components carry the
three Z_3 characters ``{1, omega, omega^2}``.  On a color antitriplet
``(R*, G*, B*)`` the characters are conjugated: ``{1, omega^2, omega}``.
On a color singlet the action is trivial.

Per-generation Weyl-fermion field count on the chiral-16:

| Field   | Color rep | Count | Z_3 character split |
|---|---|---|---|
| Q_L     | 3 (triplet)    | 6 (2 SU(2)_L doublet × 3 colors) | (2, 2, 2) over (1, ω, ω²) |
| u_R^c   | 3̄ (antitriplet) | 3 | (1, 1, 1) over (1, ω², ω) — i.e. (1, 1, 1) over (1, ω, ω²) |
| d_R^c   | 3̄              | 3 | same as u_R^c                  |
| L_L     | 1 (singlet)    | 2 (SU(2)_L doublet)              | (2, 0, 0)                       |
| e_R^c   | 1              | 1                               | (1, 0, 0)                       |
| ν_R^c   | 1              | 1                               | (1, 0, 0)                       |

Summing over characters:

    trivial (1):   2 + 1 + 1 + 2 + 1 + 1 = 8
    omega   (ω):   2 + 1 + 1               = 4   (Q_L G + u_R^c B̄ + d_R^c B̄)
    omega^2 (ω²):  2 + 1 + 1               = 4   (Q_L B + u_R^c Ḡ + d_R^c Ḡ)

Decomposition: ``16 = 8 + 4 + 4``.

This is **asymmetric** — leptons sit entirely in the trivial character
while quarks distribute across all three.  Three equivalent generations
would require multiplicities ``(16/3, 16/3, 16/3)`` which is not an
integer split.

**Verdict**: the color Z_3 center identification with spatial body-diagonal
Z_3 does NOT produce three symmetric generation copies.
"""

from __future__ import annotations

from dataclasses import dataclass


# Per-generation field count by (color rep, Z_3 char distribution)
PER_GENERATION_FIELD_CONTENT = (
    # (name, color_rep, total_count, z3_distribution as (trivial, omega, omega^2))
    ("Q_L",   "3",   6, (2, 2, 2)),
    ("u_R^c", "3bar", 3, (1, 1, 1)),
    ("d_R^c", "3bar", 3, (1, 1, 1)),
    ("L_L",   "1",   2, (2, 0, 0)),
    ("e_R^c", "1",   1, (1, 0, 0)),
    ("nu_R^c","1",   1, (1, 0, 0)),
)


def per_generation_total_fermion_count() -> int:
    return sum(item[2] for item in PER_GENERATION_FIELD_CONTENT)


def z3_decomposition_per_generation() -> tuple[int, int, int]:
    """Return ``(trivial, omega, omega^2)`` multiplicities for one generation."""

    trivial = sum(item[3][0] for item in PER_GENERATION_FIELD_CONTENT)
    omega = sum(item[3][1] for item in PER_GENERATION_FIELD_CONTENT)
    omega2 = sum(item[3][2] for item in PER_GENERATION_FIELD_CONTENT)
    return trivial, omega, omega2


def three_generations_required_multiplicity() -> float:
    """Return ``16/3`` — the symmetric per-character multiplicity required."""

    return 16 / 3


def multiplicities_are_symmetric() -> bool:
    """Return whether the three Z_3 characters carry equal multiplicities."""

    trivial, omega, omega2 = z3_decomposition_per_generation()
    return trivial == omega == omega2


@dataclass(frozen=True)
class ColorZ3DecompositionPayload:
    chiral16_total_count: int
    trivial_multiplicity: int
    omega_multiplicity: int
    omega2_multiplicity: int
    is_symmetric: bool
    expected_symmetric_value: float
    verdict: str
    interpretation: str


def color_z3_decomposition_payload() -> ColorZ3DecompositionPayload:
    """Run the Phase D-2 kill test."""

    trivial, omega, omega2 = z3_decomposition_per_generation()
    total = per_generation_total_fermion_count()
    symmetric = multiplicities_are_symmetric()
    expected_symmetric = three_generations_required_multiplicity()

    if total == 16 and not symmetric:
        verdict = "COLOR Z_3 KILL — chiral-16 decomposes as 8 + 4 + 4"
        interpretation = (
            f"Under the SU(3)_c Cartan Z_3 subgroup (g = diag(1, ω, ω²)), the "
            f"chiral-16 internal carrier decomposes as ({trivial}, {omega}, "
            f"{omega2}) into the (trivial, ω, ω²) characters.  This is "
            f"ASYMMETRIC — leptons (color singlets) sit entirely in the "
            f"trivial character, while quarks distribute across all three.  "
            f"Three equivalent generations would require multiplicities "
            f"({expected_symmetric:.3f}, {expected_symmetric:.3f}, "
            f"{expected_symmetric:.3f}), which is not an integer split.  "
            f"The color Z_3 center identification fails to produce three "
            f"symmetric generation copies."
        )
    else:
        verdict = "COLOR Z_3 UNEXPECTED"
        interpretation = (
            f"Decomposition: ({trivial}, {omega}, {omega2}).  Symmetric: "
            f"{symmetric}.  Investigate."
        )

    return ColorZ3DecompositionPayload(
        chiral16_total_count=total,
        trivial_multiplicity=trivial,
        omega_multiplicity=omega,
        omega2_multiplicity=omega2,
        is_symmetric=symmetric,
        expected_symmetric_value=expected_symmetric,
        verdict=verdict,
        interpretation=interpretation,
    )
