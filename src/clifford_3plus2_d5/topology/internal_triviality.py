"""Phase D-1: chiral-16 internal Z_3-triviality demonstration.

The chiral-16 internal carrier (from ``lepton.clifford_patisalam``) is
built from Cl(0,10) gamma matrices acting on a 32-real-dimensional
spinor subspace.  These gammas carry **internal** indices 0..9; the
chiral-16 has NO spatial coordinate dependence.

The body-diagonal Z_3 rotation acts on SPATIAL coordinates (x, y, z),
not on internal Cl(0,10) indices.  By construction, the chiral-16
internal carrier is invariant under spatial Z_3 — it transforms as the
trivial representation.

This module documents this triviality and provides a programmatic
verification: the natural action of spatial Z_3 on the chiral-16 is
the identity.

**Verdict**: under the spatial body-diagonal Z_3 acting via spatial
rotations alone, the chiral-16 decomposes as 16 × (trivial rep) = 16.
No three-generation structure emerges.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def chiral16_dimension() -> int:
    """Return 16 — the complex dimension of the chiral-16 carrier."""

    return 16


def chiral16_real_dimension() -> int:
    """Return 32 — the real dimension of the chiral-16 carrier."""

    return 32


def spatial_z3_action_on_internal() -> sp.Matrix:
    """Return the identity matrix on the internal carrier.

    The spatial Z_3 (body-diagonal rotation) acts trivially on internal
    indices because the chiral-16 is built from Cl(0,10) gamma matrices
    with internal labels, not spatial coordinates.  The identity action
    is forced by the carrier construction.
    """

    return sp.eye(chiral16_real_dimension())


def chiral16_is_z3_trivial() -> bool:
    """Return whether the chiral-16 internal carrier is Z_3-trivial.

    True by construction.  This function is a documentation hook for
    the audit; the underlying fact is structural, not computational.
    """

    action = spatial_z3_action_on_internal()
    return action == sp.eye(chiral16_real_dimension())


def chiral16_z3_decomposition() -> dict[str, int]:
    """Return the chiral-16 decomposition under spatial Z_3.

    Since the action is trivial, all 16 complex (= 32 real) basis
    vectors carry trivial Z_3 character.
    """

    return {
        "trivial (chi = 1)": 16,
        "omega (chi = e^{2 pi i / 3})": 0,
        "omega^2 (chi = e^{-2 pi i / 3})": 0,
    }


def three_generations_dimensional_check() -> dict[str, int]:
    """Return arithmetic on whether 3 generations could fit into 16 trivial Z_3 chars.

    Three equivalent generations under Z_3 would require equal
    multiplicity (≈ 16/3) in each character, but 16 is not divisible
    by 3.  Additionally, all 16 components are in the trivial character
    — no ω or ω^2 components exist.
    """

    return {
        "chiral16_complex_dim": 16,
        "ideal_per_generation": 0,  # 16/3 is non-integer
        "trivial_char_actual": 16,
        "omega_char_actual": 0,
        "omega2_char_actual": 0,
    }


@dataclass(frozen=True)
class InternalTrivialityAuditPayload:
    chiral16_complex_dim: int
    chiral16_real_dim: int
    z3_acts_trivially: bool
    z3_decomposition: dict[str, int]
    can_yield_three_generations: bool
    verdict: str
    interpretation: str


def internal_z3_triviality_payload() -> InternalTrivialityAuditPayload:
    """Run the chiral-16 Z_3-triviality audit."""

    trivial = chiral16_is_z3_trivial()
    decomposition = chiral16_z3_decomposition()
    can_yield = False  # 16 = 16 × trivial; no ω, ω^2 components

    if trivial and not can_yield:
        verdict = "TOPOLOGY KILL — chiral-16 internal is Z_3-trivial"
        interpretation = (
            "The chiral-16 internal carrier is built from Cl(0,10) gamma "
            "matrices acting on a 32-real-dimensional spinor subspace.  "
            "These gammas have INTERNAL indices, NOT spatial coordinates.  "
            "The body-diagonal Z_3 acts on SPATIAL (x, y, z), so by "
            "construction the chiral-16 is Z_3-trivial.  Decomposition: "
            "16 = 16 (trivial char) + 0 (ω) + 0 (ω²).  No three-generation "
            "structure can emerge from spatial Z_3 alone acting on this "
            "carrier."
        )
    else:
        verdict = "TOPOLOGY UNEXPECTED"
        interpretation = (
            f"Trivial: {trivial}.  Decomposition: {decomposition}.  "
            "Unexpected pattern — investigate."
        )

    return InternalTrivialityAuditPayload(
        chiral16_complex_dim=chiral16_dimension(),
        chiral16_real_dim=chiral16_real_dimension(),
        z3_acts_trivially=trivial,
        z3_decomposition=decomposition,
        can_yield_three_generations=can_yield,
        verdict=verdict,
        interpretation=interpretation,
    )
