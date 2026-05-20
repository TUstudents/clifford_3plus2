"""Phase SC-2: BCC lattice + Bialynicki-Birula walk centrosymmetry audit.

Centrosymmetry under spatial inversion ``r → -r`` is the load-bearing
symmetry for the Strong-CP audit.  If the BCC walk is invariant
under inversion (combined with Dirac parity P = γ⁰), then every
effective-Hamiltonian correction H^(n) at order ε^n has definite
parity under ``k → -k``: g-irrep for even n, u-irrep for odd n
(with the specific u-irrep content to be pinned by Phase SC-3).

This module verifies:

1. BCC lattice site set ``Z³ ∪ (Z + 1/2)³`` is closed under
   ``r → -r``.
2. BB walk Dirac hops satisfy ``γ⁰ · W_full(h) · γ⁰⁻¹ = W_full(-h)``
   where ``W_full(h) = block_diag(W_R(h), W_L(h))`` and
   ``W_L(h) = W_R(-h)`` (the opposite-helicity convention).
3. The induced Bloch symbol ``D(k) = sum_h W_full(h) e^{i ε k·h}``
   satisfies ``γ⁰ · D(k) · γ⁰⁻¹ = D(-k)`` — the momentum-space
   form of centrosymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.spacetime_qca import block_diag
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import opposite_helicity_hops
from clifford_3plus2_d5.strongcp.reuse import (
    bcc_dirac_symbol,
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    gamma0,
    same_matrix,
    symbolic_momentum,
)


def bcc_lattice_site_set_is_centrosymmetric(extent: int = 3) -> bool:
    """Verify ``Z³ ∪ (Z + 1/2)³`` is closed under ``r → -r``.

    Trivial check: every site ``(x, y, z)`` with integer or
    half-integer coordinates has its negation also in the lattice.
    Examined explicitly for ``|x|, |y|, |z| ≤ extent``.
    """

    integer_sites = {
        (i, j, k)
        for i in range(-extent, extent + 1)
        for j in range(-extent, extent + 1)
        for k in range(-extent, extent + 1)
    }
    half_sites = {
        (i + sp.Rational(1, 2), j + sp.Rational(1, 2), k + sp.Rational(1, 2))
        for i in range(-extent, extent)
        for j in range(-extent, extent)
        for k in range(-extent, extent)
    }
    all_sites = integer_sites | half_sites
    for site in all_sites:
        negated = tuple(-component for component in site)
        if negated not in all_sites:
            return False
    return True


def bb_dirac_hop(direction_index: int) -> sp.Matrix:
    """Return the 4×4 Dirac BB hop ``W_full(h) = block_diag(W_R, W_L)``."""

    right_hops = bialynicki_birula_hops()
    left_hops = opposite_helicity_hops(right_hops)
    return block_diag(right_hops[direction_index], left_hops[direction_index])


def bb_walk_is_centrosymmetric_per_direction() -> bool:
    """Verify ``γ⁰ · W_full(h) · γ⁰⁻¹ = W_full(-h)`` for every BCC direction."""

    directions = bialynicki_birula_directions()
    g0 = gamma0()
    g0_inv = g0.inv()
    for index, direction in enumerate(directions):
        w_h = bb_dirac_hop(index)
        minus_h = tuple(-component for component in direction)
        minus_index = directions.index(minus_h)
        w_minus_h = bb_dirac_hop(minus_index)
        transformed = (g0 * w_h * g0_inv).applyfunc(sp.simplify)
        if not same_matrix(w_minus_h, transformed):
            return False
    return True


def bloch_symbol_is_centrosymmetric() -> bool:
    """Verify ``γ⁰ · D(k) · γ⁰⁻¹ = D(-k)`` at the symbol level.

    The Bloch symbol of the BCC Dirac walk is a polynomial in ε and k.
    Centrosymmetry of the walk + parity of momentum implies this
    symbolic identity holds.
    """

    epsilon, kx, ky, kz = symbolic_momentum()
    symbol = bcc_dirac_symbol(epsilon, kx, ky, kz)
    flipped = bcc_dirac_symbol(epsilon, -kx, -ky, -kz)
    g0 = gamma0()
    g0_inv = g0.inv()
    transformed = (g0 * symbol * g0_inv).applyfunc(sp.simplify)
    residual = (flipped - transformed).applyfunc(sp.expand).applyfunc(sp.simplify)
    return same_matrix(residual, sp.zeros(symbol.rows, symbol.cols))


@dataclass(frozen=True)
class BCCCentrosymmetryAuditPayload:
    """Result of the Phase SC-2 audit."""

    lattice_sites_centrosymmetric: bool
    walk_hops_centrosymmetric: bool
    bloch_symbol_centrosymmetric: bool
    all_centrosymmetric: bool
    verdict: str
    interpretation: str


def bcc_centrosymmetry_payload() -> BCCCentrosymmetryAuditPayload:
    """Run the full Phase SC-2 centrosymmetry audit."""

    lattice = bcc_lattice_site_set_is_centrosymmetric()
    walk = bb_walk_is_centrosymmetric_per_direction()
    bloch = bloch_symbol_is_centrosymmetric()
    all_ok = lattice and walk and bloch

    if all_ok:
        verdict = "BCC CENTROSYMMETRIC — parity selection rule applies"
        interpretation = (
            "BCC lattice sites are closed under spatial inversion.  "
            "The Bialynicki-Birula Dirac walk satisfies "
            "γ⁰·W_full(h)·γ⁰⁻¹ = W_full(-h) for every body-diagonal "
            "direction h.  The induced Bloch symbol obeys the "
            "momentum-space centrosymmetry "
            "γ⁰·D(k)·γ⁰⁻¹ = D(-k).  Therefore every effective-"
            "Hamiltonian correction H^(n) extracted via BCH from "
            "D(k) has definite parity under k → -k: g-irrep for "
            "even n, u-irrep for odd n.  Phase SC-3 pins the "
            "specific u-irrep content of H^(1), H^(3) and applies "
            "the cubic-group selection rule to θ_QCD."
        )
    else:
        verdict = "CENTROSYMMETRY INCONSISTENCY"
        interpretation = (
            f"lattice: {lattice}; walk hops: {walk}; bloch: {bloch}.  "
            "Investigate before proceeding to Phase SC-3."
        )

    return BCCCentrosymmetryAuditPayload(
        lattice_sites_centrosymmetric=lattice,
        walk_hops_centrosymmetric=walk,
        bloch_symbol_centrosymmetric=bloch,
        all_centrosymmetric=all_ok,
        verdict=verdict,
        interpretation=interpretation,
    )
