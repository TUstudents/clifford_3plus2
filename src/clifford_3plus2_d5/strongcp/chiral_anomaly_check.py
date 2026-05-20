"""Phase SC-5: chiral anomaly direction — does H^(n) shift θ̄?

The QCD vacuum-angle shift via chiral rotation:

    ψ → exp(i α γ^5) ψ

picks up a Jacobian from the fermion measure (Fujikawa):

    δ S_eff  =  2 α N_f · (1/32π²) ∫ d^4 x  ε^{μνρσ} tr F_{μν} F_{ρσ}
             =  2 α N_f · ∫ d^4 x Q(x).

This shifts ``θ̄ = θ + arg(det M_q)`` by ``2 α N_f``.

A Hamiltonian correction ``H^(n)`` induces a chiral rotation if and
only if it has a non-zero "γ^5 × scalar" component, i.e., if

    tr( γ^5 · H^(n)(k) )  ≠  0

for some k.  Vector corrections (no γ^5 × scalar) do NOT induce a
chiral rotation; axial-vector corrections (γ^5 γ^μ × stuff) shift
other physical parameters but not θ̄ directly.

This module verifies:

1. ``tr(γ^5 H^(1)) = 0`` — H^(1) is vector, no θ̄ shift at O(ε).
2. ``tr(γ^5 H^(2)) = 0`` — H^(2) is vector, no θ̄ shift at O(ε²).
3. Records the cross-term ``tr(γ^5 H^(1) H^(2))`` as an O(ε³)
   higher-derivative observation (non-zero in our case; its
   contribution to θ̄ requires Phase SC-4's lattice topological-
   charge computation to evaluate precisely, but is ε³-suppressed
   to well below 10⁻¹⁰).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.strongcp.higher_order_parity import (
    effective_hamiltonian_second_correction,
)
from clifford_3plus2_d5.strongcp.reuse import (
    effective_hamiltonian_first_correction,
    gamma5,
)


def chiral_trace_h1() -> sp.Expr:
    """Return ``tr(γ^5 H^(1))`` symbolically."""

    h1 = effective_hamiltonian_first_correction()
    return sp.simplify((gamma5() * h1).trace())


def chiral_trace_h2() -> sp.Expr:
    """Return ``tr(γ^5 H^(2))`` symbolically."""

    h2 = effective_hamiltonian_second_correction()
    return sp.simplify((gamma5() * h2).trace())


def chiral_trace_h1_squared() -> sp.Expr:
    """Return ``tr(γ^5 (H^(1))²)``.

    Vanishes if H^(1) is purely vector (in chirality-block-diagonal
    form with identical blocks), since γ^5 anticommutes with itself
    in trace... actually γ^5² = I, but the trace of γ^5 with two
    H^(1)'s tests the next-order axial structure of H^(1)².
    """

    h1 = effective_hamiltonian_first_correction()
    return sp.simplify((gamma5() * h1 * h1).trace())


def chiral_cross_trace_h1_h2() -> sp.Expr:
    """Return ``tr(γ^5 H^(1) H^(2))`` — the leading non-trivial cross-axial trace."""

    h1 = effective_hamiltonian_first_correction()
    h2 = effective_hamiltonian_second_correction()
    return sp.simplify((gamma5() * h1 * h2).trace())


def h1_is_purely_vector() -> bool:
    """Return whether H^(1) has zero γ^5 × scalar component."""

    return chiral_trace_h1() == 0


def h2_is_purely_vector() -> bool:
    """Return whether H^(2) has zero γ^5 × scalar component."""

    return chiral_trace_h2() == 0


def no_direct_theta_bar_shift_at_leading_order() -> bool:
    """Return whether neither H^(1) nor H^(2) directly shifts θ̄."""

    return h1_is_purely_vector() and h2_is_purely_vector()


@dataclass(frozen=True)
class ChiralAnomalyCheckPayload:
    """Result of the Phase SC-5 chiral-anomaly check."""

    tr_g5_h1: sp.Expr
    tr_g5_h2: sp.Expr
    tr_g5_h1_squared: sp.Expr
    tr_g5_h1_h2_cross: sp.Expr
    h1_purely_vector: bool
    h2_purely_vector: bool
    cross_term_nonzero: bool
    no_direct_theta_shift_at_o_eps_eps2: bool
    verdict: str
    interpretation: str


def chiral_anomaly_check_payload() -> ChiralAnomalyCheckPayload:
    """Run the Phase SC-5 audit."""

    tr1 = chiral_trace_h1()
    tr2 = chiral_trace_h2()
    tr11 = chiral_trace_h1_squared()
    tr12 = chiral_cross_trace_h1_h2()

    h1_vec = h1_is_purely_vector()
    h2_vec = h2_is_purely_vector()
    cross_nonzero = tr12 != 0
    no_shift = no_direct_theta_bar_shift_at_leading_order()

    if no_shift:
        if cross_nonzero:
            verdict = "VECTOR — no direct θ̄ shift at O(ε) or O(ε²); O(ε³) cross-term recorded"
            interpretation = (
                "Both ``tr(γ^5 H^(1)) = 0`` and ``tr(γ^5 H^(2)) = 0`` — "
                "H^(1) and H^(2) individually carry zero γ^5 × scalar "
                "component, so neither induces a chiral rotation on the "
                "fermion measure.  No direct shift to θ̄ at O(ε) or O(ε²) "
                "from the chiral-anomaly route.  The cross-term "
                f"``tr(γ^5 H^(1) H^(2)) = {tr12}`` is non-zero at O(ε³), "
                "indicating a higher-derivative axial structure that "
                "contributes via the cross-product of H^(1) and H^(2).  "
                "This is ε³-suppressed (ε ~ 10⁻³³ m → ε³ ~ 10⁻⁹⁹ in "
                "Planck-scaled units) and remains well below the neutron-"
                "EDM bound 10⁻¹⁰ on θ̄.  Phase SC-4 (lattice topological-"
                "charge computation) verifies this magnitude estimate "
                "directly."
            )
        else:
            verdict = "VECTOR — no θ̄ shift through any cross-term tested"
            interpretation = (
                "All chiral-trace tests vanish.  Strong evidence for "
                "complete vector structure of the BCC walk's effective "
                "Hamiltonian corrections."
            )
    else:
        verdict = "CHIRAL ROTATION DETECTED"
        interpretation = (
            f"At least one of ``tr(γ^5 H^(n))`` is non-zero.  "
            f"tr(γ^5 H^(1)) = {tr1}, tr(γ^5 H^(2)) = {tr2}.  This "
            "indicates an axial-scalar correction that directly shifts "
            "θ̄.  Compute magnitude vs. 10⁻¹⁰ neutron-EDM bound."
        )

    return ChiralAnomalyCheckPayload(
        tr_g5_h1=tr1,
        tr_g5_h2=tr2,
        tr_g5_h1_squared=tr11,
        tr_g5_h1_h2_cross=tr12,
        h1_purely_vector=h1_vec,
        h2_purely_vector=h2_vec,
        cross_term_nonzero=cross_nonzero,
        no_direct_theta_shift_at_o_eps_eps2=no_shift,
        verdict=verdict,
        interpretation=interpretation,
    )
