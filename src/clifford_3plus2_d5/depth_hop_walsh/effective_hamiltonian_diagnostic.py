"""W3 — effective-Hamiltonian DIAGNOSTIC (never passes or kills).

Reports the strong-CP / effective-Hamiltonian cubic-harmonic content for context:
the BCH/log effective Hamiltonian has A2u(H^(2)) = 0 (no effective Lorentz theta
term) and H^(1) != 0 (a nonzero degree-2 Lorentz correction). This is the
low-energy Lorentz grammar — distinct from the primitive hop-shell alphabet tested
in W2. It would only bear on the stronger (wrong) claim "the family A2u mode must
survive as an effective Lorentz A2u term," which the depth conjecture does NOT make.

``diagnostic_only`` is always True: this gate cannot alter the depth verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.depth_hop_walsh.reuse import (
    effective_hamiltonian_first_correction,
    h2_a2u_component_is_zero,
    same_matrix,
)


def effective_h1_is_nonzero() -> bool:
    """Return whether the O(eps) effective Hamiltonian H^(1) (degree 2) is nonzero."""

    h1 = effective_hamiltonian_first_correction()
    return not same_matrix(h1, h1.zeros(*h1.shape))


@dataclass(frozen=True)
class EffectiveHamiltonianDiagnosticPayload:
    """W3 diagnostic payload — reports only, never a pass/kill of the depth gate."""

    final_verdict: str
    effective_a2u_is_zero: bool
    effective_h1_nonzero: bool
    diagnostic_only: bool
    interpretation: str


def effective_hamiltonian_diagnostic_payload() -> EffectiveHamiltonianDiagnosticPayload:
    """Return the W3 diagnostic (verdict string is informational only)."""

    a2u_zero = h2_a2u_component_is_zero()
    h1_nonzero = effective_h1_is_nonzero()

    interpretation = (
        f"Effective (BCH/log) Hamiltonian: A2u(H^(2)) zero = {a2u_zero} (no "
        f"effective Lorentz theta term), H^(1) nonzero = {h1_nonzero} (a degree-2 "
        "Lorentz correction). This is the low-energy Lorentz grammar, distinct from "
        "the primitive hop-shell alphabet (W2). It does NOT pass or kill the "
        "family-depth mechanism: a missing effective A2u says nothing about a "
        "primitive hop-shell A2u coefficient."
    )

    return EffectiveHamiltonianDiagnosticPayload(
        final_verdict="DEPTH_EFFECTIVE_HAMILTONIAN_DIAGNOSTIC",
        effective_a2u_is_zero=a2u_zero,
        effective_h1_nonzero=h1_nonzero,
        diagnostic_only=True,
        interpretation=interpretation,
    )
