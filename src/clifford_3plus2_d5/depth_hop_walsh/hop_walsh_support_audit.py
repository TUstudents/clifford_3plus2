"""W2 — [111]-singlet support test (PRIMARY gate).

Tests whether the raw BCC Weyl hop source carries the primitive [111] parity tower
``A1g(0) + T1u(1) + A2u(3)`` with NO degree-2 even quadrupole ``T2g``. Evaluated
separately per helicity (support = nonzero norm, not coefficient equality between
helicities, since A2u is parity-odd and flips sign under helicity).

Verdicts (per helicity, granular kill provenance):
* DEPTH_HOP_WALSH_SUPPORT_PASS           — baseline, T1u[111], A2u present; T2g[111] absent
* DEPTH_HOP_WALSH_SUPPORT_KILL_MISSING_A2U   — A2u absent
* DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT   — T2g[111] present
* DEPTH_HOP_WALSH_SUPPORT_KILL_MISSING_T1U   — T1u[111] absent

Combined: PASS iff both helicities PASS; HELICITY_SPLIT if exactly one passes;
otherwise the most specific shared kill reason.

A PASS establishes Claim A only (the angular 0,1,3 source ladder). It does NOT
derive the depths (the bridge d_radial = 2*Walsh-degree and the sqrt(5)
compatibility remain unproven).
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_hop_walsh.hop_walsh_decomposition import (
    a1g_baseline,
    a2u_component,
    frobenius_norm_squared,
    is_zero_symbolic,
    t1u_singlet,
    t2g_singlet,
    walsh_coefficients,
)

PASS = "DEPTH_HOP_WALSH_SUPPORT_PASS"
KILL_MISSING_A2U = "DEPTH_HOP_WALSH_SUPPORT_KILL_MISSING_A2U"
KILL_T2G_PRESENT = "DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT"
KILL_MISSING_T1U = "DEPTH_HOP_WALSH_SUPPORT_KILL_MISSING_T1U"
HELICITY_SPLIT = "DEPTH_HOP_WALSH_SUPPORT_HELICITY_SPLIT"


def hop_walsh_support_verdict(
    baseline_present: bool,
    t1u_present: bool,
    a2u_present: bool,
    t2g_present: bool,
) -> str:
    """Return the per-helicity verdict. Pure decision (KILL-testable).

    The required pattern is: baseline + T1u[111] + A2u present, T2g[111] absent.
    Kills carry specific provenance; A2u/T1u absence and T2g presence are reported
    in a fixed priority so a synthetic control hits exactly one reason.
    """

    if not a2u_present:
        return KILL_MISSING_A2U
    if t2g_present:
        return KILL_T2G_PRESENT
    if not t1u_present:
        return KILL_MISSING_T1U
    if baseline_present:
        return PASS
    # Baseline missing is unusual; treat as a missing-T1u-class structural kill.
    return KILL_MISSING_T1U


def combined_verdict(right_verdict: str, left_verdict: str) -> str:
    """Combine the two per-helicity verdicts.

    PASS iff both pass; HELICITY_SPLIT if exactly one passes (informative, never a
    generic kill); otherwise the shared/most-specific kill reason.
    """

    if right_verdict == PASS and left_verdict == PASS:
        return PASS
    if (right_verdict == PASS) != (left_verdict == PASS):
        return HELICITY_SPLIT
    # Neither passes: prefer a shared reason, else the right-handed reason.
    if right_verdict == left_verdict:
        return right_verdict
    return right_verdict


@dataclass(frozen=True)
class HelicitySupport:
    """Per-helicity support flags and verdict."""

    helicity: str
    baseline_present: bool
    t1u_singlet_present: bool
    a2u_present: bool
    t2g_singlet_present: bool
    t2g_full_triplet_zero: bool
    verdict: str


def _present(matrix: sp.Matrix) -> bool:
    """Two-tier support: prefer exact symbolic nonzero; Frobenius norm as fallback."""

    if not is_zero_symbolic(matrix):
        return True
    return frobenius_norm_squared(matrix) != 0


def helicity_support(helicity: str) -> HelicitySupport:
    """Return the support flags and verdict for one helicity."""

    coefficients = walsh_coefficients(helicity)
    baseline = _present(a1g_baseline(helicity))
    t1u = _present(t1u_singlet(helicity))
    a2u = _present(a2u_component(helicity))
    t2g = _present(t2g_singlet(helicity))
    triplet_zero = all(
        is_zero_symbolic(coefficients[subset]) for subset in ("xy", "yz", "zx")
    )
    verdict = hop_walsh_support_verdict(baseline, t1u, a2u, t2g)
    return HelicitySupport(
        helicity=helicity,
        baseline_present=baseline,
        t1u_singlet_present=t1u,
        a2u_present=a2u,
        t2g_singlet_present=t2g,
        t2g_full_triplet_zero=triplet_zero,
        verdict=verdict,
    )


@dataclass(frozen=True)
class HopWalshSupportPayload:
    """W2 payload: per-helicity support and the combined verdict."""

    final_verdict: str
    right: HelicitySupport
    left: HelicitySupport
    right_handed_verdict: str
    left_handed_verdict: str
    interpretation: str


def hop_walsh_support_audit_payload() -> HopWalshSupportPayload:
    """Return the W2 primary verdict."""

    right = helicity_support("right")
    left = helicity_support("left")
    final = combined_verdict(right.verdict, left.verdict)

    if final == PASS:
        interpretation = (
            "The raw BCC Weyl hop source carries the primitive [111] parity tower "
            "A1g(0) + T1u(1) + A2u(3) with no degree-2 even T2g singlet, for both "
            "helicities. This establishes Claim A only — the angular 0,1,3 source "
            "ladder. It does NOT derive the depths: the bridge d_radial = 2*degree "
            "and the sqrt(5) = sqrt(2_BCC + 3_color) compatibility remain unproven."
        )
    elif final == HELICITY_SPLIT:
        interpretation = (
            f"The support pattern holds for one helicity but not the other "
            f"(right={right.verdict}, left={left.verdict}). Reported as a split, "
            "not collapsed into a generic kill."
        )
    elif final == KILL_T2G_PRESENT:
        interpretation = (
            "The raw BCC Weyl hop source carries a nonzero degree-2 even [111] T2g "
            "singlet — exactly the quadrupole the cube/parity selection rule must "
            "remove. The {0,2,6} ladder is NOT realized as a parity-selected cube "
            "source; the depth embedding remains an honest free fit."
        )
    elif final == KILL_MISSING_A2U:
        interpretation = (
            "The raw BCC Weyl hop source has no primitive A2u (xyz) component, so "
            "the depth-6 family mode is absent. The cube/parity story is dead; the "
            "depth embedding remains an honest free fit."
        )
    else:
        interpretation = (
            "The raw BCC Weyl hop source lacks [111] vector (T1u) support. The "
            "cube/parity story is not realized; the depth embedding remains a free fit."
        )

    return HopWalshSupportPayload(
        final_verdict=final,
        right=right,
        left=left,
        right_handed_verdict=right.verdict,
        left_handed_verdict=left.verdict,
        interpretation=interpretation,
    )
