"""Phase SC-6 helper: relate the BCC walk's θ̄ contribution to neutron-EDM bound.

The neutron electric-dipole-moment bound (Abel et al., 2020,
arXiv:2001.11966) requires

    |θ̄|  ≤  10⁻¹⁰     (1)

The strongcp audit produces a verdict on the BCC walk's contribution
to θ̄ at each order in ε.  The four pre-named verdict classes:

| Order of θ̄ contribution            | Verdict             |
| ---                                  | ---                 |
| θ̄ = 0 to all orders (structural)    | STRONG-CP TRIVIAL  |
| θ̄ = O(ε^n), n ≥ 1, |θ̄| ≪ 10⁻¹⁰      | STRONG-CP SAFE     |
| θ̄ = O(ε^n), |θ̄| ≳ 10⁻¹⁰             | STRONG-CP TENSION  |

For the BCC walk: ε ≲ 2 × 10⁻³³ m from sme/'s UNFALSIFIABLE PASS at
~10² × ℓ_P.  Any ε^n contribution with n ≥ 1 is suppressed by at
least ``(ε / 1 fm)^n`` ≈ ``(2 × 10⁻³³ / 10⁻¹⁵)^n`` ≈ ``10⁻¹⁸n`` —
vastly below 10⁻¹⁰ for any n ≥ 1.

The selection-rule + chiral-anomaly arguments in Phases SC-1..SC-3
+ SC-5 establish that θ̄ contribution = 0 at O(ε) and O(ε²) (TRIVIAL
at these orders).  A non-trivial cross-term ``tr(γ^5 H^(1) H^(2))``
at O(ε³) hints at a potential O(ε³) contribution; phase SC-4
(deferred — lattice topological-charge density) is the direct
computation confirming this contribution either vanishes or is
ε³-suppressed below 10⁻¹⁰.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


NEUTRON_EDM_BOUND_ON_THETA_BAR: sp.Expr = sp.Float("1e-10")
EPSILON_UPPER_BOUND_METRES: sp.Expr = sp.Float("2e-33")  # from sme/
TYPICAL_HADRONIC_SCALE_METRES: sp.Expr = sp.Float("1e-15")  # ~ 1 fm


def epsilon_over_hadron_ratio() -> sp.Expr:
    """Return ``ε_max / Λ_QCD⁻¹``, the natural suppression factor per ε."""

    return sp.simplify(
        EPSILON_UPPER_BOUND_METRES / TYPICAL_HADRONIC_SCALE_METRES
    )


def theta_bar_upper_bound_at_order(n: int) -> sp.Expr:
    """Return ``(ε / Λ_QCD⁻¹)^n`` — the natural θ̄ suppression at order ε^n."""

    return sp.simplify(epsilon_over_hadron_ratio() ** n)


def is_safe_at_order(n: int) -> bool:
    """Return whether an ε^n-suppressed θ̄ contribution is below 10⁻¹⁰."""

    bound = theta_bar_upper_bound_at_order(n)
    return float(bound) < float(NEUTRON_EDM_BOUND_ON_THETA_BAR)


@dataclass(frozen=True)
class ThetaBarConstraintPayload:
    """Result of the θ̄ bound analysis."""

    neutron_edm_bound: sp.Expr
    epsilon_max_metres: sp.Expr
    hadronic_scale_metres: sp.Expr
    suppression_per_epsilon: sp.Expr
    safe_at_order_one: bool
    safe_at_order_two: bool
    safe_at_order_three: bool
    interpretation: str


def theta_bar_constraint_payload() -> ThetaBarConstraintPayload:
    """Run the θ̄ bound analysis assuming any ε^n suppression at order n."""

    ratio = epsilon_over_hadron_ratio()
    safe1 = is_safe_at_order(1)
    safe2 = is_safe_at_order(2)
    safe3 = is_safe_at_order(3)

    interpretation = (
        f"Neutron-EDM bound on θ̄: |θ̄| ≤ {float(NEUTRON_EDM_BOUND_ON_THETA_BAR):.0e}.  "
        f"sme/ sidecar's UNFALSIFIABLE PASS gives ε ≲ "
        f"{float(EPSILON_UPPER_BOUND_METRES):.1e} m.  Natural suppression "
        f"per ε factor: ε / Λ_QCD⁻¹ ≈ "
        f"{float(EPSILON_UPPER_BOUND_METRES) / float(TYPICAL_HADRONIC_SCALE_METRES):.2e}.  "
        f"At order ε^n the natural upper bound on θ̄ is "
        f"(ε / Λ_QCD⁻¹)^n.  Safe at order 1: {safe1}; order 2: {safe2}; "
        f"order 3: {safe3}.  Any ε^n ≥ 1 contribution to θ̄ is ε^n-suppressed "
        f"to well below 10⁻¹⁰ — STRONG-CP SAFE at every order n ≥ 1."
    )

    return ThetaBarConstraintPayload(
        neutron_edm_bound=NEUTRON_EDM_BOUND_ON_THETA_BAR,
        epsilon_max_metres=EPSILON_UPPER_BOUND_METRES,
        hadronic_scale_metres=TYPICAL_HADRONIC_SCALE_METRES,
        suppression_per_epsilon=ratio,
        safe_at_order_one=safe1,
        safe_at_order_two=safe2,
        safe_at_order_three=safe3,
        interpretation=interpretation,
    )
