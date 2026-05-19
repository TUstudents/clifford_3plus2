"""Phase A-4 (FA-5, FA-6, FA-7): symbolic ε constraint and scale verdict.

Given H^(1)'s three identified d^{(5)} components (from Phase A-2)
and the experimental bound on d^{(5)} (from Phase A-3's literature
note), this module computes the maximum allowed ε and classifies it
into one of four pre-named scale verdicts.

Convention:
    ε is the lattice scale in metres.
    The cited experimental bound is in GeV⁻¹; we convert via the
    natural-units conversion ``GeV⁻¹ ≈ 1.97 × 10⁻¹⁶ m``.

The PLAN's four pre-named outcomes:

    SUB-PLANCK KILL      ε_max ≤ ℓ_P
    PLANCK-CONSISTENT    ε_max within 10× of ℓ_P (10⁻³⁵ – 10⁻³⁴ m)
    UNFALSIFIABLE PASS   10⁻³⁴ m < ε_max ≤ 10⁻²⁵ m
    OBSERVABLE POSITIVE  ε_max > 10⁻²⁵ m
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.sme.sme_tensor_mapping import (
    T2gTensorEntry,
    t2g_tensor_entries,
)


# Physical constants (exact SymPy rationals or numerical Floats).
# ℓ_P (Planck length) = 1.616255(18) × 10⁻³⁵ m  (CODATA).
PLANCK_LENGTH_METRES: sp.Expr = sp.Float("1.616255e-35")

# Natural-units conversion: 1 GeV⁻¹ = ℏc / (1 GeV) ≈ 1.973 × 10⁻¹⁶ m.
GEV_INVERSE_TO_METRES: sp.Expr = sp.Float("1.97327e-16")

# Threshold separating UNFALSIFIABLE PASS from OBSERVABLE POSITIVE.
OBSERVABLE_THRESHOLD_METRES: sp.Expr = sp.Float("1e-25")

# Tightest representative bound on the d^{(5)} fermion-sector
# coefficient, in GeV⁻¹.  Drawn from the order-of-magnitude bound in
# the Phase A-3 literature note (electron-sector sidereal modulation,
# pending Kostelecky-Russell entry-id verification).
DIM5_FERMION_BOUND_GEV_INVERSE: sp.Expr = sp.Float("1e-17")


def dim5_fermion_bound_in_metres() -> sp.Expr:
    """Return the d^{(5)} bound converted to metres.

    bound_m = bound_GeV_inverse × (GeV⁻¹ → metres conversion).
    """

    return sp.simplify(
        DIM5_FERMION_BOUND_GEV_INVERSE * GEV_INVERSE_TO_METRES
    )


def epsilon_bound_per_component(entry: T2gTensorEntry) -> sp.Expr:
    """Return ε ≤ bound / |coefficient| for a single d^{(5)} component.

    Each entry has coefficient ±1 (the symbolic T_{2g} tensor entry
    extracted from H^(1)).  The ε bound from this single component is
    simply ``bound / |coefficient| = bound``.
    """

    bound = dim5_fermion_bound_in_metres()
    magnitude = sp.Abs(entry.coefficient)
    return sp.simplify(bound / magnitude)


def epsilon_bound_tightest_face() -> sp.Expr:
    """Return the minimum ε bound over all three d^{(5)} components.

    Multiple components contribute; the constraint on ε is the
    tightest single-component bound (worst-case).  Per Phase A-2 all
    three coefficients have magnitude 1, so the result equals
    ``dim5_fermion_bound_in_metres()``.
    """

    entries = t2g_tensor_entries()
    bounds = [epsilon_bound_per_component(entry) for entry in entries]
    return sp.simplify(sp.Min(*bounds))


def scale_verdict(epsilon_bound: sp.Expr) -> str:
    """Classify the ε bound into one of the four pre-named scale verdicts."""

    bound_float = float(epsilon_bound)
    planck_float = float(PLANCK_LENGTH_METRES)
    observable_float = float(OBSERVABLE_THRESHOLD_METRES)

    if bound_float <= planck_float:
        return "SUB-PLANCK KILL"
    if bound_float <= 10 * planck_float:
        return "PLANCK-CONSISTENT"
    if bound_float <= observable_float:
        return "UNFALSIFIABLE PASS"
    return "OBSERVABLE POSITIVE"


def epsilon_bound_orders_of_magnitude_above_planck(
    epsilon_bound: sp.Expr,
) -> sp.Expr:
    """Return ``log10(epsilon_bound / Planck_length)`` for diagnostic use."""

    ratio = sp.simplify(epsilon_bound / PLANCK_LENGTH_METRES)
    return sp.log(ratio, 10)


@dataclass(frozen=True)
class EpsilonConstraintPayload:
    """Result of the Phase A-4 constraint extraction."""

    epsilon_bound_metres: sp.Expr
    epsilon_bound_orders_above_planck: sp.Expr
    planck_length_metres: sp.Expr
    observable_threshold_metres: sp.Expr
    dim5_fermion_bound_gev_inverse: sp.Expr
    dim5_fermion_bound_metres: sp.Expr
    scale_verdict: str
    verdict_class_explanation: str


def epsilon_constraint_payload() -> EpsilonConstraintPayload:
    """Run the Phase A-4 ε constraint computation."""

    bound = epsilon_bound_tightest_face()
    verdict = scale_verdict(bound)
    log_ratio = epsilon_bound_orders_of_magnitude_above_planck(bound)
    bound_m = dim5_fermion_bound_in_metres()

    if verdict == "SUB-PLANCK KILL":
        explanation = (
            "ε must be pushed below the Planck length to satisfy the "
            "tightest applicable d^{(5)} bound.  This indicates a "
            "structural inconsistency: the lattice scale of the BCC walk "
            "cannot accommodate the experimental constraint without going "
            "sub-Planckian.  Clean kill of the program at the ε scale."
        )
    elif verdict == "PLANCK-CONSISTENT":
        explanation = (
            "ε is bounded near the Planck length (within 10×).  The "
            "program is observation-consistent at this representative "
            "bound but unfalsifiable at current experimental reach.  "
            "Honest PASS — the program is alive but predicts no near-"
            "term observable effects in this channel."
        )
    elif verdict == "UNFALSIFIABLE PASS":
        explanation = (
            f"ε is bounded above the Planck length (by ~{sp.N(log_ratio, 3)} "
            f"orders of magnitude) but below current observational reach "
            f"(10⁻²⁵ m).  Honest PASS, slightly looser than strict Planck-"
            "consistency.  The program is alive but predicts no near-term "
            "observable effects in this channel."
        )
    else:  # OBSERVABLE POSITIVE
        explanation = (
            f"ε is bounded at ~{sp.N(bound, 3)} m, above the observable "
            f"threshold of 10⁻²⁵ m.  The program predicts measurable "
            "near-future CP-violating cubic-anisotropy effects at this "
            "ε scale.  Publishable positive — investigate specific "
            "experimental signatures."
        )

    return EpsilonConstraintPayload(
        epsilon_bound_metres=bound,
        epsilon_bound_orders_above_planck=log_ratio,
        planck_length_metres=PLANCK_LENGTH_METRES,
        observable_threshold_metres=OBSERVABLE_THRESHOLD_METRES,
        dim5_fermion_bound_gev_inverse=DIM5_FERMION_BOUND_GEV_INVERSE,
        dim5_fermion_bound_metres=bound_m,
        scale_verdict=verdict,
        verdict_class_explanation=explanation,
    )
