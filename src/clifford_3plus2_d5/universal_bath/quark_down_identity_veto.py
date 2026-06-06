"""Session 23 down identity-return veto.

Sessions 17 and 18 localized the down-sector bottom fork:

    baseline bottom count: 4  -> sqrt(2/3)
    odd-shell bottom count: 5 -> sqrt(5/6)

Session 21 reframed the down readout as a Hermitian current covariance rather
than a scalar vector, and Session 22 identified the quark source line as the
selected-S2 odd current.  The remaining question is whether a down-type mass
return may use the identity/direct contact line.

This session implements the finite retarded-current criterion:

    a down mass event must leave the visible sheet before returning.

In the primitive shell, the only zero-excursion return is the even direct line.
Vetoing that line leaves exactly the five odd channels

    2_BCC + 3_color,

which is the Session 18 rank-five bottom candidate.  The result is conditional
on accepting the retarded-current criterion as the physical down readout rule;
it is not derived from bare BB block algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    COLOR,
    DIRECT,
    EVEN,
    ODD,
    quark_primitive_channels,
)
from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    down_baseline_clebsch_vector,
    down_baseline_counts,
)
from clifford_3plus2_d5.universal_bath.quark_active_current_readout import (
    DOWN_IDENTITY_VETO_PREMISE,
    quark_active_current_readout_payload,
)
from clifford_3plus2_d5.universal_bath.quark_current_parity_selector import (
    ODD_CURRENT_PHYSICAL_PREMISE,
    quark_current_parity_selector_payload,
)
from clifford_3plus2_d5.universal_bath.quark_down_odd_shell import (
    BOTTOM_ODD_SHELL_READOUT_PREMISE,
    bcc_odd_subset,
    odd_shell_subset,
    primitive_counts,
    quark_down_odd_shell_payload,
)

RETARDED_DOWN_CURRENT_PREMISE = (
    "down_mass_event_requires_nonidentity_hidden_excursion_before_return"
)


@dataclass(frozen=True)
class PrimitiveReturnDecision:
    """Retarded-current decision for one primitive down return channel."""

    name: str
    parity: str
    sector: str
    return_order: int
    is_identity_direct: bool
    hidden_excursion: bool
    retarded_current_allowed: bool


@dataclass(frozen=True)
class QuarkDownIdentityVetoPayload:
    """Session 23 down identity-return veto verdict."""

    final_verdict: str
    active_current_readout_pass: bool
    current_parity_selector_pass: bool
    odd_shell_pass: bool
    primitive_decisions: tuple[PrimitiveReturnDecision, ...]
    direct_identity_names: tuple[str, ...]
    retarded_allowed_names: tuple[str, ...]
    retarded_rejected_names: tuple[str, ...]
    direct_identity_is_unique: bool
    direct_identity_vetoed: bool
    all_allowed_returns_are_odd: bool
    allowed_return_count: int
    allowed_return_breakdown: dict[str, int]
    retarded_counts: dict[str, int]
    retarded_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    baseline_counts: dict[str, int]
    baseline_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    baseline_control_rejected_by_retarded_predicate: bool
    rank_five_selected_inside_retarded_model: bool
    down_identity_veto_premise_reduced: bool
    microscopic_bare_bcc_derivation: bool
    remaining_physical_premises: tuple[str, ...]
    interpretation: str


def primitive_return_decisions() -> tuple[PrimitiveReturnDecision, ...]:
    """Return the retarded-current decision for every primitive channel."""

    decisions: list[PrimitiveReturnDecision] = []
    for channel in quark_primitive_channels():
        is_identity = channel.parity == EVEN and channel.sector == DIRECT
        return_order = 0 if is_identity else 1
        hidden_excursion = return_order > 0
        retarded_allowed = hidden_excursion and channel.parity == ODD
        decisions.append(
            PrimitiveReturnDecision(
                name=channel.name,
                parity=channel.parity,
                sector=channel.sector,
                return_order=return_order,
                is_identity_direct=is_identity,
                hidden_excursion=hidden_excursion,
                retarded_current_allowed=retarded_allowed,
            )
        )
    return tuple(decisions)


def direct_identity_decisions() -> tuple[PrimitiveReturnDecision, ...]:
    """Return identity/direct contact decisions."""

    return tuple(decision for decision in primitive_return_decisions() if decision.is_identity_direct)


def retarded_allowed_decisions() -> tuple[PrimitiveReturnDecision, ...]:
    """Return nonidentity hidden-excursion current returns."""

    return tuple(
        decision for decision in primitive_return_decisions() if decision.retarded_current_allowed
    )


def retarded_rejected_decisions() -> tuple[PrimitiveReturnDecision, ...]:
    """Return primitive channels rejected by the retarded-current criterion."""

    return tuple(
        decision for decision in primitive_return_decisions() if not decision.retarded_current_allowed
    )


def direct_identity_is_unique() -> bool:
    """Return whether the primitive shell has exactly one direct identity line."""

    identities = direct_identity_decisions()
    return len(identities) == 1 and identities[0].name == "direct_even_return"


def direct_identity_vetoed() -> bool:
    """Return whether the retarded-current criterion rejects the identity line."""

    identities = direct_identity_decisions()
    return direct_identity_is_unique() and all(
        not decision.retarded_current_allowed for decision in identities
    )


def allowed_return_breakdown() -> dict[str, int]:
    """Return sector counts for retarded-allowed returns."""

    allowed = retarded_allowed_decisions()
    return {
        "bcc_odd": sum(1 for decision in allowed if decision.sector == BCC),
        "color_odd": sum(1 for decision in allowed if decision.sector == COLOR),
        "odd_total": len(allowed),
        "direct_even": sum(1 for decision in allowed if decision.sector == DIRECT),
    }


def all_allowed_returns_are_odd() -> bool:
    """Return whether every retarded return is an odd current channel."""

    allowed = retarded_allowed_decisions()
    return len(allowed) > 0 and all(decision.parity == ODD for decision in allowed)


def retarded_down_counts() -> dict[str, int]:
    """Return down counts selected by the retarded-current criterion."""

    return {
        "d": len(quark_primitive_channels()),
        "s": bcc_odd_subset().count,
        "b": len(retarded_allowed_decisions()),
    }


def sqrt_count_profile(counts: dict[str, int], denominator: int = 6) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return ``sqrt(n/denominator)`` in down order ``(d,s,b)``."""

    return (
        sp.sqrt(sp.Rational(counts["d"], denominator)),
        sp.sqrt(sp.Rational(counts["s"], denominator)),
        sp.sqrt(sp.Rational(counts["b"], denominator)),
    )


def retarded_down_profile() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the down Clebsch profile selected by retarded identity veto."""

    return sqrt_count_profile(retarded_down_counts())


def baseline_control_rejected_by_retarded_predicate() -> bool:
    """Return whether the S3/projector baseline fails the retarded predicate."""

    return (
        down_baseline_counts()["b"] != len(retarded_allowed_decisions())
        and down_baseline_clebsch_vector() != retarded_down_profile()
    )


def rank_five_selected_inside_retarded_model() -> bool:
    """Return whether retarded current selection gives the rank-five odd shell."""

    allowed_names = {decision.name for decision in retarded_allowed_decisions()}
    odd_names = {channel.name for channel in odd_shell_subset().channels}
    return (
        direct_identity_vetoed()
        and allowed_names == odd_names
        and retarded_down_counts() == primitive_counts()
        and retarded_down_profile()
        == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
    )


def quark_down_identity_veto_payload() -> QuarkDownIdentityVetoPayload:
    """Return the Session 23 down identity-return veto payload."""

    active = quark_active_current_readout_payload()
    parity = quark_current_parity_selector_payload()
    odd = quark_down_odd_shell_payload()
    decisions = primitive_return_decisions()
    identities = direct_identity_decisions()
    allowed = retarded_allowed_decisions()
    rejected = retarded_rejected_decisions()
    retarded_counts = retarded_down_counts()
    retarded_profile = retarded_down_profile()
    baseline_counts = down_baseline_counts()
    baseline_profile = down_baseline_clebsch_vector()
    allowed_breakdown = allowed_return_breakdown()

    active_pass = active.final_verdict == "QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS"
    parity_pass = parity.final_verdict == "QUARK_CURRENT_PARITY_SELECTOR_PASS"
    odd_pass = odd.final_verdict == "QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS"
    unique_identity = direct_identity_is_unique()
    identity_vetoed = direct_identity_vetoed()
    allowed_odd = all_allowed_returns_are_odd()
    baseline_rejected = baseline_control_rejected_by_retarded_predicate()
    rank_five_selected = rank_five_selected_inside_retarded_model()
    premise_reduced = rank_five_selected and DOWN_IDENTITY_VETO_PREMISE in active.remaining_physical_inputs

    checks_pass = (
        active_pass
        and parity_pass
        and odd_pass
        and unique_identity
        and identity_vetoed
        and allowed_odd
        and len(allowed) == 5
        and allowed_breakdown
        == {"bcc_odd": 2, "color_odd": 3, "odd_total": 5, "direct_even": 0}
        and retarded_counts == {"d": 6, "s": 2, "b": 5}
        and retarded_counts == primitive_counts()
        and retarded_profile
        == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
        and baseline_counts == {"d": 6, "s": 2, "b": 4}
        and baseline_profile
        == (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)))
        and baseline_rejected
        and rank_five_selected
        and premise_reduced
    )

    if checks_pass:
        final_verdict = "DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS"
        interpretation = (
            "Inside the retarded down-current model, a mass return must leave "
            "the visible sheet before returning.  The primitive shell has one "
            "zero-excursion identity/direct line, and the criterion vetoes it. "
            "The allowed returns are exactly the five odd current channels "
            "2_BCC+3_color, so the down fork selects the rank-five bottom "
            "profile (1,1/sqrt(3),sqrt(5/6)).  The physical retarded-current "
            "criterion is still a premise, not a derivation from bare BB "
            "blocks."
        )
    else:
        final_verdict = "DOWN_IDENTITY_RETURN_VETO_AUDIT_KILL"
        interpretation = (
            "The active-current, current-parity, odd-shell, identity-line, "
            "retarded-return, count, profile, or baseline-control checks "
            "failed."
        )

    return QuarkDownIdentityVetoPayload(
        final_verdict=final_verdict,
        active_current_readout_pass=active_pass,
        current_parity_selector_pass=parity_pass,
        odd_shell_pass=odd_pass,
        primitive_decisions=decisions,
        direct_identity_names=tuple(decision.name for decision in identities),
        retarded_allowed_names=tuple(decision.name for decision in allowed),
        retarded_rejected_names=tuple(decision.name for decision in rejected),
        direct_identity_is_unique=unique_identity,
        direct_identity_vetoed=identity_vetoed,
        all_allowed_returns_are_odd=allowed_odd,
        allowed_return_count=len(allowed),
        allowed_return_breakdown=allowed_breakdown,
        retarded_counts=retarded_counts,
        retarded_profile=retarded_profile,
        baseline_counts=baseline_counts,
        baseline_profile=baseline_profile,
        baseline_control_rejected_by_retarded_predicate=baseline_rejected,
        rank_five_selected_inside_retarded_model=rank_five_selected,
        down_identity_veto_premise_reduced=premise_reduced,
        microscopic_bare_bcc_derivation=False,
        remaining_physical_premises=(
            ODD_CURRENT_PHYSICAL_PREMISE,
            RETARDED_DOWN_CURRENT_PREMISE,
            BOTTOM_ODD_SHELL_READOUT_PREMISE,
        ),
        interpretation=interpretation,
    )
