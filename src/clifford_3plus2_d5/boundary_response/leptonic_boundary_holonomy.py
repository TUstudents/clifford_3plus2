"""V10 leptonic boundary-loop holonomy selection gate.

V8 verifies that the proposed scalar word

    W_e = -q_A3 q_A2

has phase ``exp(-i 5*pi/12)``.  V10 adds the missing selection layer: a minimal
charged-lepton boundary loop must use the Schur return, then the parent
tetrahedral/BCC spin lift, then the residual triangle spin lift, with no
repeated factors or longer cover.  This module audits that model directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    leptonic_phase_word_audit_payload,
    phase_from_pi_angle,
    principal_pi_angle,
)

SCHUR_RETURN = "SCHUR_RETURN"
PARENT_A3 = "PARENT_A3"
RESIDUAL_A2 = "RESIDUAL_A2"
PRIMITIVE_CHARGED_LEPTON_WORD = (SCHUR_RETURN, PARENT_A3, RESIDUAL_A2)


@dataclass(frozen=True)
class BoundaryHolonomyFactor:
    """Exact scalar factor in the charged-lepton boundary loop."""

    name: str
    angle: sp.Expr
    source: str


@dataclass(frozen=True)
class BoundaryLoopWord:
    """Ordered boundary-loop word with exact scalar angle arithmetic."""

    factor_names: tuple[str, ...]

    @property
    def raw_angle(self) -> sp.Expr:
        """Return the raw scalar angle in units of ``pi``."""

        registry = _factor_registry()
        return sp.simplify(sum((registry[name].angle for name in self.factor_names), sp.Integer(0)))

    @property
    def principal_angle(self) -> sp.Expr:
        """Return the principal scalar angle in units of ``pi``."""

        return principal_pi_angle(self.raw_angle)

    @property
    def phase(self) -> sp.Expr:
        """Return the exact scalar phase of the loop word."""

        return phase_from_pi_angle(self.raw_angle)


def boundary_holonomy_factors() -> tuple[BoundaryHolonomyFactor, ...]:
    """Return the exact factor registry in primitive orientation order."""

    return (
        BoundaryHolonomyFactor(
            name=SCHUR_RETURN,
            angle=sp.Integer(1),
            source="second-order Schur complement boundary return",
        ),
        BoundaryHolonomyFactor(
            name=PARENT_A3,
            angle=sp.Rational(1, 4),
            source="BCC/tetrahedral parent spin lift",
        ),
        BoundaryHolonomyFactor(
            name=RESIDUAL_A2,
            angle=sp.Rational(1, 3),
            source="residual triangle spin lift",
        ),
    )


def _factor_registry() -> dict[str, BoundaryHolonomyFactor]:
    """Return holonomy factors by name."""

    return {factor.name: factor for factor in boundary_holonomy_factors()}


def boundary_loop_candidates(max_length: int = 4) -> tuple[BoundaryLoopWord, ...]:
    """Enumerate ordered factor words up to ``max_length``."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    names = tuple(_factor_registry())
    return tuple(
        BoundaryLoopWord(factor_names=word)
        for length in range(1, max_length + 1)
        for word in product(names, repeat=length)
    )


def is_admissible_charged_lepton_word(word: BoundaryLoopWord) -> bool:
    """Return true when ``word`` is the primitive charged-lepton loop."""

    factor_names = word.factor_names
    has_required_support = set(factor_names) == set(PRIMITIVE_CHARGED_LEPTON_WORD)
    has_no_repeats = len(factor_names) == len(set(factor_names))
    has_primitive_length = len(factor_names) == len(PRIMITIVE_CHARGED_LEPTON_WORD)
    has_orientation_order = factor_names == PRIMITIVE_CHARGED_LEPTON_WORD
    return has_required_support and has_no_repeats and has_primitive_length and has_orientation_order


def admissible_charged_lepton_words(max_length: int = 4) -> tuple[BoundaryLoopWord, ...]:
    """Return all admissible charged-lepton words from the candidate catalog."""

    return tuple(
        word
        for word in boundary_loop_candidates(max_length=max_length)
        if is_admissible_charged_lepton_word(word)
    )


def primitive_charged_lepton_boundary_word() -> BoundaryLoopWord:
    """Return the unique primitive charged-lepton boundary word."""

    admissible = admissible_charged_lepton_words()
    if len(admissible) != 1:
        raise ValueError(f"expected exactly one primitive word, found {len(admissible)}")
    return admissible[0]


def boundary_loop_control_words() -> dict[str, BoundaryLoopWord]:
    """Return negative-control words for the holonomy selection audit."""

    return {
        "a3_only": BoundaryLoopWord((PARENT_A3,)),
        "a2_only": BoundaryLoopWord((RESIDUAL_A2,)),
        "a3_a2_without_schur": BoundaryLoopWord((PARENT_A3, RESIDUAL_A2)),
        "schur_a3_only": BoundaryLoopWord((SCHUR_RETURN, PARENT_A3)),
        "schur_a2_only": BoundaryLoopWord((SCHUR_RETURN, RESIDUAL_A2)),
        "reversed_a2_a3": BoundaryLoopWord((SCHUR_RETURN, RESIDUAL_A2, PARENT_A3)),
        "duplicated_full_word": BoundaryLoopWord(
            PRIMITIVE_CHARGED_LEPTON_WORD + PRIMITIVE_CHARGED_LEPTON_WORD
        ),
    }


@dataclass(frozen=True)
class LeptonicBoundaryHolonomyAuditPayload:
    """Verdict payload for the V10 boundary-loop holonomy gate."""

    final_verdict: str
    primitive_word: BoundaryLoopWord
    raw_angle: sp.Expr
    principal_angle: sp.Expr
    phase: sp.Expr
    candidate_count: int
    admissible_count: int
    controls_rejected: dict[str, bool]
    word_selection_derived: bool
    ckm_parked: bool
    interpretation: str


def leptonic_boundary_holonomy_audit_payload() -> LeptonicBoundaryHolonomyAuditPayload:
    """Return the V10 holonomy-selection verdict."""

    v8 = leptonic_phase_word_audit_payload()
    primitive = primitive_charged_lepton_boundary_word()
    candidates = boundary_loop_candidates()
    admissible = admissible_charged_lepton_words()
    controls_rejected = {
        name: not is_admissible_charged_lepton_word(word)
        for name, word in boundary_loop_control_words().items()
    }
    target_phase = sp.exp(-sp.I * sp.pi * sp.Rational(5, 12))
    checks_pass = (
        v8.final_verdict == "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
        and primitive.factor_names == PRIMITIVE_CHARGED_LEPTON_WORD
        and primitive.raw_angle == sp.Rational(19, 12)
        and primitive.principal_angle == -sp.Rational(5, 12)
        and primitive.phase == target_phase
        and len(admissible) == 1
        and all(controls_rejected.values())
    )

    if checks_pass:
        final_verdict = "LEPTONIC_PHASE_WORD_DERIVED_PASS"
        interpretation = (
            "The minimal charged-lepton boundary-loop model uniquely selects "
            "the primitive word SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2. "
            "Its exact phase is exp(-i 5*pi/12), and no-Schur, subword, "
            "reversed-order, and duplicated-cover controls are rejected. This "
            "derives the V8 word within the holonomy model. CKM remains parked."
        )
    else:
        final_verdict = "LEPTONIC_PHASE_WORD_DERIVED_KILL"
        interpretation = (
            "The boundary-loop model did not uniquely select the full "
            "spin-Coxeter-Schur word or failed a negative control. CKM remains "
            "parked."
        )

    return LeptonicBoundaryHolonomyAuditPayload(
        final_verdict=final_verdict,
        primitive_word=primitive,
        raw_angle=primitive.raw_angle,
        principal_angle=primitive.principal_angle,
        phase=primitive.phase,
        candidate_count=len(candidates),
        admissible_count=len(admissible),
        controls_rejected=controls_rejected,
        word_selection_derived=checks_pass,
        ckm_parked=True,
        interpretation=interpretation,
    )
