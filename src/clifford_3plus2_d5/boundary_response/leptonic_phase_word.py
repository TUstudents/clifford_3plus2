"""V8 conditional leptonic phase-word audit.

V7 derives the charged-lepton leakage scalar but leaves the CP phase parked.
V8 audits only the exact arithmetic of the proposed spin-Coxeter-Schur word

    W_e = -q_A3 q_A2.

The BCC/tetrahedral parent spin lift contributes ``pi/4``.  The residual
triangle spin lift contributes ``pi/3``.  The second-order Schur complement
contributes a minus sign, represented as an added ``pi`` phase.  Therefore the
raw angle is

    1 + 1/4 + 1/3 = 19/12

in units of ``pi``, whose principal representative is ``-5/12``.

This module does not derive the boundary loop selecting the full word and does
not assemble PMNS.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def coxeter_spin_lift_angle(name: str) -> sp.Rational:
    """Return the spin-lift angle in units of ``pi`` for a Coxeter factor."""

    match name:
        case "A3":
            return sp.Rational(1, 4)
        case "A2":
            return sp.Rational(1, 3)
        case _:
            raise ValueError(f"unknown Coxeter spin-lift factor: {name}")


def second_order_schur_angle() -> sp.Integer:
    """Return the minus-sign Schur contribution as a ``pi`` angle."""

    return sp.Integer(1)


def principal_pi_angle(angle: sp.Expr) -> sp.Expr:
    """Return the principal representative of a phase angle in units of ``pi``."""

    principal = sp.simplify((sp.simplify(angle) + 1) % 2 - 1)
    if principal == -1:
        return sp.Integer(1)
    return principal


def phase_from_pi_angle(angle: sp.Expr) -> sp.Expr:
    """Return ``exp(i pi angle)`` using the principal phase representative."""

    return sp.exp(sp.I * sp.pi * principal_pi_angle(angle))


def raw_leptonic_phase_word_angle(
    *,
    include_schur: bool = True,
    word: tuple[str, ...] = ("A3", "A2"),
) -> sp.Expr:
    """Return the raw scalar word angle in units of ``pi``."""

    angle = sum((coxeter_spin_lift_angle(name) for name in word), sp.Integer(0))
    if include_schur:
        angle += second_order_schur_angle()
    return sp.simplify(angle)


def leptonic_phase_word_angle(
    *,
    include_schur: bool = True,
    word: tuple[str, ...] = ("A3", "A2"),
) -> sp.Expr:
    """Return the principal scalar word angle in units of ``pi``."""

    return principal_pi_angle(raw_leptonic_phase_word_angle(include_schur=include_schur, word=word))


def leptonic_phase_word_phase(
    *,
    include_schur: bool = True,
    word: tuple[str, ...] = ("A3", "A2"),
) -> sp.Expr:
    """Return the exact phase for the scalar word."""

    return phase_from_pi_angle(
        raw_leptonic_phase_word_angle(include_schur=include_schur, word=word)
    )


@dataclass(frozen=True)
class LeptonicPhaseWordAuditPayload:
    """Verdict payload for the V8 conditional leptonic phase-word audit."""

    final_verdict: str
    raw_full_angle: sp.Expr
    principal_full_angle: sp.Expr
    full_phase: sp.Expr
    cp_conjugate_angle: sp.Expr
    no_schur_matches: bool
    a3_only_matches: bool
    a2_only_matches: bool
    word_selection_derived: bool
    pmns_ckm_parked: bool
    interpretation: str


def leptonic_phase_word_audit_payload() -> LeptonicPhaseWordAuditPayload:
    """Return the V8 conditional phase-word verdict."""

    target = -sp.Rational(5, 12)
    raw_full = raw_leptonic_phase_word_angle()
    principal_full = leptonic_phase_word_angle()
    full_phase = leptonic_phase_word_phase()
    cp_conjugate = principal_pi_angle(-principal_full)

    no_schur_matches = leptonic_phase_word_angle(include_schur=False) == target
    a3_only_matches = leptonic_phase_word_angle(word=("A3",)) == target
    a2_only_matches = leptonic_phase_word_angle(word=("A2",)) == target
    phase_matches = full_phase == sp.exp(-sp.I * sp.pi * sp.Rational(5, 12))

    arithmetic_pass = (
        raw_full == sp.Rational(19, 12)
        and principal_full == target
        and phase_matches
        and cp_conjugate == sp.Rational(5, 12)
        and not no_schur_matches
        and not a3_only_matches
        and not a2_only_matches
    )

    if arithmetic_pass:
        final_verdict = "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
        interpretation = (
            "The proposed scalar spin-Coxeter-Schur word has exact phase "
            "W_e = -q_A3 q_A2 = exp(-i 5*pi/12), and the no-Schur and "
            "single-factor subword controls differ. This verifies only the "
            "conditional phase arithmetic; selecting the full boundary word "
            "still requires an explicit holonomy theorem. PMNS/CKM remain "
            "parked."
        )
    else:
        final_verdict = "LEPTONIC_PHASE_WORD_KILL"
        interpretation = (
            "The proposed phase word failed exact arithmetic or subword "
            "controls. PMNS/CKM remain parked."
        )

    return LeptonicPhaseWordAuditPayload(
        final_verdict=final_verdict,
        raw_full_angle=raw_full,
        principal_full_angle=principal_full,
        full_phase=full_phase,
        cp_conjugate_angle=cp_conjugate,
        no_schur_matches=no_schur_matches,
        a3_only_matches=a3_only_matches,
        a2_only_matches=a2_only_matches,
        word_selection_derived=False,
        pmns_ckm_parked=True,
        interpretation=interpretation,
    )
