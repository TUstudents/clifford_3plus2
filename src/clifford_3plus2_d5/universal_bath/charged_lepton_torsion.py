"""Session 05 charged-lepton ``2/9`` torsion gate.

This module separates the exact source-geometry fact from the stronger
unproved dynamical claim.  The frozen charged-lepton port has occupations
``p_a=2/3`` and ``p_u=1/3``.  Their incoherent two-channel transition weight is
``p_a p_u = 2/9``.  The coherent amplitude is instead ``sqrt(2)/3`` and is
explicitly rejected as the torsion value.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.universal_bath.charged_lepton_cmv import (
    CHARGED_LEPTON_SOURCE_LABEL,
    charged_lepton_source_components,
    frozen_charged_lepton_source,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceStatus,
    source_dictionary_payload,
)


@dataclass(frozen=True)
class ChargedLeptonTorsionPayload:
    """Session 05 charged-lepton torsion verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    source_label: str
    source_reduction: ReductionKind
    residual_components: dict[str, sp.Expr]
    occupation_weights: dict[str, sp.Expr]
    occupation_weights_normalized: bool
    b_occupation_zero: bool
    torsion_transition_weight: sp.Expr
    expected_torsion_weight: sp.Expr
    coherent_transition_amplitude: sp.Expr
    coherent_amplitude_rejected: bool
    equal_weight_control: sp.Expr
    equal_weight_control_rejected: bool
    one_port_controls_rejected: bool
    cmv_phase_not_rederived: bool
    interpretation: str


def source_occupation_weights() -> dict[str, sp.Expr]:
    """Return exact port occupation weights of the frozen charged-lepton source."""

    components = charged_lepton_source_components()
    return {
        label: sp.simplify(component * sp.conjugate(component))
        for label, component in components.items()
    }


def occupation_weights_normalized(weights: dict[str, sp.Expr] | None = None) -> bool:
    """Return whether the occupation weights sum to one."""

    selected = source_occupation_weights() if weights is None else weights
    return sp.simplify(sum(selected.values(), sp.Integer(0)) - 1) == 0


def torsion_transition_weight(
    first: str = "a",
    second: str = "u",
    weights: dict[str, sp.Expr] | None = None,
) -> sp.Expr:
    """Return the incoherent two-channel occupation transition weight."""

    selected = source_occupation_weights() if weights is None else weights
    return sp.simplify(selected[first] * selected[second])


def charged_lepton_torsion_weight() -> sp.Expr:
    """Return the charged-lepton ``2/9`` torsion candidate."""

    return torsion_transition_weight("a", "u")


def coherent_transition_amplitude() -> sp.Expr:
    """Return the coherent ``a-u`` amplitude control."""

    components = charged_lepton_source_components()
    return sp.simplify(components["a"] * sp.conjugate(components["u"]))


def equal_weight_transition_control() -> sp.Expr:
    """Return the transition weight from an equal two-port control."""

    control = {"a": sp.Rational(1, 2), "u": sp.Rational(1, 2), "b": sp.Integer(0)}
    return torsion_transition_weight("a", "u", control)


def one_port_transition_controls_rejected() -> bool:
    """Return whether one-port controls fail to reproduce ``2/9``."""

    target = sp.Rational(2, 9)
    a_only = {"a": sp.Integer(1), "u": sp.Integer(0), "b": sp.Integer(0)}
    u_only = {"a": sp.Integer(0), "u": sp.Integer(1), "b": sp.Integer(0)}
    return (
        torsion_transition_weight("a", "u", a_only) != target
        and torsion_transition_weight("a", "u", u_only) != target
    )


def charged_lepton_torsion_payload() -> ChargedLeptonTorsionPayload:
    """Return the Session 05 charged-lepton torsion verdict."""

    source_payload = source_dictionary_payload()
    source = frozen_charged_lepton_source()
    components = charged_lepton_source_components()
    occupations = source_occupation_weights()
    torsion = charged_lepton_torsion_weight()
    coherent = coherent_transition_amplitude()
    equal_control = equal_weight_transition_control()
    expected = sp.Rational(2, 9)

    source_pass = (
        source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
        and source.label == CHARGED_LEPTON_SOURCE_LABEL
        and source.status == SourceStatus.FROZEN
        and source.reduction == ReductionKind.CMV_OPUC
    )
    weights_normalized = occupation_weights_normalized(occupations)
    b_zero = sp.simplify(occupations["b"]) == 0
    coherent_rejected = sp.simplify(coherent - expected) != 0
    equal_rejected = sp.simplify(equal_control - expected) != 0
    one_port_rejected = one_port_transition_controls_rejected()

    checks_pass = (
        source_pass
        and sp.simplify(components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
        and sp.simplify(components["u"] - 1 / sp.sqrt(3)) == 0
        and components["b"] == 0
        and occupations == {"a": sp.Rational(2, 3), "u": sp.Rational(1, 3), "b": 0}
        and weights_normalized
        and b_zero
        and sp.simplify(torsion - expected) == 0
        and sp.simplify(torsion_transition_weight("u", "a") - expected) == 0
        and coherent_rejected
        and equal_rejected
        and one_port_rejected
    )

    if checks_pass:
        final_verdict = "CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS"
        interpretation = (
            "The frozen charged-lepton source e1 has occupation weights "
            "p_a=2/3 and p_u=1/3, so the incoherent a-u transition weight is "
            "p_a p_u=2/9.  Thus 2/9 jumps out as a source-geometry torsion "
            "moment.  The coherent amplitude is sqrt(2)/3, not 2/9, so this "
            "session does not claim to derive the CMV phase or the full "
            "charged-lepton mass angle from the torsion alone."
        )
    else:
        final_verdict = "CHARGED_LEPTON_TORSION_2_OVER_9_KILL"
        interpretation = (
            "The charged-lepton torsion gate failed the frozen-source, "
            "occupation, transition-weight, or negative-control checks."
        )

    return ChargedLeptonTorsionPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        source_label=source.label,
        source_reduction=source.reduction,
        residual_components=components,
        occupation_weights=occupations,
        occupation_weights_normalized=weights_normalized,
        b_occupation_zero=b_zero,
        torsion_transition_weight=torsion,
        expected_torsion_weight=expected,
        coherent_transition_amplitude=coherent,
        coherent_amplitude_rejected=coherent_rejected,
        equal_weight_control=equal_control,
        equal_weight_control_rejected=equal_rejected,
        one_port_controls_rejected=one_port_rejected,
        cmv_phase_not_rederived=True,
        interpretation=interpretation,
    )
