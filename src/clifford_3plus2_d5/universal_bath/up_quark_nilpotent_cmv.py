"""Session 06 up-quark nilpotent CMV head.

The up-sector source vector is not frozen in the universal-bath dictionary.
This module therefore proves only the conditional finite-head statement:
the BB first-hop survival branch injects a nilpotent length-3 Taylor head with
``x=1/sqrt(2)``, producing ``(1/4,1/sqrt(2),1)`` followed by the free CMV tail.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.radial_response.up_stacking import (
    geometric_stack_vector,
    up_stacking_payload,
)
from clifford_3plus2_d5.scalar_clebsch.taylor_up import (
    nilpotent_flag,
    nilpotent_order_is_three,
    old_up_clebsch_vector,
    taylor_kernel_matrix,
    taylor_shell_profile,
    taylor_up_audit_payload,
    up_clebsch_vector,
)
from clifford_3plus2_d5.universal_bath.opuc import (
    free_verblunsky_tail,
    is_free_verblunsky_tail,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceAnchor,
    SourceStatus,
    bb_first_hop_survival_operator,
    source_dictionary_payload,
)

UP_QUARK_SOURCE_LABEL = "up_quark_boundary_source"


@dataclass(frozen=True)
class UpQuarkNilpotentCMVPayload:
    """Session 06 up-quark nilpotent finite-head verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    quark_source_unresolved: bool
    source_label: str
    source_reduction: ReductionKind
    survival_operator: sp.Matrix
    survival_weight: sp.Expr
    injection_amplitude: sp.Expr
    scalar_clebsch_prerequisite_pass: bool
    up_stacking_prerequisite_pass: bool
    nilpotent_flag: sp.Matrix
    nilpotent_order_three: bool
    taylor_kernel: sp.Matrix
    taylor_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    expected_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    taylor_profile_matches: bool
    geometric_control_profile: tuple[sp.Expr, sp.Expr, sp.Expr]
    geometric_control_rejected: bool
    old_sqrt2_control: tuple[sp.Expr, sp.Expr, sp.Expr]
    old_sqrt2_control_rejected: bool
    finite_verblunsky_head: tuple[sp.Expr, ...]
    finite_head_inside_unit_disk: bool
    free_tail_after_head: bool
    full_quark_source_not_derived: bool
    interpretation: str


def unresolved_up_quark_source() -> SourceAnchor:
    """Return the unresolved Session 02 up-quark source anchor."""

    payload = source_dictionary_payload()
    anchors = {anchor.label: anchor for anchor in payload.unresolved_sources}
    return anchors[UP_QUARK_SOURCE_LABEL]


def bb_survival_weight() -> sp.Expr:
    """Return the normalized same-normal BB first-hop survival weight."""

    survival = bb_first_hop_survival_operator()
    if survival != sp.eye(2) / 2:
        raise ValueError("BB first-hop survival operator is not I/2")
    return sp.Rational(1, 2)


def tail_injection_amplitude() -> sp.Expr:
    """Return the injection amplitude forced by the BB survival branch."""

    return sp.sqrt(bb_survival_weight())


def up_nilpotent_taylor_profile() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the up profile from the survival-injected nilpotent head."""

    return taylor_shell_profile(tail_injection_amplitude())


def up_nilpotent_taylor_kernel() -> sp.Matrix:
    """Return ``exp(xN)`` with ``x`` fixed by the BB survival branch."""

    return taylor_kernel_matrix(tail_injection_amplitude())


def geometric_control_profile() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the non-Taylor geometric control at the same injection amplitude."""

    return geometric_stack_vector(tail_injection_amplitude())


def finite_nilpotent_verblunsky_head() -> tuple[sp.Expr, ...]:
    """Return a finite CMV/OPUC head encoding the nilpotent entry amplitudes."""

    x = tail_injection_amplitude()
    return (sp.simplify(x), sp.simplify(x**2 / 2))


def finite_head_inside_unit_disk(coefficients: tuple[sp.Expr, ...] | None = None) -> bool:
    """Return whether every finite CMV coefficient has modulus below one."""

    selected = finite_nilpotent_verblunsky_head() if coefficients is None else coefficients
    return all(bool(sp.N(abs(coefficient), 50) < 1) for coefficient in selected)


def up_nilpotent_verblunsky_sequence(tail_length: int = 4) -> tuple[sp.Expr, ...]:
    """Return finite nilpotent head followed by the universal free CMV tail."""

    return (*finite_nilpotent_verblunsky_head(), *free_verblunsky_tail(tail_length))


def up_quark_nilpotent_cmv_payload() -> UpQuarkNilpotentCMVPayload:
    """Return the Session 06 up-quark nilpotent CMV verdict."""

    source_payload = source_dictionary_payload()
    source = unresolved_up_quark_source()
    scalar_payload = taylor_up_audit_payload()
    stacking_payload = up_stacking_payload()
    survival = bb_first_hop_survival_operator()
    weight = bb_survival_weight()
    injection = tail_injection_amplitude()
    profile = up_nilpotent_taylor_profile()
    expected = up_clebsch_vector()
    geometric = geometric_control_profile()
    old_control = old_up_clebsch_vector()
    head = finite_nilpotent_verblunsky_head()
    coefficients = up_nilpotent_verblunsky_sequence()
    free_tail = is_free_verblunsky_tail(coefficients[len(head) :])

    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    source_unresolved = (
        source.label == UP_QUARK_SOURCE_LABEL
        and source.status == SourceStatus.UNRESOLVED
        and source.reduction == ReductionKind.CMV_OPUC
        and source.port_vector is None
        and source.normal_depth is None
    )
    scalar_pass = scalar_payload.final_verdict == "NILPOTENT_TAYLOR_UP_CLEBSCH_PASS"
    stacking_pass = stacking_payload.final_verdict == "UP_STACKING_LAW_EXPONENTIAL_FAVORED"
    profile_matches = profile == expected
    geometric_rejected = geometric != expected
    old_rejected = old_control != expected and old_control[1] == sp.sqrt(2)
    inside_disk = finite_head_inside_unit_disk(head)
    nilpotent_order = nilpotent_order_is_three()

    checks_pass = (
        source_pass
        and source_unresolved
        and survival == sp.eye(2) / 2
        and weight == sp.Rational(1, 2)
        and sp.simplify(injection - 1 / sp.sqrt(2)) == 0
        and scalar_pass
        and stacking_pass
        and nilpotent_order
        and up_nilpotent_taylor_kernel() == taylor_kernel_matrix(1 / sp.sqrt(2))
        and profile_matches
        and geometric_rejected
        and old_rejected
        and head == (1 / sp.sqrt(2), sp.Rational(1, 4))
        and inside_disk
        and free_tail
    )

    if checks_pass:
        final_verdict = "UP_NILPOTENT_CMV_HEAD_CONDITIONAL_PASS"
        interpretation = (
            "The BB same-normal survival branch gives weight 1/2 and injection "
            "amplitude x=1/sqrt(2).  Feeding that value into the length-3 "
            "nilpotent Taylor head exp(xN) gives the up profile "
            "(1/4,1/sqrt(2),1), while the geometric and old sqrt(2) controls "
            "fail.  The finite head lies inside the CMV disk and is followed "
            "by the free universal tail.  This is conditional because the "
            "up-quark BCC source vector remains intentionally unresolved."
        )
    else:
        final_verdict = "UP_NILPOTENT_CMV_HEAD_KILL"
        interpretation = (
            "The up nilpotent CMV head failed the source-prerequisite, "
            "survival-injection, Taylor, control, disk, or free-tail checks."
        )

    return UpQuarkNilpotentCMVPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        quark_source_unresolved=source_unresolved,
        source_label=source.label,
        source_reduction=source.reduction,
        survival_operator=survival,
        survival_weight=weight,
        injection_amplitude=injection,
        scalar_clebsch_prerequisite_pass=scalar_pass,
        up_stacking_prerequisite_pass=stacking_pass,
        nilpotent_flag=nilpotent_flag(),
        nilpotent_order_three=nilpotent_order,
        taylor_kernel=up_nilpotent_taylor_kernel(),
        taylor_profile=profile,
        expected_profile=expected,
        taylor_profile_matches=profile_matches,
        geometric_control_profile=geometric,
        geometric_control_rejected=geometric_rejected,
        old_sqrt2_control=old_control,
        old_sqrt2_control_rejected=old_rejected,
        finite_verblunsky_head=head,
        finite_head_inside_unit_disk=inside_disk,
        free_tail_after_head=free_tail,
        full_quark_source_not_derived=True,
        interpretation=interpretation,
    )
