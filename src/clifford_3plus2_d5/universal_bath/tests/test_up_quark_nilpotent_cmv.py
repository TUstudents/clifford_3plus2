"""Tests for the Session 06 up-quark nilpotent CMV head."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.opuc import is_free_verblunsky_tail
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import SourceStatus
from clifford_3plus2_d5.universal_bath.up_quark_nilpotent_cmv import (
    UP_QUARK_SOURCE_LABEL,
    bb_survival_weight,
    finite_head_inside_unit_disk,
    finite_nilpotent_verblunsky_head,
    geometric_control_profile,
    tail_injection_amplitude,
    unresolved_up_quark_source,
    up_nilpotent_taylor_kernel,
    up_nilpotent_taylor_profile,
    up_nilpotent_verblunsky_sequence,
    up_quark_nilpotent_cmv_payload,
)


def test_up_quark_source_remains_unresolved_in_write_once_dictionary() -> None:
    source = unresolved_up_quark_source()

    assert source.label == UP_QUARK_SOURCE_LABEL
    assert source.status == SourceStatus.UNRESOLVED
    assert source.reduction == ReductionKind.CMV_OPUC
    assert source.port_vector is None
    assert source.normal_depth is None


def test_bb_survival_forces_one_over_sqrt_two_injection() -> None:
    assert bb_survival_weight() == sp.Rational(1, 2)
    assert sp.simplify(tail_injection_amplitude() - 1 / sp.sqrt(2)) == 0


def test_nilpotent_taylor_profile_matches_up_clebsches() -> None:
    expected = (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))

    assert up_nilpotent_taylor_profile() == expected
    assert up_nilpotent_taylor_kernel()[0, 2] == sp.Rational(1, 4)


def test_geometric_control_is_rejected_at_same_injection() -> None:
    assert geometric_control_profile() == (sp.Rational(1, 2), 1 / sp.sqrt(2), sp.Integer(1))
    assert geometric_control_profile() != up_nilpotent_taylor_profile()


def test_finite_nilpotent_head_is_inside_disk_then_free_tail() -> None:
    head = finite_nilpotent_verblunsky_head()
    sequence = up_nilpotent_verblunsky_sequence()

    assert head == (1 / sp.sqrt(2), sp.Rational(1, 4))
    assert finite_head_inside_unit_disk(head)
    assert is_free_verblunsky_tail(sequence[len(head) :])


def test_up_quark_nilpotent_cmv_payload_reports_conditional_pass() -> None:
    payload = up_quark_nilpotent_cmv_payload()

    assert payload.final_verdict == "UP_NILPOTENT_HEAD_CONDITIONAL_PASS"
    assert payload.source_dictionary_pass
    assert payload.quark_source_unresolved
    assert payload.scalar_clebsch_prerequisite_pass
    assert payload.up_stacking_prerequisite_pass
    assert payload.nilpotent_order_three
    assert payload.taylor_profile_matches
    assert payload.geometric_control_rejected
    assert payload.old_sqrt2_control_rejected
    assert payload.finite_head_inside_unit_disk
    assert payload.free_tail_after_head
    assert payload.full_quark_source_not_derived
