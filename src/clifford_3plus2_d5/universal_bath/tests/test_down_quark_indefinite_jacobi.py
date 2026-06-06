"""Tests for the Session 07 down-quark indefinite Jacobi head."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.down_quark_indefinite_jacobi import (
    DOWN_QUARK_SOURCE_LABEL,
    down_quark_indefinite_jacobi_payload,
    regular_s3_baseline_head,
    regular_s3_candidate_head,
    signature_breakdown_control_detected,
    signed_norm_squared,
    three_port_cannot_host_rank_five_candidate,
    three_port_permutation_head,
    unresolved_down_quark_source,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import SourceStatus


def test_down_quark_source_remains_unresolved_in_write_once_dictionary() -> None:
    source = unresolved_down_quark_source()

    assert source.label == DOWN_QUARK_SOURCE_LABEL
    assert source.status == SourceStatus.UNRESOLVED
    assert source.reduction == ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI
    assert source.port_vector is None
    assert source.normal_depth is None


def test_three_port_and_regular_shells_reproduce_clean_baseline() -> None:
    expected = (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)))
    three_port = three_port_permutation_head()
    regular = regular_s3_baseline_head()

    assert three_port.ranks == {"d": 3, "s": 1, "b": 2}
    assert three_port.denominator == 3
    assert three_port.clebsch_vector == expected
    assert three_port.matches_baseline
    assert regular.ranks == {"d": 6, "s": 2, "b": 4}
    assert regular.denominator == 6
    assert regular.clebsch_vector == expected
    assert regular.matches_baseline


def test_rank_five_candidate_requires_regular_s3_shell_and_is_not_forced() -> None:
    expected = (sp.Integer(1), 1 / sp.sqrt(3), sp.sqrt(sp.Rational(5, 6)))
    candidate = regular_s3_candidate_head()

    assert three_port_cannot_host_rank_five_candidate()
    assert candidate.ranks == {"d": 6, "s": 2, "b": 5}
    assert candidate.denominator == 6
    assert candidate.clebsch_vector == expected
    assert candidate.matches_candidate
    assert not candidate.forced_by_symmetry


def test_signature_breakdown_control_is_detected_not_clamped() -> None:
    assert signed_norm_squared((sp.Integer(0), sp.Integer(1)), (1, -1)) == -1
    assert signature_breakdown_control_detected()


def test_down_quark_indefinite_jacobi_payload_reports_conditional_pass() -> None:
    payload = down_quark_indefinite_jacobi_payload()

    assert payload.final_verdict == "DOWN_INDEFINITE_JACOBI_HEAD_CONDITIONAL_PASS"
    assert payload.source_dictionary_pass
    assert payload.quark_source_unresolved
    assert payload.subset_prerequisite_pass
    assert payload.s3_projector_prerequisite_pass
    assert payload.three_port_head.matches_baseline
    assert payload.regular_baseline_head.matches_baseline
    assert payload.regular_candidate_head.matches_candidate
    assert payload.three_port_cannot_host_candidate
    assert payload.regular_candidate_available
    assert not payload.regular_candidate_forced_by_s3_alone
    assert payload.rank_two_requires_defect_polarization
    assert payload.rank_five_not_unique
    assert payload.signature_breakdown_control_detected
    assert payload.selected_physical_head == "unselected_open_rank_five_gate"
