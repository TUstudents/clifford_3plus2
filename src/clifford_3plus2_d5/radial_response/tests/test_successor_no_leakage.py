"""Tests for the R7 scalar successor/no-leakage certificate."""

from clifford_3plus2_d5.radial_response.successor_no_leakage import (
    ScalarFailureClass,
    ScalarSuccessorVerdict,
    ScalarSuccessorVeto,
    scalar_allowed_successors,
    scalar_forbidden_rows_have_vetoes,
    scalar_leakage_controls_rejected,
    scalar_successor_certificate,
    scalar_successor_certificate_is_complete,
    scalar_successor_certificate_payload,
    scalar_successor_certificate_row,
    scalar_successor_candidate_basis,
    scalar_successors_are_z2_pair,
    scalar_third_successor_control_rejected,
)


def test_scalar_successor_certificate_is_complete() -> None:
    assert len(scalar_successor_certificate()) == len(scalar_successor_candidate_basis())
    assert scalar_successor_certificate_is_complete()
    assert scalar_forbidden_rows_have_vetoes()


def test_only_two_z2_conjugate_successors_are_allowed() -> None:
    assert scalar_allowed_successors() == ("triality_plus", "triality_minus")
    assert scalar_successors_are_z2_pair()
    for label in scalar_allowed_successors():
        row = scalar_successor_certificate_row(label)
        assert row.verdict == ScalarSuccessorVerdict.ALLOW
        assert row.vetoes == ()
        assert row.failure_class == ScalarFailureClass.ALLOW


def test_leakage_and_asymmetry_controls_are_vetoed() -> None:
    assert scalar_leakage_controls_rejected()
    expected_vetoes = {
        "same_state": ScalarSuccessorVeto.HEIGHT_LOWERING,
        "wrong_height": ScalarSuccessorVeto.HEIGHT_LOWERING,
        "two_tick_repair": ScalarSuccessorVeto.ONE_TICK_REPAIR,
        "external_leakage": ScalarSuccessorVeto.BOUNDARY_REPAIR_SECTOR,
        "asymmetric_sector": ScalarSuccessorVeto.SCALAR_SECTOR,
    }
    for label, veto in expected_vetoes.items():
        row = scalar_successor_certificate_row(label)
        assert row.verdict == ScalarSuccessorVerdict.FORBID
        assert veto in row.vetoes


def test_third_successor_control_is_rejected_by_z2_pair_rule() -> None:
    row = scalar_successor_certificate_row("third_repair_successor")
    assert scalar_third_successor_control_rejected()
    assert row.verdict == ScalarSuccessorVerdict.FORBID
    assert row.failure_class == ScalarFailureClass.THIRD_SUCCESSOR
    assert ScalarSuccessorVeto.Z2_CONJUGATE_PAIR in row.vetoes


def test_scalar_successor_payload_passes_conditionally() -> None:
    payload = scalar_successor_certificate_payload()
    assert payload.final_verdict == "SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS"
    assert payload.basis_size == 8
    assert payload.row_count == 8
    assert payload.allowed_successors == ("triality_plus", "triality_minus")
    assert payload.certificate_complete
    assert payload.forbidden_rows_have_vetoes
    assert payload.exactly_two_successors
    assert payload.z2_pair_successors
    assert payload.leakage_controls_rejected
    assert payload.third_successor_control_rejected
    assert payload.microscopic_basis_completeness_derived is False
