"""Tests for the R9 QCA-to-S3 scalar boundary reduction gate."""

from clifford_3plus2_d5.radial_response.qca_s3_reduction import (
    QCABoundaryFailureClass,
    QCABoundaryReductionVerdict,
    QCABoundaryReductionVeto,
    all_tetrahedral_exit_automorphisms,
    identity_control_rejected,
    induced_residual_image_is_full_s3,
    induced_residual_s3_image,
    non_automorphism_controls_rejected,
    preserves_selected_exit,
    qca_boundary_reduction_rows,
    qca_non_automorphism_control_rows,
    qca_s3_reduction_payload,
    qca_s3_reduction_pass,
    qca_scalar_boundary_allowed_elements,
    qca_scalar_boundary_allowed_labels,
    selected_exit_moving_controls_rejected,
    selected_exit_stabilizer_count,
    transposition_controls_rejected,
)
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    TRIALITY_MINUS,
    TRIALITY_PLUS,
    allowed_scalar_s3_successor_elements,
    allowed_scalar_s3_successor_labels,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import s3_elements


def test_selected_exit_stabilizer_induces_full_residual_s3() -> None:
    assert len(all_tetrahedral_exit_automorphisms()) == 24
    assert selected_exit_stabilizer_count() == 6
    assert set(induced_residual_s3_image()) == set(s3_elements())
    assert induced_residual_image_is_full_s3()


def test_selected_exit_moving_automorphisms_are_rejected() -> None:
    moving_rows = [
        row
        for row in qca_boundary_reduction_rows()
        if not preserves_selected_exit(row.candidate)
    ]
    assert len(moving_rows) == 18
    assert selected_exit_moving_controls_rejected()
    assert all(
        row.verdict == QCABoundaryReductionVerdict.FORBID
        and row.failure_class == QCABoundaryFailureClass.VACUUM_FRAME_BREAKING
        and QCABoundaryReductionVeto.VACUUM_FRAME_PRESERVING in row.vetoes
        for row in moving_rows
    )


def test_identity_and_transposition_controls_are_rejected() -> None:
    assert identity_control_rejected()
    assert transposition_controls_rejected()
    same_state_rows = [
        row
        for row in qca_boundary_reduction_rows()
        if row.failure_class == QCABoundaryFailureClass.SAME_STATE
    ]
    z2_rows = [
        row
        for row in qca_boundary_reduction_rows()
        if row.failure_class == QCABoundaryFailureClass.HERMITIAN_Z2
    ]
    assert len(same_state_rows) == 1
    assert len(z2_rows) == 3


def test_non_automorphism_controls_are_rejected() -> None:
    assert non_automorphism_controls_rejected()
    for row in qca_non_automorphism_control_rows():
        assert row.verdict == QCABoundaryReductionVerdict.FORBID
        assert row.failure_class == QCABoundaryFailureClass.NON_AUTOMORPHISM
        assert QCABoundaryReductionVeto.TETRAHEDRAL_AUTOMORPHISM in row.vetoes


def test_allowed_qca_scalar_successors_match_r8_triality_pair() -> None:
    assert qca_scalar_boundary_allowed_labels() == allowed_scalar_s3_successor_labels()
    assert qca_scalar_boundary_allowed_elements() == allowed_scalar_s3_successor_elements()
    assert qca_scalar_boundary_allowed_labels() == ("triality_plus", "triality_minus")
    assert qca_scalar_boundary_allowed_elements() == (TRIALITY_PLUS, TRIALITY_MINUS)


def test_qca_s3_reduction_payload_passes_conditionally() -> None:
    payload = qca_s3_reduction_payload()
    assert qca_s3_reduction_pass()
    assert payload.final_verdict == "QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS"
    assert payload.tetrahedral_automorphism_count == 24
    assert payload.selected_stabilizer_count == 6
    assert payload.selected_moving_rejected_count == 18
    assert payload.induced_s3_count == 6
    assert payload.induced_s3_is_full
    assert payload.identity_control_rejected
    assert payload.transposition_controls_rejected
    assert payload.non_automorphism_controls_rejected
    assert payload.allowed_successors == ("triality_plus", "triality_minus")
    assert payload.r8_successors_match
    assert payload.microscopic_automorphism_premise_derived is False
