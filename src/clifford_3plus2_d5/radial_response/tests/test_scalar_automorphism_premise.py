"""Tests for the R10 scalar automorphism premise gate."""

from clifford_3plus2_d5.radial_response.scalar_automorphism_premise import (
    ScalarAutomorphismFailureClass,
    ScalarAutomorphismPremiseVerdict,
    ScalarAutomorphismPremiseVeto,
    lift_residual_action_to_selected_exit_map,
    scalar_automorphism_allowed_maps,
    scalar_automorphism_allowed_successors,
    scalar_automorphism_certificate_is_complete,
    scalar_automorphism_control_rejected,
    scalar_automorphism_forbidden_rows_have_vetoes,
    scalar_automorphism_premise_certificate,
    scalar_automorphism_premise_pass,
    scalar_automorphism_premise_payload,
    scalar_automorphism_premise_row,
)
from clifford_3plus2_d5.radial_response.qca_s3_reduction import (
    residual_action_for_candidate,
)
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    TRIALITY_MINUS,
    TRIALITY_PLUS,
)


def test_residual_actions_lift_to_selected_exit_preserving_maps() -> None:
    plus_map = lift_residual_action_to_selected_exit_map(TRIALITY_PLUS)
    minus_map = lift_residual_action_to_selected_exit_map(TRIALITY_MINUS)
    assert plus_map[0] == 0
    assert minus_map[0] == 0
    assert residual_action_for_candidate(plus_map) == TRIALITY_PLUS
    assert residual_action_for_candidate(minus_map) == TRIALITY_MINUS


def test_scalar_automorphism_certificate_allows_only_triality_maps() -> None:
    assert scalar_automorphism_certificate_is_complete()
    assert scalar_automorphism_forbidden_rows_have_vetoes()
    assert scalar_automorphism_allowed_maps() == (
        "triality_plus_automorphism",
        "triality_minus_automorphism",
    )
    assert scalar_automorphism_allowed_successors() == (
        "triality_plus",
        "triality_minus",
    )
    rows = scalar_automorphism_premise_certificate()
    allowed = [row for row in rows if row.verdict == ScalarAutomorphismPremiseVerdict.ALLOW]
    assert len(rows) == 9
    assert len(allowed) == 2


def test_scalar_automorphism_controls_are_rejected_with_exact_vetoes() -> None:
    controls = (
        (
            "selected_exit_moving_automorphism",
            ScalarAutomorphismFailureClass.VACUUM_FRAME_BREAKING,
            ScalarAutomorphismPremiseVeto.VACUUM_FRAME_PRESERVING,
        ),
        (
            "generic_linear_exit_mixture",
            ScalarAutomorphismFailureClass.NON_DETERMINISTIC_EXIT_MAP,
            ScalarAutomorphismPremiseVeto.DETERMINISTIC_EXIT_MAP,
        ),
        (
            "non_automorphism_exit_map",
            ScalarAutomorphismFailureClass.NON_AUTOMORPHISM,
            ScalarAutomorphismPremiseVeto.TETRAHEDRAL_AUTOMORPHISM,
        ),
        (
            "two_tick_triality_map",
            ScalarAutomorphismFailureClass.NONLOCAL,
            ScalarAutomorphismPremiseVeto.ONE_TICK_LOCALITY,
        ),
        (
            "spin_coupled_triality_map",
            ScalarAutomorphismFailureClass.NON_SCALAR_LOCAL,
            ScalarAutomorphismPremiseVeto.SCALAR_LOCALITY,
        ),
        (
            "identity_same_state_map",
            ScalarAutomorphismFailureClass.SAME_STATE,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        ),
        (
            "hermitian_z2_transposition_map",
            ScalarAutomorphismFailureClass.HERMITIAN_Z2,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        ),
    )
    for label, failure_class, veto in controls:
        assert scalar_automorphism_control_rejected(label, failure_class, veto)


def test_nonlocal_and_nonscalar_triality_controls_do_not_count_as_allowed() -> None:
    nonlocal_row = scalar_automorphism_premise_row("two_tick_triality_map")
    nonscalar_row = scalar_automorphism_premise_row("spin_coupled_triality_map")
    assert nonlocal_row.scalar_label == "triality_plus"
    assert nonscalar_row.scalar_label == "triality_plus"
    assert nonlocal_row.verdict == ScalarAutomorphismPremiseVerdict.FORBID
    assert nonscalar_row.verdict == ScalarAutomorphismPremiseVerdict.FORBID


def test_scalar_automorphism_premise_payload_passes_conditionally() -> None:
    payload = scalar_automorphism_premise_payload()
    assert scalar_automorphism_premise_pass()
    assert payload.final_verdict == "SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS"
    assert payload.candidate_count == 9
    assert payload.allowed_maps == (
        "triality_plus_automorphism",
        "triality_minus_automorphism",
    )
    assert payload.allowed_successors == ("triality_plus", "triality_minus")
    assert payload.certificate_complete
    assert payload.forbidden_rows_have_vetoes
    assert payload.selected_exit_moving_control_rejected
    assert payload.generic_linear_control_rejected
    assert payload.non_automorphism_control_rejected
    assert payload.nonlocal_control_rejected
    assert payload.nonscalar_control_rejected
    assert payload.identity_control_rejected
    assert payload.hermitian_z2_control_rejected
    assert payload.r9_reduction_applies
    assert payload.declared_scalar_local_map_sufficient
    assert payload.full_bb_qca_update_derived is False
