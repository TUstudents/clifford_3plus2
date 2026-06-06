"""Tests for Session 08B quark color-lift audit."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_color_lift import (
    active_color_embedding,
    color_return_contraction_is_scalar,
    color_scalar_operator,
    commutes_with_su3,
    fixed_color_embedding,
    fixed_color_operator,
    quark_color_lift_payload,
    spectator_color_embedding,
    visible_color_is_scalar,
)


def test_fixed_color_vector_is_rejected_by_su3_covariance() -> None:
    operator = fixed_color_operator()
    embedding = fixed_color_embedding()

    assert operator == sp.diag(1, 0, 0)
    assert not visible_color_is_scalar(operator)
    assert not commutes_with_su3(operator)
    assert not embedding.preserves_visible_color
    assert not embedding.reaches_regular_s3_shell


def test_spectator_color_embedding_preserves_color_but_stays_three_port() -> None:
    embedding = spectator_color_embedding()

    assert color_scalar_operator() == sp.eye(3) / 3
    assert embedding.visible_color_scalar
    assert embedding.commutes_with_su3
    assert embedding.preserves_visible_color
    assert not embedding.reaches_regular_s3_shell
    assert embedding.hidden_shell_breakdown == {"ports": 3}
    assert embedding.down_head is not None
    assert embedding.down_head.matches_baseline
    assert embedding.candidate_head is None


def test_active_color_embedding_preserves_visible_color_and_reaches_regular_s3() -> None:
    embedding = active_color_embedding()

    assert embedding.visible_color_scalar
    assert embedding.commutes_with_su3
    assert embedding.preserves_visible_color
    assert embedding.reaches_regular_s3_shell
    assert embedding.hidden_shell_breakdown == {
        "even_direct": 1,
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "total": 6,
    }
    assert embedding.down_head is not None
    assert embedding.down_head.matches_baseline
    assert embedding.candidate_head is not None
    assert embedding.candidate_head.matches_candidate
    assert not embedding.candidate_head.forced_by_symmetry


def test_su3_return_contraction_is_color_scalar() -> None:
    assert color_return_contraction_is_scalar()


def test_quark_color_lift_payload_reports_conditional_pass_not_source_freeze() -> None:
    payload = quark_color_lift_payload()

    assert payload.final_verdict == "QUARK_COLOR_LIFT_NO_SELECTION_AUDIT"
    assert payload.height_door_prerequisite_pass
    assert payload.quark_shell_prerequisite_pass
    assert payload.fixed_color_rejected
    assert payload.spectator_preserves_color
    assert payload.spectator_stays_three_port
    assert payload.active_preserves_visible_color
    assert payload.active_regular_s3_baseline
    assert payload.active_rank_five_candidate_available
    assert not payload.active_rank_five_candidate_forced
    assert payload.color_return_contraction_scalar
    assert not payload.gauge_alone_selects_active
    assert not payload.source_freeze_ready
