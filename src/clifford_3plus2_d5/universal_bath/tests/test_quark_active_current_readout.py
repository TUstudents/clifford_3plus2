"""Tests for Session 21 quark active-current readout ansatz."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_active_current_readout import (
    CURRENT_SOURCE_PREMISE,
    DOWN_CURRENT_MEASURE_PREMISE,
    DOWN_IDENTITY_VETO_PREMISE,
    active_current_is_b,
    active_current_line_residual_uab,
    active_current_unique_non_scalar_line,
    coherent_up_profile_from_b,
    down_depths_from_first_return,
    down_identity_veto_selects_odd_shell_if_assumed,
    first_return_orders_from_b,
    quark_active_current_readout_payload,
    up_depths_from_first_return,
)


def test_active_current_line_is_unique_b_direction() -> None:
    assert active_current_line_residual_uab() == sp.Matrix([0, 0, 1])
    assert active_current_is_b()
    assert active_current_unique_non_scalar_line()


def test_first_passage_orders_from_b_give_quark_depths() -> None:
    assert first_return_orders_from_b() == {"u": 2, "a": 1, "b": 0}
    assert up_depths_from_first_return() == {"u": 6, "a": 3, "b": 0}
    assert down_depths_from_first_return() == {"u": 6, "a": 4, "b": 2}


def test_coherent_up_profile_from_b_is_taylor_not_geometric() -> None:
    assert coherent_up_profile_from_b() == (
        sp.Rational(1, 4),
        1 / sp.sqrt(2),
        sp.Integer(1),
    )


def test_down_identity_veto_candidate_is_available_but_not_derived() -> None:
    assert down_identity_veto_selects_odd_shell_if_assumed()


def test_session_21_payload_reports_active_current_conditional_pass() -> None:
    payload = quark_active_current_readout_payload()

    assert payload.final_verdict == "QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS"
    assert payload.active_plane_pass
    assert payload.height_bridge_pass
    assert payload.up_head_pass
    assert payload.down_odd_shell_pass
    assert payload.active_current_is_b
    assert payload.active_current_unique_non_scalar_line
    assert payload.first_return_orders_light_to_heavy == (2, 1, 0)
    assert payload.up_radial_depths == {"u": 6, "a": 3, "b": 0}
    assert payload.down_radial_depths == {"u": 6, "a": 4, "b": 2}
    assert payload.up_profile_matches_conditional_head
    assert payload.up_geometric_control_rejected
    assert payload.down_readout_is_covariance
    assert payload.down_baseline_counts == {"d": 6, "s": 2, "b": 4}
    assert payload.down_baseline_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(2, 3)),
    )
    assert payload.down_odd_shell_counts == {"d": 6, "s": 2, "b": 5}
    assert payload.down_odd_shell_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert payload.down_identity_veto_selects_odd_shell_if_assumed
    assert not payload.down_identity_veto_microscopically_derived
    assert payload.source_freeze_candidate
    assert not payload.source_freeze_ready
    assert payload.remaining_physical_inputs == (
        CURRENT_SOURCE_PREMISE,
        DOWN_CURRENT_MEASURE_PREMISE,
        DOWN_IDENTITY_VETO_PREMISE,
    )
