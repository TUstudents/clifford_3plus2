"""Tests for Session 22 quark current-parity selector."""

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    residual_projectors,
)
from clifford_3plus2_d5.universal_bath.quark_current_parity_selector import (
    ODD_CURRENT_PHYSICAL_PREMISE,
    a_rejected_as_radial_even,
    active_plane_alone_insufficient,
    current_parity_selects_b,
    even_projector_is_u_plus_a,
    line_has_parity,
    odd_projector_is_b,
    quark_current_parity_selector_payload,
    selected_pair_current_is_b,
    selected_pair_current_line_residual_uab,
    selected_pair_current_line_standard,
    selected_s2_action_residual_uab,
    selected_s2_even_projector,
    selected_s2_odd_projector,
    selected_s2_parity_by_line,
    selected_scalar_has_no_current_component,
    u_rejected_as_scalar_even,
)


def test_selected_s2_action_is_even_even_odd_in_residual_basis() -> None:
    assert selected_s2_action_residual_uab() == sp.diag(1, 1, -1)
    assert selected_s2_parity_by_line() == {"u": 1, "a": 1, "b": -1}


def test_selected_s2_projectors_are_even_ua_and_odd_b() -> None:
    projectors = residual_projectors()

    assert even_projector_is_u_plus_a()
    assert odd_projector_is_b()
    assert selected_s2_even_projector() == projectors["u"] + projectors["a"]
    assert selected_s2_odd_projector() == projectors["b"]


def test_selected_pair_current_is_the_odd_b_line() -> None:
    current = selected_pair_current_line_standard()

    assert selected_pair_current_line_residual_uab() == sp.Matrix([0, 0, 1])
    assert selected_pair_current_is_b()
    assert line_has_parity(current, -1)
    assert not line_has_parity(current, 1)


def test_controls_reject_scalar_active_line_and_radial_even_line() -> None:
    assert selected_scalar_has_no_current_component()
    assert active_plane_alone_insufficient()
    assert u_rejected_as_scalar_even()
    assert a_rejected_as_radial_even()
    assert current_parity_selects_b()


def test_session_22_payload_reports_current_parity_selector_pass() -> None:
    payload = quark_current_parity_selector_payload()

    assert payload.final_verdict == "QUARK_CURRENT_PARITY_SELECTOR_PASS"
    assert payload.selected_s2_residual_action_uab == sp.diag(1, 1, -1)
    assert payload.parity_by_line == {"u": 1, "a": 1, "b": -1}
    assert payload.even_projector_is_u_plus_a
    assert payload.odd_projector_is_b
    assert payload.active_plane_pass
    assert payload.current_line_residual_uab == sp.Matrix([0, 0, 1])
    assert payload.current_line_is_b
    assert payload.current_line_is_selected_pair_current
    assert payload.current_line_is_odd
    assert payload.selected_scalar_has_no_current_component
    assert payload.active_plane_alone_insufficient
    assert payload.u_rejected_as_scalar_even
    assert payload.a_rejected_as_radial_even
    assert payload.current_parity_selects_b
    assert payload.session21_source_premise_reduced
    assert payload.remaining_physical_premise == ODD_CURRENT_PHYSICAL_PREMISE
