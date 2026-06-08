"""Tests for Session 02 D3 clock source identities."""

import sympy as sp

from clifford_3plus2_d5.threeclocks.clock_spine import matrix_equal
from clifford_3plus2_d5.threeclocks.d3_clock import (
    d3_clock_source_payload,
    d3_relations_hold,
    down_profile_if_contact_return_allowed,
    down_profile_if_rank_five_shell_supplied,
    first_return_orders_from_b_target,
    literal_port_cut_equals_repair_flag,
    literal_port_cut_in_residual_frame,
    nilpotent_repair_flag_target,
    radial_identity_holds,
    repair_flag_is_nilpotent,
    selected_tooth_components,
    tangent_identity_holds,
    three_port_laplacian_spectrum,
    up_profile_if_repair_flag_supplied,
)


def test_d3_relations_hold_for_selected_clock() -> None:
    assert d3_relations_hold()


def test_selected_tooth_decomposes_into_trace_and_radial_only() -> None:
    assert selected_tooth_components() == (1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)), 0)


def test_clock_first_and_second_differences_select_b_and_a() -> None:
    assert tangent_identity_holds()
    assert radial_identity_holds()


def test_three_port_laplacian_is_degenerate_control_not_down_shell() -> None:
    assert three_port_laplacian_spectrum() == (0, 3, 3)


def test_literal_port_cut_does_not_smuggle_in_repair_flag() -> None:
    assert not literal_port_cut_equals_repair_flag()
    assert not matrix_equal(
        literal_port_cut_in_residual_frame(),
        nilpotent_repair_flag_target(),
    )


def test_repair_flag_and_profiles_are_conditional_targets() -> None:
    assert repair_flag_is_nilpotent()
    assert first_return_orders_from_b_target() == (2, 1, 0)
    assert up_profile_if_repair_flag_supplied() == (sp.Rational(1, 4), 1 / sp.sqrt(2), 1)
    assert down_profile_if_rank_five_shell_supplied() == (
        1,
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert down_profile_if_contact_return_allowed() == (
        1,
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(2, 3)),
    )


def test_session_02_payload_keeps_open_gates_open() -> None:
    payload = d3_clock_source_payload()

    assert payload.final_verdict == "D3_CLOCK_SOURCE_IDENTITY_PASS"
    assert payload.d3_relations_pass
    assert payload.selected_tooth_has_no_tangent
    assert payload.tangent_current_is_b
    assert payload.radial_second_difference_is_a
    assert payload.quark_source_b_derived
    assert payload.three_port_laplacian_is_degenerate_control
    assert not payload.literal_cut_equals_repair_flag
    assert payload.repair_flag_nilpotent_target_pass
    assert payload.first_return_orders_target == (2, 1, 0)
    assert "derive_representation_basis_repair_flag_N_from_clock_defect" in (
        payload.open_theorem_targets
    )
    assert "compute_CKM_from_two_sided_clock_kernels" in payload.open_theorem_targets
