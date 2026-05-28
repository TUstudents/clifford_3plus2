"""Tests for the V11 quark primitive boundary-shell gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    leptonic_boundary_holonomy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.pmns_conditional import (
    pmns_conditional_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    BCC,
    COLOR,
    EVEN,
    ODD,
    clifford_relations_hold,
    commuting_odd_generators_control,
    missing_color_channel_control,
    nonflat_quark_boundary_phase_angle,
    quark_boundary_coin,
    quark_boundary_phase_angle,
    quark_boundary_phase_factor,
    quark_boundary_shell_audit_payload,
    quark_gamma_sum,
    quark_odd_clifford_generators,
    quark_primitive_channels,
    quark_shell_dimension_breakdown,
)


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_quark_shell_has_one_even_and_five_odd_channels() -> None:
    channels = quark_primitive_channels()
    assert len(channels) == 6
    assert sum(1 for channel in channels if channel.parity == EVEN) == 1
    assert sum(1 for channel in channels if channel.parity == ODD) == 5


def test_quark_shell_splits_odd_channels_as_bcc_plus_color() -> None:
    channels = quark_primitive_channels()
    assert sum(1 for channel in channels if channel.parity == ODD and channel.sector == BCC) == 2
    assert sum(1 for channel in channels if channel.parity == ODD and channel.sector == COLOR) == 3
    assert quark_shell_dimension_breakdown() == {
        "even_direct": 1,
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "total": 6,
    }


def test_quark_odd_generators_satisfy_cl5_relations() -> None:
    generators = quark_odd_clifford_generators()
    assert len(generators) == 5
    assert clifford_relations_hold(generators)


def test_quark_gamma_sum_squares_to_five_identity() -> None:
    gamma_sum = quark_gamma_sum()
    _assert_matrix_equal(gamma_sum * gamma_sum, 5 * sp.eye(4))


def test_quark_boundary_coin_is_unitary() -> None:
    coin = quark_boundary_coin()
    _assert_matrix_equal(coin.conjugate().T * coin, sp.eye(4))


def test_quark_boundary_phase_matches_flat_coin_prediction() -> None:
    assert quark_boundary_phase_factor() == (1 + sp.I * sp.sqrt(5)) / sp.sqrt(6)
    assert quark_boundary_phase_angle() == sp.atan(sp.sqrt(5))


def test_quark_shell_negative_controls_are_rejected() -> None:
    missing_breakdown = quark_shell_dimension_breakdown(missing_color_channel_control())
    assert missing_breakdown["color_odd"] == 2
    assert missing_breakdown["odd_total"] == 4
    assert sp.simplify(nonflat_quark_boundary_phase_angle(2) - quark_boundary_phase_angle()) != 0

    commuting_sum = quark_gamma_sum(commuting_odd_generators_control())
    residual = commuting_sum * commuting_sum - 5 * sp.eye(4)
    assert any(sp.simplify(entry) != 0 for entry in residual)


def test_quark_boundary_shell_payload_reports_q1_pass() -> None:
    payload = quark_boundary_shell_audit_payload()
    assert payload.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
    assert payload.shell_breakdown["total"] == 6
    assert payload.gamma_sum_square_matches
    assert payload.coin_unitary
    assert payload.phase_angle == sp.atan(sp.sqrt(5))
    assert payload.missing_color_control_rejected
    assert payload.nonflat_control_rejected
    assert payload.commuting_control_rejected
    assert payload.ckm_parked


def test_v9_and_v10_regressions_remain_stable() -> None:
    assert pmns_conditional_audit_payload().final_verdict == "PMNS_CONDITIONAL_ASSEMBLY_PASS"
    assert (
        leptonic_boundary_holonomy_audit_payload().final_verdict
        == "LEPTONIC_PHASE_WORD_DERIVED_PASS"
    )
