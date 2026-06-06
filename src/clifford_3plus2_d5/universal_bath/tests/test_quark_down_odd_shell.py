"""Tests for Session 18 down odd-shell rank-five audit."""

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import BCC, ODD
from clifford_3plus2_d5.universal_bath.quark_down_odd_shell import (
    BOTTOM_ODD_SHELL_READOUT_PREMISE,
    bcc_middle_rank_two,
    bcc_odd_subset,
    odd_shell_is_complement_of_even_direct,
    odd_shell_subset,
    primitive_counts,
    primitive_parity_selects_rank_five_line,
    quark_down_odd_shell_payload,
)


def test_primitive_predicate_counts_are_6_2_5() -> None:
    assert primitive_counts() == {"d": 6, "s": 2, "b": 5}
    assert odd_shell_subset().count == 5
    assert bcc_odd_subset().count == 2
    assert all(
        channel.parity == ODD and channel.sector == BCC
        for channel in bcc_odd_subset().channels
    )


def test_primitive_parity_selects_the_rank_five_line() -> None:
    assert odd_shell_is_complement_of_even_direct()
    assert primitive_parity_selects_rank_five_line()
    assert bcc_middle_rank_two()


def test_quark_down_odd_shell_payload_reports_conditional_pass() -> None:
    payload = quark_down_odd_shell_payload()

    assert payload.final_verdict == "QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS"
    assert payload.active_color_microcanonical_pass
    assert payload.down_subset_pass
    assert payload.s3_projector_pass
    assert payload.s3_rank_five_ambiguous
    assert payload.primitive_counts == {"d": 6, "s": 2, "b": 5}
    assert payload.candidate_counts == {"d": 6, "s": 2, "b": 5}
    assert payload.candidate_clebsch_vector == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert payload.odd_shell_rank_five
    assert payload.odd_shell_is_complement_of_even_direct
    assert payload.bcc_middle_rank_two
    assert payload.color_only_middle_control_rejected
    assert payload.compressed_parity_control_rejected_for_middle
    assert payload.primitive_parity_selects_rank_five_line
    assert payload.remaining_readout_premise == BOTTOM_ODD_SHELL_READOUT_PREMISE
    assert not payload.source_freeze_ready
