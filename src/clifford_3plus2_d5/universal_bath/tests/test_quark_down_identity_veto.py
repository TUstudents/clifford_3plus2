"""Tests for Session 23 down identity-return veto."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_down_identity_veto import (
    RETARDED_DOWN_CURRENT_PREMISE,
    allowed_return_breakdown,
    all_allowed_returns_are_odd,
    baseline_control_rejected_by_retarded_predicate,
    direct_identity_decisions,
    direct_identity_is_unique,
    direct_identity_vetoed,
    primitive_return_decisions,
    quark_down_identity_veto_payload,
    rank_five_selected_inside_retarded_model,
    retarded_allowed_decisions,
    retarded_down_counts,
    retarded_down_profile,
    retarded_rejected_decisions,
)


def test_primitive_decisions_have_one_identity_and_five_retarded_returns() -> None:
    decisions = primitive_return_decisions()
    identities = direct_identity_decisions()
    allowed = retarded_allowed_decisions()
    rejected = retarded_rejected_decisions()

    assert len(decisions) == 6
    assert len(identities) == 1
    assert identities[0].name == "direct_even_return"
    assert identities[0].return_order == 0
    assert not identities[0].hidden_excursion
    assert direct_identity_is_unique()
    assert direct_identity_vetoed()
    assert tuple(decision.name for decision in rejected) == ("direct_even_return",)
    assert len(allowed) == 5
    assert all(decision.return_order == 1 for decision in allowed)
    assert all_allowed_returns_are_odd()


def test_retarded_allowed_returns_are_2_bcc_plus_3_color() -> None:
    assert allowed_return_breakdown() == {
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "direct_even": 0,
    }


def test_retarded_identity_veto_selects_rank_five_bottom_profile() -> None:
    assert retarded_down_counts() == {"d": 6, "s": 2, "b": 5}
    assert retarded_down_profile() == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert baseline_control_rejected_by_retarded_predicate()
    assert rank_five_selected_inside_retarded_model()


def test_session_23_payload_reports_conditional_rank_five_pass() -> None:
    payload = quark_down_identity_veto_payload()

    assert payload.final_verdict == "DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS"
    assert payload.active_current_readout_pass
    assert payload.current_parity_selector_pass
    assert payload.odd_shell_pass
    assert payload.direct_identity_names == ("direct_even_return",)
    assert payload.retarded_rejected_names == ("direct_even_return",)
    assert payload.direct_identity_is_unique
    assert payload.direct_identity_vetoed
    assert payload.all_allowed_returns_are_odd
    assert payload.allowed_return_count == 5
    assert payload.allowed_return_breakdown == {
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "direct_even": 0,
    }
    assert payload.retarded_counts == {"d": 6, "s": 2, "b": 5}
    assert payload.retarded_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )
    assert payload.baseline_counts == {"d": 6, "s": 2, "b": 4}
    assert payload.baseline_profile == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(2, 3)),
    )
    assert payload.baseline_control_rejected_by_retarded_predicate
    assert payload.rank_five_selected_inside_retarded_model
    assert payload.down_identity_veto_premise_reduced
    assert not payload.microscopic_bare_bcc_derivation
    assert RETARDED_DOWN_CURRENT_PREMISE in payload.remaining_physical_premises
