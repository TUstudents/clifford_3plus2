"""Tests for the V10 leptonic boundary-loop holonomy gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.leptonic_boundary_holonomy import (
    PARENT_A3,
    RESIDUAL_A2,
    SCHUR_RETURN,
    BoundaryLoopWord,
    admissible_charged_lepton_words,
    boundary_holonomy_factors,
    boundary_loop_candidates,
    boundary_loop_control_words,
    is_admissible_charged_lepton_word,
    leptonic_boundary_holonomy_audit_payload,
    primitive_charged_lepton_boundary_word,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    leptonic_phase_word_audit_payload,
)
from clifford_3plus2_d5.boundary_response.pmns_conditional import (
    pmns_conditional_audit_payload,
)


def test_holonomy_factor_registry_has_exact_angles() -> None:
    factors = {factor.name: factor for factor in boundary_holonomy_factors()}
    assert factors[SCHUR_RETURN].angle == 1
    assert factors[PARENT_A3].angle == sp.Rational(1, 4)
    assert factors[RESIDUAL_A2].angle == sp.Rational(1, 3)


def test_primitive_boundary_word_has_v8_phase() -> None:
    word = primitive_charged_lepton_boundary_word()
    assert word.factor_names == (SCHUR_RETURN, PARENT_A3, RESIDUAL_A2)
    assert word.raw_angle == sp.Rational(19, 12)
    assert word.principal_angle == -sp.Rational(5, 12)
    assert word.phase == sp.exp(-sp.I * sp.pi * sp.Rational(5, 12))


def test_candidate_catalog_has_unique_admissible_word() -> None:
    candidates = boundary_loop_candidates(max_length=4)
    admissible = admissible_charged_lepton_words(max_length=4)
    assert len(candidates) == 120
    assert len(admissible) == 1
    assert admissible[0] == primitive_charged_lepton_boundary_word()


def test_invalid_candidate_length_is_rejected() -> None:
    try:
        boundary_loop_candidates(max_length=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected max_length validation")


def test_negative_control_words_are_not_admissible() -> None:
    controls = boundary_loop_control_words()
    assert set(controls) == {
        "a3_only",
        "a2_only",
        "a3_a2_without_schur",
        "schur_a3_only",
        "schur_a2_only",
        "reversed_a2_a3",
        "duplicated_full_word",
    }
    assert all(not is_admissible_charged_lepton_word(word) for word in controls.values())


def test_reversed_word_has_same_scalar_angle_but_wrong_orientation() -> None:
    primitive = primitive_charged_lepton_boundary_word()
    reversed_word = BoundaryLoopWord((SCHUR_RETURN, RESIDUAL_A2, PARENT_A3))
    assert reversed_word.raw_angle == primitive.raw_angle
    assert reversed_word.phase == primitive.phase
    assert not is_admissible_charged_lepton_word(reversed_word)


def test_holonomy_payload_reports_derived_pass() -> None:
    payload = leptonic_boundary_holonomy_audit_payload()
    assert payload.final_verdict == "LEPTONIC_PHASE_WORD_DERIVED_PASS"
    assert payload.raw_angle == sp.Rational(19, 12)
    assert payload.principal_angle == -sp.Rational(5, 12)
    assert payload.admissible_count == 1
    assert all(payload.controls_rejected.values())
    assert payload.word_selection_derived
    assert payload.ckm_parked


def test_v8_and_v9_regressions_remain_stable() -> None:
    assert (
        leptonic_phase_word_audit_payload().final_verdict
        == "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
    )
    assert pmns_conditional_audit_payload().final_verdict == "PMNS_CONDITIONAL_ASSEMBLY_PASS"
