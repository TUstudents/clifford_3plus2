"""Tests for the Session 04 charged-lepton CMV finite head."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.charged_lepton_cmv import (
    CHARGED_LEPTON_SOURCE_LABEL,
    alpha_modulus_squared,
    b_leakage_control_rejected,
    charged_lepton_cmv_head_payload,
    charged_lepton_depth_amplitude,
    charged_lepton_phase,
    charged_lepton_phase_angle,
    charged_lepton_rotation_sine,
    charged_lepton_rotation_sine_for_depth,
    charged_lepton_rotation_sine_squared,
    charged_lepton_source_components,
    charged_lepton_verblunsky_alpha,
    charged_lepton_verblunsky_sequence,
    cmv_givens_head,
    frozen_charged_lepton_source,
    matrix_is_unitary,
)
from clifford_3plus2_d5.universal_bath.opuc import is_free_verblunsky_tail
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import SourceStatus
from clifford_3plus2_d5.universal_bath.tail import silver_epsilon


def test_frozen_charged_lepton_source_is_cmv_e1_depth_two() -> None:
    source = frozen_charged_lepton_source()

    assert source.label == CHARGED_LEPTON_SOURCE_LABEL
    assert source.status == SourceStatus.FROZEN
    assert source.reduction == ReductionKind.CMV_OPUC
    assert source.normal_depth == 2


def test_charged_lepton_source_components_match_e1_decomposition() -> None:
    components = charged_lepton_source_components()

    assert sp.simplify(components["a"] - sp.sqrt(sp.Rational(2, 3))) == 0
    assert sp.simplify(components["u"] - 1 / sp.sqrt(3)) == 0
    assert components["b"] == 0


def test_two_step_leakage_derives_rotation_sine() -> None:
    expected_sine = sp.sqrt(sp.Rational(3, 2)) * silver_epsilon() ** 2
    expected_sine_squared = sp.Rational(3, 2) * silver_epsilon() ** 4

    assert sp.simplify(charged_lepton_depth_amplitude(2) - silver_epsilon() ** 2) == 0
    assert sp.simplify(charged_lepton_rotation_sine() - expected_sine) == 0
    assert sp.simplify(charged_lepton_rotation_sine_squared() - expected_sine_squared) == 0


def test_wrong_depth_controls_do_not_reproduce_depth_two_sine() -> None:
    expected = charged_lepton_rotation_sine()

    assert sp.simplify(charged_lepton_rotation_sine_for_depth(1) - expected) != 0
    assert sp.simplify(charged_lepton_rotation_sine_for_depth(3) - expected) != 0


def test_phase_word_is_exact_boundary_holonomy_phase() -> None:
    expected_phase = sp.cos(-sp.pi * sp.Rational(5, 12)) + sp.I * sp.sin(
        -sp.pi * sp.Rational(5, 12)
    )

    assert charged_lepton_phase_angle() == -sp.Rational(5, 12)
    assert sp.simplify(charged_lepton_phase() - expected_phase) == 0


def test_verblunsky_alpha_has_expected_modulus_and_free_tail() -> None:
    alpha = charged_lepton_verblunsky_alpha()
    coefficients = charged_lepton_verblunsky_sequence()

    assert sp.simplify(alpha_modulus_squared(alpha) - charged_lepton_rotation_sine_squared()) == 0
    assert coefficients[0] == alpha
    assert is_free_verblunsky_tail(coefficients[1:])


def test_cmv_givens_head_is_exactly_unitary() -> None:
    head = cmv_givens_head(charged_lepton_verblunsky_alpha())

    assert matrix_is_unitary(head)


def test_b_leakage_control_is_rejected() -> None:
    assert b_leakage_control_rejected()


def test_charged_lepton_cmv_head_payload_reports_pass_and_parks_pmns() -> None:
    payload = charged_lepton_cmv_head_payload()

    assert payload.final_verdict == "CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS"
    assert payload.source_dictionary_pass
    assert payload.holonomy_prerequisite_pass
    assert payload.source_label == CHARGED_LEPTON_SOURCE_LABEL
    assert payload.source_reduction == ReductionKind.CMV_OPUC
    assert payload.source_depth == 2
    assert payload.alpha_inside_unit_disk
    assert payload.cmv_head_unitary
    assert payload.free_tail_after_head
    assert payload.depth_one_control_rejected
    assert payload.depth_three_control_rejected
    assert payload.b_leakage_control_rejected
    assert payload.holonomy_controls_rejected
    assert payload.pmns_parked
