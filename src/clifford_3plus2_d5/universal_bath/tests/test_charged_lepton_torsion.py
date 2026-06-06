"""Tests for the Session 05 charged-lepton ``2/9`` torsion gate."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.charged_lepton_torsion import (
    charged_lepton_torsion_payload,
    charged_lepton_torsion_weight,
    coherent_transition_amplitude,
    equal_weight_transition_control,
    occupation_weights_normalized,
    one_port_transition_controls_rejected,
    source_occupation_weights,
    torsion_transition_weight,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind


def test_source_occupation_weights_are_exact_e1_probabilities() -> None:
    weights = source_occupation_weights()

    assert weights == {"a": sp.Rational(2, 3), "u": sp.Rational(1, 3), "b": 0}
    assert occupation_weights_normalized(weights)


def test_torsion_weight_is_two_over_nine_from_incoherent_transition() -> None:
    assert sp.simplify(charged_lepton_torsion_weight() - sp.Rational(2, 9)) == 0
    assert sp.simplify(torsion_transition_weight("u", "a") - sp.Rational(2, 9)) == 0


def test_coherent_amplitude_is_not_the_torsion_weight() -> None:
    coherent = coherent_transition_amplitude()

    assert sp.simplify(coherent - sp.sqrt(2) / 3) == 0
    assert sp.simplify(coherent - sp.Rational(2, 9)) != 0


def test_equal_and_one_port_controls_are_rejected() -> None:
    assert sp.simplify(equal_weight_transition_control() - sp.Rational(1, 4)) == 0
    assert sp.simplify(equal_weight_transition_control() - sp.Rational(2, 9)) != 0
    assert one_port_transition_controls_rejected()


def test_charged_lepton_torsion_payload_reports_pass_but_not_phase_derivation() -> None:
    payload = charged_lepton_torsion_payload()

    assert payload.final_verdict == "CHARGED_LEPTON_TORSION_2_OVER_9_PASS"
    assert payload.source_dictionary_pass
    assert payload.source_reduction == ReductionKind.CMV_OPUC
    assert payload.occupation_weights_normalized
    assert payload.b_occupation_zero
    assert sp.simplify(payload.torsion_transition_weight - sp.Rational(2, 9)) == 0
    assert payload.coherent_amplitude_rejected
    assert payload.equal_weight_control_rejected
    assert payload.one_port_controls_rejected
    assert payload.cmv_phase_not_rederived
