"""Tests for Session 19 charged-lepton trace/torsion origin audit."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.charged_lepton_trace_torsion_origin import (
    TORSION_DYNAMICS_PREMISE,
    TRACE_PATH_ORIGIN_PREMISE,
    charged_lepton_trace_torsion_origin_payload,
    cmv_phase_word_is_not_torsion_angle,
    one_trace_path_control_rejected,
    required_trace_path_count_for_equipartition,
    supplied_trace_path_count,
    three_trace_path_control_rejected,
    torsion_weight_inserted_into_theta,
    trace_weight_for_path_count,
    traceless_weight_for_path_count,
    two_trace_paths_force_equipartition,
)


def test_trace_path_count_formula_selects_exactly_two_paths() -> None:
    assert trace_weight_for_path_count(1) == sp.Rational(1, 3)
    assert traceless_weight_for_path_count(1) == sp.Rational(2, 3)
    assert trace_weight_for_path_count(2) == sp.Rational(1, 2)
    assert traceless_weight_for_path_count(2) == sp.Rational(1, 2)
    assert trace_weight_for_path_count(3) == sp.Rational(3, 5)
    assert required_trace_path_count_for_equipartition() == 2
    assert two_trace_paths_force_equipartition()


def test_minimal_graph_supplies_two_trace_paths_but_does_not_derive_them() -> None:
    assert supplied_trace_path_count() == 2
    assert one_trace_path_control_rejected()
    assert three_trace_path_control_rejected()


def test_torsion_weight_is_inserted_as_angle_not_dynamically_mapped() -> None:
    assert torsion_weight_inserted_into_theta()
    assert cmv_phase_word_is_not_torsion_angle()


def test_session_19_payload_reports_origin_audit() -> None:
    payload = charged_lepton_trace_torsion_origin_payload()

    assert payload.final_verdict == "CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT"
    assert payload.boundary_graph_pass
    assert payload.torsion_gate_pass
    assert payload.holonomy_gate_pass
    assert payload.source_occupation_weights == {
        "a": sp.Rational(2, 3),
        "u": sp.Rational(1, 3),
        "b": 0,
    }
    assert payload.required_trace_path_count == 2
    assert payload.supplied_trace_path_count == 2
    assert payload.supplied_trace_paths_match_required_count
    assert payload.two_trace_paths_force_equipartition
    assert payload.one_trace_path_control_rejected
    assert payload.three_trace_path_control_rejected
    assert not payload.trace_path_origin_derived_from_bcc_higgs
    assert sp.simplify(payload.torsion_occupation_weight - sp.Rational(2, 9)) == 0
    assert sp.simplify(payload.torsion_used_as_rotation_angle - sp.Rational(2, 9)) == 0
    assert payload.torsion_weight_inserted_into_theta
    assert not payload.occupation_to_angle_map_derived
    assert payload.cmv_phase_word_pass
    assert payload.cmv_phase_word_is_not_torsion_angle
    assert payload.minimal_graph_accepts_supplied_theta
    assert payload.koide_still_exact_with_supplied_theta
    assert payload.remaining_declared_inputs == (
        TRACE_PATH_ORIGIN_PREMISE,
        TORSION_DYNAMICS_PREMISE,
    )
