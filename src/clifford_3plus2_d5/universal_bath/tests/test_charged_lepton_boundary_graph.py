"""Tests for Session 14 charged-lepton minimal boundary graph."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.universal_bath.charged_lepton_boundary_graph import (
    charged_lepton_base_angle,
    charged_lepton_boundary_angle,
    charged_lepton_boundary_graph_payload,
    charged_lepton_torsion_angle,
    koide_parameter,
    left_coupling_matrix,
    normalized_square_root_vector_uab,
    one_sided_hermitian_control_rejected,
    one_trace_path_control_rejected,
    pole_residue,
    residue_on_selected_port,
    right_coupling_matrix,
    selected_port_uab,
    target_residue,
    trace_weight,
    traceless_weight,
)


def test_charged_lepton_angle_uses_base_and_torsion_inputs() -> None:
    """The graph angle is assembled from holonomy base plus 2/9 torsion."""

    assert sp.simplify(charged_lepton_base_angle() + 2 * sp.pi / 3) == 0
    assert sp.simplify(charged_lepton_torsion_angle() - sp.Rational(2, 9)) == 0
    assert sp.simplify(charged_lepton_boundary_angle() + 2 * sp.pi / 3 + sp.Rational(2, 9)) == 0


def test_two_sided_residue_matches_target_graph() -> None:
    """The four-state pole graph realizes sqrt(2) P_u + R_theta P_perp."""

    theta = charged_lepton_boundary_angle()

    assert left_coupling_matrix().shape == (4, 3)
    assert right_coupling_matrix(theta).shape == (4, 3)
    assert pole_residue(theta) == target_residue(theta)


def test_selected_port_action_gives_equipartition() -> None:
    """Acting on e1 gives equal trace and traceless weights."""

    theta = charged_lepton_boundary_angle()
    action = residue_on_selected_port(theta)
    normalized = normalized_square_root_vector_uab(theta)

    assert selected_port_uab() == sp.Matrix([1 / sp.sqrt(3), sp.sqrt(sp.Rational(2, 3)), 0])
    assert sp.simplify(action[0] - sp.sqrt(sp.Rational(2, 3))) == 0
    assert sp.trigsimp(sp.simplify(action[1] ** 2 + action[2] ** 2 - sp.Rational(2, 3))) == 0
    assert sp.simplify(trace_weight(theta) - sp.Rational(1, 2)) == 0
    assert sp.simplify(traceless_weight(theta) - sp.Rational(1, 2)) == 0
    assert sp.trigsimp(sp.simplify((normalized.T * normalized)[0] - 1)) == 0


def test_koide_and_controls() -> None:
    """Koide is exact; one-trace and one-sided Hermitian controls fail."""

    theta = charged_lepton_boundary_angle()

    assert sp.simplify(koide_parameter(theta) - sp.Rational(2, 3)) == 0
    assert one_trace_path_control_rejected()
    assert one_sided_hermitian_control_rejected(theta)


def test_session_14_payload() -> None:
    """Session 14 is a minimal graph realization, not a torsion derivation."""

    payload = charged_lepton_boundary_graph_payload()

    assert payload.final_verdict == "CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS"
    assert payload.source_dictionary_pass
    assert payload.cmv_head_pass
    assert payload.torsion_gate_pass
    assert payload.holonomy_gate_pass
    assert payload.residue_matches_target
    assert payload.trace_traceless_equipartition
    assert payload.koide_parameter_is_two_thirds
    assert payload.one_trace_path_control_rejected
    assert payload.one_sided_hermitian_control_rejected
    assert payload.torsion_angle_not_derived_by_graph
    assert "active_cmv_torsion" in payload.remaining_declared_inputs[1]
