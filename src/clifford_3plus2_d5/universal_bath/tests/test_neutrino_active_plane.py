"""Tests for Session 11 selected active-plane incidence."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_selected_s2_invariant,
    projector,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
    active_projector_from_channels,
    active_projector_from_incidence,
    detraced_selected_radial_line,
    opposite_edge_line_from_incidence,
    radial_projector_from_incidence,
    raw_selected_port_active_control,
    selected_boundary_port,
    selected_port_components_aub,
    selected_s2_symmetry_alone_not_sufficient,
    selected_s2_symmetry_only_counterexample,
)
from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import (
    active_neutrino_projector,
    matrix_equal,
)


def test_selected_port_detraces_to_radial_a() -> None:
    """The selected residual port fixes a by subtracting the collective trace."""

    components = selected_port_components_aub()
    radial = detraced_selected_radial_line()
    projectors = residual_projectors()

    assert sp.simplify(components[0] - sp.sqrt(sp.Rational(2, 3))) == 0
    assert sp.simplify(components[1] - 1 / sp.sqrt(3)) == 0
    assert components[2] == 0
    assert matrix_equal(projector(radial), projectors["a"])


def test_opposite_edge_current_is_unique_b_line() -> None:
    """The current orthogonal to collective and radial lines is b."""

    vectors = residual_vectors()
    edge = opposite_edge_line_from_incidence()

    assert matrix_equal(projector(edge), residual_projectors()["b"])
    assert sp.simplify((edge.T * vectors["u"])[0]) == 0
    assert sp.simplify((edge.T * vectors["a"])[0]) == 0


def test_active_plane_matches_session_10_projector() -> None:
    """Incidence orthogonality gives the same active plane used in Session 10."""

    active_from_incidence = active_projector_from_incidence()
    active_from_channels = active_projector_from_channels()
    radial = radial_projector_from_incidence()
    session_10 = active_neutrino_projector()

    assert matrix_equal(active_from_incidence, session_10)
    assert matrix_equal(active_from_channels, session_10)
    assert active_from_incidence.rank() == 2
    assert radial.rank() == 1
    assert matrix_equal(active_from_incidence + radial, sp.eye(3))


def test_selected_s2_symmetry_alone_does_not_select_active_plane() -> None:
    """An S2-invariant operator can still mix the radial and active sectors."""

    counterexample = selected_s2_symmetry_only_counterexample()
    raw_active = raw_selected_port_active_control()
    session_10 = active_neutrino_projector()

    assert is_selected_s2_invariant(counterexample)
    assert selected_s2_symmetry_alone_not_sufficient()
    assert not matrix_equal(raw_active, session_10)
    assert matrix_equal(projector(selected_boundary_port()), counterexample)


def test_session_11_active_plane_payload() -> None:
    """Session 11 upgrades the projector selection to incidence level only."""

    payload = active_plane_incidence_payload()

    assert payload.final_verdict == "NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS"
    assert payload.detraced_line_matches_a
    assert payload.opposite_edge_line_matches_b
    assert payload.active_projector_matches_session_10
    assert payload.active_radial_resolution_identity
    assert payload.active_projector_selected_s2_invariant
    assert payload.active_projector_not_full_s3_invariant
    assert payload.selected_s2_symmetry_alone_not_sufficient
    assert payload.raw_selected_port_line_control_rejected
    assert payload.session_10_family_moment_gate_passes
    assert "bb_q_mismatch_penalizes" in payload.remaining_declared_inputs[0]
