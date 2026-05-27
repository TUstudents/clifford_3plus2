"""Tests for the V2 framed sterile effective audit."""

from __future__ import annotations

import inspect

import sympy as sp

from clifford_3plus2_d5.boundary_response.framed_sterile import (
    collective_tail_channel,
    collective_tail_incidence_raw,
    framed_sterile_audit_payload,
    framed_sterile_coupling_matrix,
    framed_sterile_effective_response,
    framed_sterile_target,
    opposite_edge_channel,
    opposite_edge_incidence_raw,
    transfer_depth_amplitude,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    is_selected_s2_invariant,
    projector,
    residual_projectors,
    residual_vectors,
)
from clifford_3plus2_d5.boundary_response.schur import matrix_equal
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def test_raw_incidences_are_local_residual_vectors_not_projectors() -> None:
    assert collective_tail_incidence_raw() == sp.Matrix([1, 1, 1])
    assert opposite_edge_incidence_raw() == sp.Matrix([0, 1, -1])


def test_collective_incidence_selects_u() -> None:
    projectors = residual_projectors()
    assert matrix_equal(projector(collective_tail_channel()), projectors["u"])


def test_opposite_edge_incidence_selects_b_and_is_s2_invariant() -> None:
    edge_projector = projector(opposite_edge_channel())
    assert matrix_equal(edge_projector, residual_projectors()["b"])
    assert is_selected_s2_invariant(edge_projector)


def test_radial_mode_is_absent_from_derived_channels() -> None:
    vectors = residual_vectors()
    assert sp.simplify((collective_tail_channel().T * vectors["a"])[0]) == 0
    assert sp.simplify((opposite_edge_channel().T * vectors["a"])[0]) == 0


def test_transfer_depth_ratio_is_epsilon() -> None:
    ratio = sp.simplify(transfer_depth_amplitude(1) / transfer_depth_amplitude(0))
    assert sp.simplify(ratio - epsilon()) == 0


def test_coupling_matrix_derives_epsilon_from_depth_not_free_parameter() -> None:
    signature = inspect.signature(framed_sterile_coupling_matrix)
    assert "epsilon" not in signature.parameters
    assert "u_weight" not in signature.parameters
    assert "b_weight" not in signature.parameters

    coupling = framed_sterile_coupling_matrix()
    assert coupling.shape == (2, 3)
    assert matrix_equal(coupling[0, :].T, epsilon() * collective_tail_channel())
    assert matrix_equal(coupling[1, :].T, opposite_edge_channel())


def test_equal_return_effective_response_matches_target() -> None:
    assert matrix_equal(framed_sterile_effective_response(), framed_sterile_target())


def test_broken_equal_return_assumption_no_longer_matches_target() -> None:
    response = framed_sterile_effective_response(collective_return=sp.Integer(2))
    assert not matrix_equal(response, framed_sterile_target())


def test_cross_return_assumption_no_longer_matches_target() -> None:
    response = framed_sterile_effective_response(cross_return=sp.Integer(1))
    assert not matrix_equal(response, framed_sterile_target())


def test_framed_sterile_audit_reports_effective_pass_but_keeps_pmns_ckm_parked() -> None:
    payload = framed_sterile_audit_payload()
    assert payload.incidence_verdict == "FRAMED_INCIDENCE_PASS"
    assert payload.transfer_depth_verdict == "TRANSFER_DEPTH_PASS"
    assert payload.response_verdict == "FRAMED_STERILE_EFFECTIVE_PASS"
    assert payload.final_verdict == "FRAMED_STERILE_EFFECTIVE_PASS"
    assert payload.pmns_ckm_parked
    assert not payload.broken_equal_return_matches_target
