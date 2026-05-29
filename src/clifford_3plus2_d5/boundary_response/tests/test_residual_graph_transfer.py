"""Tests for the V26 residual graph transfer-recurrence theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import k3_adjacency
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    REMAINING_DECLARED_INPUTS_AFTER_GRAPH_TRANSFER,
    causal_transfer_matrix_from_degree,
    decaying_transfer_factor_from_degree,
    graph_row_sums,
    residual_complete_graph_adjacency,
    residual_graph_decaying_factor,
    residual_graph_degree,
    residual_graph_transfer_audit_payload,
    residual_graph_transfer_matrix,
    transfer_polynomial_from_degree,
)
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    transfer_matrix,
    transfer_polynomial,
)


def test_k3_adjacency_matches_existing_tail_adjacency() -> None:
    assert residual_complete_graph_adjacency(3) == k3_adjacency()


def test_k3_row_sums_are_degree_two() -> None:
    adjacency = residual_complete_graph_adjacency(3)
    assert residual_graph_degree(3) == 2
    assert graph_row_sums(adjacency) == (2, 2, 2)


def test_transfer_matrix_is_radial_quotient_of_k3_degree() -> None:
    assert causal_transfer_matrix_from_degree(2) == sp.Matrix([[2, 1], [1, 0]])
    assert residual_graph_transfer_matrix(3) == transfer_matrix()


def test_k3_degree_derives_existing_transfer_polynomial() -> None:
    eps = sp.symbols("eps")
    assert sp.simplify(
        transfer_polynomial_from_degree(2, eps) - transfer_polynomial(eps)
    ) == 0
    assert sp.simplify(transfer_polynomial_from_degree(2, epsilon())) == 0


def test_k3_degree_derives_existing_decaying_transfer_factor() -> None:
    assert sp.simplify(decaying_transfer_factor_from_degree(2) - epsilon()) == 0
    assert sp.simplify(residual_graph_decaying_factor(3) - epsilon()) == 0


def test_k2_and_k4_controls_have_different_decaying_factors() -> None:
    k2_factor = residual_graph_decaying_factor(2)
    k4_factor = residual_graph_decaying_factor(4)
    assert sp.simplify(k2_factor - epsilon()) != 0
    assert sp.simplify(k4_factor - epsilon()) != 0


def test_scaled_continuation_control_changes_root() -> None:
    scaled_factor = residual_graph_decaying_factor(3, continuation=2)
    assert sp.simplify(scaled_factor - epsilon()) != 0
    eps = sp.symbols("eps")
    assert sp.simplify(
        transfer_polynomial_from_degree(2, eps, continuation=2)
        - (2 * eps**2 + 2 * eps - 1)
    ) == 0


def test_residual_graph_transfer_payload_reports_pass() -> None:
    payload = residual_graph_transfer_audit_payload()
    assert payload.final_verdict == "RESIDUAL_GRAPH_TRANSFER_RECURRENCE_PASS"
    assert payload.residual_graph_size == 3
    assert payload.residual_degree == 2
    assert payload.adjacency_matches_k3_tail
    assert payload.row_sums_equal_degree
    assert payload.derived_transfer_matrix == transfer_matrix()
    assert sp.simplify(payload.derived_polynomial - transfer_polynomial(sp.symbols("eps"))) == 0
    assert sp.simplify(payload.derived_decaying_factor - epsilon()) == 0
    assert payload.negative_controls_rejected
    assert payload.scaled_continuation_changes_factor
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_GRAPH_TRANSFER
    assert payload.remaining_declared_inputs == (
        "vacuum_framing",
        "unit_outward_causal_continuation_or_chain_normalization",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
