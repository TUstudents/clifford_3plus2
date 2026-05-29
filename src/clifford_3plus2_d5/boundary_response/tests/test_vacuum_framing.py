"""Tests for the V27 BCC vacuum-framing orbit theorem."""

from __future__ import annotations

from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import k3_adjacency
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_complete_graph_adjacency,
    residual_graph_decaying_factor,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon
from clifford_3plus2_d5.boundary_response.vacuum_framing import (
    REMAINING_DECLARED_INPUTS_AFTER_VACUUM_FRAMING,
    antipodal_pairs_from_bcc_body_diagonals,
    bcc_unoriented_exit_representatives,
    framed_residual_adjacency,
    framed_residual_exits,
    residual_s3_permutations_from_stabilizer,
    selected_exit_stabilizer_permutations,
    tetrahedral_exit_centroid,
    tetrahedral_exit_gram_matrix,
    unframed_exit_adjacency,
    vacuum_framing_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import (
    bcc_body_diagonal_directions,
    squared_norm,
)


def test_bcc_body_diagonals_are_eight_unique_normalized_directions() -> None:
    directions = bcc_body_diagonal_directions()
    assert len(directions) == 8
    assert len(set(directions)) == 8
    assert all(sp.simplify(squared_norm(direction) - 1) == 0 for direction in directions)


def test_antipodal_quotient_has_four_pairs() -> None:
    pairs = antipodal_pairs_from_bcc_body_diagonals()
    assert len(pairs) == 4
    assert {direction for pair in pairs for direction in pair} == set(
        bcc_body_diagonal_directions()
    )
    for left, right in pairs:
        assert tuple(sp.simplify(-component) for component in left) == right


def test_unoriented_representatives_form_regular_tetrahedron() -> None:
    reps = bcc_unoriented_exit_representatives()
    assert reps == (
        (1 / sp.sqrt(3), 1 / sp.sqrt(3), 1 / sp.sqrt(3)),
        (1 / sp.sqrt(3), -1 / sp.sqrt(3), -1 / sp.sqrt(3)),
        (-1 / sp.sqrt(3), 1 / sp.sqrt(3), -1 / sp.sqrt(3)),
        (-1 / sp.sqrt(3), -1 / sp.sqrt(3), 1 / sp.sqrt(3)),
    )
    expected_gram = sp.Matrix(
        4,
        4,
        lambda row, col: sp.Integer(1) if row == col else -sp.Rational(1, 3),
    )
    assert tetrahedral_exit_gram_matrix() == expected_gram
    assert tetrahedral_exit_centroid() == (0, 0, 0)


def test_framing_one_exit_leaves_three_residual_exits() -> None:
    residual = framed_residual_exits(selected=0)
    assert len(residual) == 3
    assert residual == bcc_unoriented_exit_representatives()[1:]


def test_residual_adjacency_is_k3() -> None:
    assert unframed_exit_adjacency() == residual_complete_graph_adjacency(4)
    assert framed_residual_adjacency(selected=0) == k3_adjacency()


def test_selected_exit_stabilizer_induces_full_residual_s3() -> None:
    stabilizer = selected_exit_stabilizer_permutations(selected=0)
    assert len(stabilizer) == 6
    assert all(perm[0] == 0 for perm in stabilizer)
    induced = residual_s3_permutations_from_stabilizer(selected=0)
    assert induced == tuple(sorted(tuple(perm) for perm in permutations(range(3))))


def test_wrong_framing_controls_do_not_reproduce_epsilon() -> None:
    no_selection_factor = residual_graph_decaying_factor(4)
    two_selected_factor = residual_graph_decaying_factor(2)
    assert sp.simplify(no_selection_factor - epsilon()) != 0
    assert sp.simplify(two_selected_factor - epsilon()) != 0


def test_vacuum_framing_payload_reports_pass() -> None:
    payload = vacuum_framing_audit_payload()
    assert payload.final_verdict == "BCC_VACUUM_FRAMING_ORBIT_PASS"
    assert payload.oriented_direction_count == 8
    assert payload.antipodal_pair_count == 4
    assert payload.representative_count == 4
    assert payload.tetrahedral_gram_matches
    assert payload.centroid_zero
    assert payload.residual_exit_count == 3
    assert payload.residual_adjacency_matches_k3
    assert payload.stabilizer_size == 6
    assert payload.stabilizer_induces_s3
    assert payload.no_selection_control_rejected
    assert payload.two_selected_control_rejected
    assert payload.v26_applies_to_residual_k3
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_VACUUM_FRAMING
    assert payload.remaining_declared_inputs == (
        "physical_vacuum_order_parameter_selects_one_exit",
        "unit_outward_causal_continuation_or_chain_normalization",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
