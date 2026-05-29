"""V27 BCC vacuum-framing orbit theorem.

The boundary-response chain assumes that vacuum framing selects one primitive
BCC exit and leaves three residual family exits.  V27 derives the finite
orbit statement from the BCC body-diagonal geometry:

    8 oriented BCC exits / antipodal pairing -> 4 tetrahedral exits,
    select one framed exit -> 3 residual exits,
    residual adjacency -> K_3,
    selected-exit stabilizer S_4 -> S_3.

This does not derive the physical vacuum order parameter that selects the
framed exit.  It proves the exact orbit quotient once that selection is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import k3_adjacency
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    residual_complete_graph_adjacency,
    residual_graph_decaying_factor,
    residual_graph_transfer_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon
from clifford_3plus2_d5.spacetime_qca.bcc_geometry import (
    Vector3,
    bcc_body_diagonal_directions,
    vector_dot,
)

REMAINING_DECLARED_INPUTS_AFTER_VACUUM_FRAMING = (
    "physical_vacuum_order_parameter_selects_one_exit",
    "unit_outward_causal_continuation_or_chain_normalization",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def _negate_vector(vector: Vector3) -> Vector3:
    """Return the antipodal BCC direction."""

    return tuple(sp.simplify(-component) for component in vector)  # type: ignore[return-value]


def _validate_exit_index(selected: int, *, size: int = 4) -> None:
    """Raise when ``selected`` is not a valid tetrahedral-exit index."""

    if selected < 0 or selected >= size:
        raise ValueError(f"selected must be in [0, {size})")


def bcc_unoriented_exit_representatives(*, normalized: bool = True) -> tuple[Vector3, ...]:
    """Return four antipodal-pair representatives forming a regular tetrahedron."""

    scale = sp.sqrt(3) if normalized else sp.Integer(1)
    raw = (
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    )
    return tuple(
        tuple(sp.Integer(component) / scale for component in vector)  # type: ignore[misc]
        for vector in raw
    )


def antipodal_pairs_from_bcc_body_diagonals() -> tuple[tuple[Vector3, Vector3], ...]:
    """Return the four unoriented BCC exit pairs."""

    directions = set(bcc_body_diagonal_directions())
    pairs = tuple(
        (representative, _negate_vector(representative))
        for representative in bcc_unoriented_exit_representatives()
    )
    paired = {direction for pair in pairs for direction in pair}
    if paired != directions:
        raise ValueError("representatives do not partition BCC body diagonals")
    return pairs


def tetrahedral_exit_gram_matrix() -> sp.Matrix:
    """Return the Gram matrix of the four primitive unoriented exits."""

    reps = bcc_unoriented_exit_representatives()
    return sp.Matrix(
        [
            [vector_dot(left, right) for right in reps]
            for left in reps
        ]
    )


def tetrahedral_exit_centroid() -> Vector3:
    """Return the centroid of the four tetrahedral exit representatives."""

    reps = bcc_unoriented_exit_representatives()
    return tuple(
        sp.simplify(sum(vector[index] for vector in reps) / len(reps))
        for index in range(3)
    )


def unframed_exit_adjacency() -> sp.Matrix:
    """Return the complete graph on the four unframed tetrahedral exits."""

    return residual_complete_graph_adjacency(4)


def framed_residual_exits(selected: int = 0) -> tuple[Vector3, ...]:
    """Return residual exits after selecting one framed vacuum exit."""

    reps = bcc_unoriented_exit_representatives()
    _validate_exit_index(selected, size=len(reps))
    return tuple(exit_vector for index, exit_vector in enumerate(reps) if index != selected)


def framed_residual_adjacency(selected: int = 0) -> sp.Matrix:
    """Return the complete residual adjacency after selecting one exit."""

    return residual_complete_graph_adjacency(len(framed_residual_exits(selected)))


def selected_exit_stabilizer_permutations(selected: int = 0) -> tuple[tuple[int, ...], ...]:
    """Return the ``S_4`` permutations that fix the selected exit."""

    _validate_exit_index(selected)
    return tuple(
        tuple(perm)
        for perm in permutations(range(4))
        if perm[selected] == selected
    )


def induced_residual_permutation(
    permutation: tuple[int, ...],
    selected: int = 0,
) -> tuple[int, ...]:
    """Return the permutation induced on residual-exit coordinates."""

    _validate_exit_index(selected, size=len(permutation))
    residual_labels = tuple(index for index in range(len(permutation)) if index != selected)
    return tuple(residual_labels.index(permutation[label]) for label in residual_labels)


def residual_s3_permutations_from_stabilizer(selected: int = 0) -> tuple[tuple[int, ...], ...]:
    """Return induced residual ``S_3`` permutations from the selected stabilizer."""

    induced = {
        induced_residual_permutation(perm, selected)
        for perm in selected_exit_stabilizer_permutations(selected)
    }
    return tuple(sorted(induced))


@dataclass(frozen=True)
class VacuumFramingAuditPayload:
    """Verdict payload for the V27 BCC vacuum-framing orbit theorem."""

    final_verdict: str
    oriented_direction_count: int
    antipodal_pair_count: int
    representative_count: int
    tetrahedral_gram_matches: bool
    centroid_zero: bool
    residual_exit_count: int
    residual_adjacency_matches_k3: bool
    stabilizer_size: int
    stabilizer_induces_s3: bool
    no_selection_control_rejected: bool
    two_selected_control_rejected: bool
    v26_applies_to_residual_k3: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def vacuum_framing_audit_payload() -> VacuumFramingAuditPayload:
    """Return the V27 BCC vacuum-framing orbit verdict."""

    directions = bcc_body_diagonal_directions()
    pairs = antipodal_pairs_from_bcc_body_diagonals()
    reps = bcc_unoriented_exit_representatives()
    gram = tetrahedral_exit_gram_matrix()
    expected_gram = sp.Matrix(
        4,
        4,
        lambda row, col: sp.Integer(1) if row == col else -sp.Rational(1, 3),
    )
    gram_matches = gram == expected_gram
    centroid_zero = all(component == 0 for component in tetrahedral_exit_centroid())
    residual_exits = framed_residual_exits(0)
    residual_adjacency_matches = framed_residual_adjacency(0) == k3_adjacency()
    stabilizer = selected_exit_stabilizer_permutations(0)
    induced = residual_s3_permutations_from_stabilizer(0)
    all_s3 = tuple(sorted(tuple(perm) for perm in permutations(range(3))))
    stabilizer_induces_s3 = induced == all_s3
    no_selection_control_rejected = sp.simplify(
        residual_graph_decaying_factor(4) - epsilon()
    ) != 0
    two_selected_control_rejected = sp.simplify(
        residual_graph_decaying_factor(2) - epsilon()
    ) != 0
    v26 = residual_graph_transfer_audit_payload()
    v26_applies = (
        v26.final_verdict == "RESIDUAL_GRAPH_TRANSFER_RECURRENCE_PASS"
        and residual_adjacency_matches
    )

    checks_pass = (
        len(directions) == 8
        and len(set(directions)) == 8
        and len(pairs) == 4
        and len(reps) == 4
        and gram_matches
        and centroid_zero
        and unframed_exit_adjacency() == residual_complete_graph_adjacency(4)
        and len(residual_exits) == 3
        and residual_adjacency_matches
        and len(stabilizer) == 6
        and stabilizer_induces_s3
        and no_selection_control_rejected
        and two_selected_control_rejected
        and v26_applies
    )

    if checks_pass:
        final_verdict = "BCC_VACUUM_FRAMING_ORBIT_PASS"
        interpretation = (
            "The eight oriented BCC body-diagonal exits quotient by antipodal "
            "pairing to four tetrahedral primitive exits. Selecting one "
            "framed exit leaves three residual exits with K3 adjacency, and "
            "the selected-exit stabilizer induces all residual S3 "
            "permutations. No-selection K4 and two-residual K2 controls fail "
            "to reproduce epsilon. The physical vacuum order parameter that "
            "chooses the framed exit remains a named input."
        )
    else:
        final_verdict = "BCC_VACUUM_FRAMING_ORBIT_KILL"
        interpretation = (
            "The BCC antipodal quotient, tetrahedral Gram matrix, selected "
            "residual K3, stabilizer action, controls, or V26 compatibility "
            "failed."
        )

    return VacuumFramingAuditPayload(
        final_verdict=final_verdict,
        oriented_direction_count=len(directions),
        antipodal_pair_count=len(pairs),
        representative_count=len(reps),
        tetrahedral_gram_matches=gram_matches,
        centroid_zero=centroid_zero,
        residual_exit_count=len(residual_exits),
        residual_adjacency_matches_k3=residual_adjacency_matches,
        stabilizer_size=len(stabilizer),
        stabilizer_induces_s3=stabilizer_induces_s3,
        no_selection_control_rejected=no_selection_control_rejected,
        two_selected_control_rejected=two_selected_control_rejected,
        v26_applies_to_residual_k3=v26_applies,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_VACUUM_FRAMING,
        interpretation=interpretation,
    )
