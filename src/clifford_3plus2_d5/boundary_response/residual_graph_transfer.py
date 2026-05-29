"""V26 residual graph transfer-recurrence theorem.

The transfer invariant used by the boundary-response sidecar was previously
kept as the isolated recurrence

    x_n = 2 x_{n+1} + x_{n+2}.

V26 derives the coefficient ``2`` from the residual complete graph ``K_3``.
After vacuum framing, the three residual family ports form ``K_3``.  Its
regular degree is two.  The minimal radial quotient with unit outward causal
continuation therefore has transfer matrix

    [[2, 1],
     [1, 0]]

and decaying polynomial

    epsilon^2 + 2 epsilon - 1 = 0.

This does not derive vacuum framing or the unit continuation normalization.
It derives the recurrence coefficient from the residual graph quotient once
those structural inputs are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.k3_tail import k3_adjacency
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    transfer_matrix,
    transfer_polynomial,
)

REMAINING_DECLARED_INPUTS_AFTER_GRAPH_TRANSFER = (
    "vacuum_framing",
    "unit_outward_causal_continuation_or_chain_normalization",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def residual_complete_graph_adjacency(size: int) -> sp.Matrix:
    """Return the adjacency matrix of the residual complete graph ``K_size``."""

    if size < 1:
        raise ValueError("size must be positive")
    return sp.ones(size, size) - sp.eye(size)


def residual_graph_degree(size: int) -> int:
    """Return the regular degree of ``K_size``."""

    if size < 1:
        raise ValueError("size must be positive")
    return size - 1


def graph_row_sums(adjacency: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact row sums for a graph adjacency matrix."""

    return tuple(sp.simplify(sum(adjacency[row, :])) for row in range(adjacency.rows))


def causal_transfer_matrix_from_degree(
    degree: sp.Expr,
    *,
    continuation: sp.Expr = sp.Integer(1),
) -> sp.Matrix:
    """Return the radial quotient transfer matrix."""

    selected_degree = sp.sympify(degree)
    selected_continuation = sp.sympify(continuation)
    return sp.Matrix([[selected_degree, selected_continuation], [1, 0]])


def transfer_polynomial_from_degree(
    degree: sp.Expr,
    transfer_factor: sp.Expr,
    *,
    continuation: sp.Expr = sp.Integer(1),
) -> sp.Expr:
    """Return ``continuation eps^2 + degree eps - 1``."""

    selected_degree = sp.sympify(degree)
    selected_factor = sp.sympify(transfer_factor)
    selected_continuation = sp.sympify(continuation)
    return sp.expand(
        selected_continuation * selected_factor**2
        + selected_degree * selected_factor
        - 1
    )


def decaying_transfer_factor_from_degree(
    degree: sp.Expr,
    *,
    continuation: sp.Expr = sp.Integer(1),
) -> sp.Expr:
    """Return the positive decaying transfer root for the radial quotient."""

    selected_degree = sp.sympify(degree)
    selected_continuation = sp.sympify(continuation)
    if sp.simplify(selected_continuation) == 0:
        raise ValueError("continuation must be nonzero")
    return sp.simplify(
        (-selected_degree + sp.sqrt(selected_degree**2 + 4 * selected_continuation))
        / (2 * selected_continuation)
    )


def residual_graph_transfer_matrix(
    size: int,
    *,
    continuation: sp.Expr = sp.Integer(1),
) -> sp.Matrix:
    """Return the radial transfer matrix derived from ``K_size``."""

    return causal_transfer_matrix_from_degree(
        residual_graph_degree(size),
        continuation=continuation,
    )


def residual_graph_decaying_factor(
    size: int,
    *,
    continuation: sp.Expr = sp.Integer(1),
) -> sp.Expr:
    """Return the decaying transfer factor derived from ``K_size``."""

    return decaying_transfer_factor_from_degree(
        residual_graph_degree(size),
        continuation=continuation,
    )


@dataclass(frozen=True)
class ResidualGraphTransferAuditPayload:
    """Verdict payload for the V26 residual graph transfer theorem."""

    final_verdict: str
    residual_graph_size: int
    residual_degree: int
    adjacency_matches_k3_tail: bool
    row_sums_equal_degree: bool
    derived_transfer_matrix: sp.Matrix
    derived_polynomial: sp.Expr
    derived_decaying_factor: sp.Expr
    k2_control_factor: sp.Expr
    k4_control_factor: sp.Expr
    negative_controls_rejected: bool
    scaled_continuation_factor: sp.Expr
    scaled_continuation_changes_factor: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def residual_graph_transfer_audit_payload() -> ResidualGraphTransferAuditPayload:
    """Return the V26 residual graph transfer verdict."""

    graph_size = 3
    degree = residual_graph_degree(graph_size)
    adjacency = residual_complete_graph_adjacency(graph_size)
    row_sums = graph_row_sums(adjacency)
    row_sums_match = all(sp.simplify(row_sum - degree) == 0 for row_sum in row_sums)
    adjacency_matches = adjacency == k3_adjacency()
    derived_matrix = residual_graph_transfer_matrix(graph_size)
    eps_symbol = sp.symbols("eps")
    derived_polynomial = transfer_polynomial_from_degree(degree, eps_symbol)
    derived_factor = residual_graph_decaying_factor(graph_size)
    k2_factor = residual_graph_decaying_factor(2)
    k4_factor = residual_graph_decaying_factor(4)
    negative_controls_rejected = (
        sp.simplify(k2_factor - epsilon()) != 0
        and sp.simplify(k4_factor - epsilon()) != 0
    )
    scaled_factor = residual_graph_decaying_factor(graph_size, continuation=2)
    scaled_changes = sp.simplify(scaled_factor - epsilon()) != 0

    checks_pass = (
        adjacency_matches
        and degree == 2
        and row_sums_match
        and derived_matrix == transfer_matrix()
        and sp.simplify(derived_polynomial - transfer_polynomial(eps_symbol)) == 0
        and sp.simplify(derived_factor - epsilon()) == 0
        and negative_controls_rejected
        and scaled_changes
    )

    if checks_pass:
        final_verdict = "RESIDUAL_GRAPH_TRANSFER_RECURRENCE_PASS"
        interpretation = (
            "The residual K3 graph has degree two. With unit outward causal "
            "continuation, its radial quotient transfer matrix is [[2, 1], "
            "[1, 0]], giving epsilon^2 + 2 epsilon - 1 = 0 and the decaying "
            "root sqrt(2)-1. K2, K4, and scaled-continuation controls change "
            "the root, so vacuum framing and unit continuation remain named "
            "inputs."
        )
    else:
        final_verdict = "RESIDUAL_GRAPH_TRANSFER_RECURRENCE_KILL"
        interpretation = (
            "The K3 adjacency, graph degree, transfer matrix, recurrence, "
            "decaying root, negative controls, or scaled-continuation control "
            "failed."
        )

    return ResidualGraphTransferAuditPayload(
        final_verdict=final_verdict,
        residual_graph_size=graph_size,
        residual_degree=degree,
        adjacency_matches_k3_tail=adjacency_matches,
        row_sums_equal_degree=row_sums_match,
        derived_transfer_matrix=derived_matrix,
        derived_polynomial=derived_polynomial,
        derived_decaying_factor=derived_factor,
        k2_control_factor=k2_factor,
        k4_control_factor=k4_factor,
        negative_controls_rejected=negative_controls_rejected,
        scaled_continuation_factor=scaled_factor,
        scaled_continuation_changes_factor=scaled_changes,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_GRAPH_TRANSFER,
        interpretation=interpretation,
    )
