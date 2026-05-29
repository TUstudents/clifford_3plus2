"""V29 unit outward-continuation normalization gate.

V26 derives the residual transfer recurrence from the framed residual ``K_3``
graph once the outward causal-continuation coefficient is fixed to one.  V29
audits that normalization as an incidence statement.

Let ``M`` map one residual shell to the next outward shell.  The scalar entering
the radial quotient is the common eigenvalue of ``M.T * M``.  A primitive
single outward continuation is the unit matching ``M = I_3``, so
``M.T * M = I_3`` and the continuation coefficient is forced to one.  Scaled,
doubled, or anisotropic controls show that this normalization is load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import sympy as sp

from clifford_3plus2_d5.boundary_response.residual_basis import residual_basis_matrix
from clifford_3plus2_d5.boundary_response.residual_graph_transfer import (
    causal_transfer_matrix_from_degree,
    decaying_transfer_factor_from_degree,
    residual_graph_transfer_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon, transfer_matrix

REMAINING_DECLARED_INPUTS_AFTER_UNIT_CONTINUATION = (
    "physical_vacuum_order_parameter_exists",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    if left.shape != right.shape:
        return False
    return all(
        sp.simplify(left[row, col] - right[row, col]) == 0
        for row in range(left.rows)
        for col in range(left.cols)
    )


def unit_outward_matching(size: int = 3) -> sp.Matrix:
    """Return the primitive one-to-one outward shell matching."""

    if size < 1:
        raise ValueError("size must be positive")
    return sp.eye(size)


def scaled_outward_matching(scale: sp.Expr, size: int = 3) -> sp.Matrix:
    """Return the scaled-matching control ``scale * I``."""

    return sp.sympify(scale) * unit_outward_matching(size)


def double_outward_matching(size: int = 3) -> sp.Matrix:
    """Return the two-copy outward-continuation control."""

    identity = unit_outward_matching(size)
    return identity.col_join(identity)


def anisotropic_outward_matching() -> sp.Matrix:
    """Return an anisotropic diagonal control with no scalar quotient."""

    return sp.diag(1, 2, 1)


def residual_permutation_matching(perm: tuple[int, int, int]) -> sp.Matrix:
    """Return a residual-label permutation matching."""

    if sorted(perm) != [0, 1, 2]:
        raise ValueError("perm must be a permutation of (0, 1, 2)")
    matrix = sp.zeros(3, 3)
    for source, target in enumerate(perm):
        matrix[target, source] = 1
    return matrix


def continuation_gram(matching: sp.Matrix) -> sp.Matrix:
    """Return ``M.T * M`` for an outward matching operator."""

    return (matching.T * matching).applyfunc(sp.simplify)


def residual_basis_continuation_matrix(
    matching: sp.Matrix,
    order: tuple[str, ...] = ("u", "a", "b"),
) -> sp.Matrix:
    """Return the continuation Gram matrix in the residual eigenbasis."""

    basis = residual_basis_matrix(order)
    return (basis.T * continuation_gram(matching) * basis).applyfunc(sp.simplify)


def residual_basis_continuation_couplings(
    matching: sp.Matrix,
    order: tuple[str, ...] = ("u", "a", "b"),
) -> tuple[sp.Expr, ...]:
    """Return diagonal continuation couplings in the residual eigenbasis."""

    matrix = residual_basis_continuation_matrix(matching, order)
    return tuple(sp.simplify(matrix[index, index]) for index in range(matrix.rows))


def uniform_continuation_coefficient(matching: sp.Matrix) -> sp.Expr | None:
    """Return the scalar ``c`` when ``M.T * M = c I``, else ``None``."""

    gram = continuation_gram(matching)
    if gram.rows != gram.cols:
        return None
    coefficient = sp.simplify(gram[0, 0])
    if _matrix_equal(gram, coefficient * sp.eye(gram.rows)):
        return coefficient
    return None


def is_unit_outward_continuation(matching: sp.Matrix) -> bool:
    """Return whether the matching has exactly unit continuation norm."""

    coefficient = uniform_continuation_coefficient(matching)
    return coefficient is not None and sp.simplify(coefficient - 1) == 0


def residual_permutation_matchings_are_unit_gauge_equivalent() -> bool:
    """Return whether all residual label permutations preserve unit norm."""

    return all(
        is_unit_outward_continuation(residual_permutation_matching(tuple(perm)))
        for perm in permutations((0, 1, 2))
    )


def transfer_matrix_from_matching(
    matching: sp.Matrix,
    *,
    degree: sp.Expr = sp.Integer(2),
) -> sp.Matrix:
    """Return the radial quotient transfer matrix from a uniform matching."""

    coefficient = uniform_continuation_coefficient(matching)
    if coefficient is None:
        raise ValueError("matching does not define a scalar continuation quotient")
    return causal_transfer_matrix_from_degree(degree, continuation=coefficient)


def decaying_factor_from_matching(
    matching: sp.Matrix,
    *,
    degree: sp.Expr = sp.Integer(2),
) -> sp.Expr:
    """Return the decaying transfer factor from a uniform matching."""

    coefficient = uniform_continuation_coefficient(matching)
    if coefficient is None:
        raise ValueError("matching does not define a scalar continuation quotient")
    return decaying_transfer_factor_from_degree(degree, continuation=coefficient)


@dataclass(frozen=True)
class UnitContinuationAuditPayload:
    """Verdict payload for the V29 unit outward-continuation gate."""

    final_verdict: str
    matching_shape: tuple[int, int]
    continuation_gram_matrix: sp.Matrix
    residual_basis_couplings: tuple[sp.Expr, ...]
    uniform_continuation_coefficient: sp.Expr | None
    derived_transfer_matrix: sp.Matrix
    derived_decaying_factor: sp.Expr
    scaled_control_factor: sp.Expr
    scaled_control_rejected: bool
    double_control_coefficient: sp.Expr | None
    double_control_rejected: bool
    anisotropic_control_rejected: bool
    permutation_matchings_gauge_equivalent: bool
    v26_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def unit_continuation_audit_payload() -> UnitContinuationAuditPayload:
    """Return the V29 unit outward-continuation verdict."""

    matching = unit_outward_matching()
    gram = continuation_gram(matching)
    couplings = residual_basis_continuation_couplings(matching)
    coefficient = uniform_continuation_coefficient(matching)
    derived_matrix = transfer_matrix_from_matching(matching)
    derived_factor = decaying_factor_from_matching(matching)

    scaled_factor = decaying_factor_from_matching(scaled_outward_matching(2))
    scaled_rejected = sp.simplify(scaled_factor - epsilon()) != 0

    double_coefficient = uniform_continuation_coefficient(double_outward_matching())
    double_rejected = (
        double_coefficient is not None
        and sp.simplify(double_coefficient - coefficient) != 0
        and sp.simplify(decaying_factor_from_matching(double_outward_matching()) - epsilon()) != 0
    )

    anisotropic_rejected = uniform_continuation_coefficient(
        anisotropic_outward_matching()
    ) is None
    permutation_equivalent = residual_permutation_matchings_are_unit_gauge_equivalent()

    v26 = residual_graph_transfer_audit_payload()
    v26_recovered = (
        v26.final_verdict == "RESIDUAL_GRAPH_TRANSFER_RECURRENCE_PASS"
        and _matrix_equal(derived_matrix, transfer_matrix())
        and sp.simplify(derived_factor - epsilon()) == 0
    )

    checks_pass = (
        _matrix_equal(gram, sp.eye(3))
        and couplings == (sp.Integer(1), sp.Integer(1), sp.Integer(1))
        and coefficient == sp.Integer(1)
        and _matrix_equal(derived_matrix, transfer_matrix())
        and sp.simplify(derived_factor - epsilon()) == 0
        and scaled_rejected
        and double_rejected
        and anisotropic_rejected
        and permutation_equivalent
        and v26_recovered
    )

    if checks_pass:
        final_verdict = "UNIT_OUTWARD_CONTINUATION_NORMALIZATION_PASS"
        interpretation = (
            "A primitive one-to-one outward shell matching has M.T*M = I, so "
            "the radial quotient continuation coefficient is one in every "
            "residual mode. This recovers the V26 transfer matrix and "
            "epsilon exactly. Scaled, doubled, and anisotropic controls fail, "
            "while residual-label permutations are gauge-equivalent unit "
            "matchings."
        )
    else:
        final_verdict = "UNIT_OUTWARD_CONTINUATION_NORMALIZATION_KILL"
        interpretation = (
            "The unit matching, residual couplings, transfer recovery, "
            "negative controls, permutation gauge equivalence, or V26 "
            "compatibility failed."
        )

    return UnitContinuationAuditPayload(
        final_verdict=final_verdict,
        matching_shape=matching.shape,
        continuation_gram_matrix=gram,
        residual_basis_couplings=couplings,
        uniform_continuation_coefficient=coefficient,
        derived_transfer_matrix=derived_matrix,
        derived_decaying_factor=derived_factor,
        scaled_control_factor=scaled_factor,
        scaled_control_rejected=scaled_rejected,
        double_control_coefficient=double_coefficient,
        double_control_rejected=double_rejected,
        anisotropic_control_rejected=anisotropic_rejected,
        permutation_matchings_gauge_equivalent=permutation_equivalent,
        v26_recovered=v26_recovered,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_UNIT_CONTINUATION,
        interpretation=interpretation,
    )
