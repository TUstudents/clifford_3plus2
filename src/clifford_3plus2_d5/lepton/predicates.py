"""Predicate implementations for the experimental lepton verdict profiles."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, identity
from clifford_3plus2_d5.qca.rule_verdict import (
    CentralIdempotent,
    ComplexStructureCandidate,
)
from clifford_3plus2_d5.lepton.verdict import (
    CentralJSolveResult,
    CentralJSolveStatus,
    CommutantPolicyResult,
    IdempotentPolicyResult,
    VerdictProfile,
)
from clifford_3plus2_d5.lepton.wall import WallContext


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def central_j_candidates_default(
    profile: VerdictProfile,
    center_basis: Sequence[sp.Matrix],
) -> CentralJSolveResult:
    if len(center_basis) > profile.max_center_dimension:
        return CentralJSolveResult(CentralJSolveStatus.TOO_LARGE)
    if len(center_basis) <= 1:
        return CentralJSolveResult(CentralJSolveStatus.NO_SOLUTIONS)

    variables = sp.symbols(f"z0:{len(center_basis)}")
    candidate = sp.zeros(profile.dimension)
    for variable, basis_matrix in zip(variables, center_basis, strict=True):
        candidate += variable * basis_matrix

    equations = []
    for value in candidate * candidate + identity(profile.dimension):
        expanded = sp.expand(value)
        if expanded != 0 and expanded not in equations:
            equations.append(expanded)
    for value in candidate.T * candidate - identity(profile.dimension):
        expanded = sp.expand(value)
        if expanded != 0 and expanded not in equations:
            equations.append(expanded)

    try:
        solutions = sp.solve(equations, variables, dict=True, simplify=False)
    except (NotImplementedError, TypeError, ValueError):
        return CentralJSolveResult(CentralJSolveStatus.UNSUPPORTED)

    candidates = []
    seen: set[tuple[sp.Expr, ...]] = set()
    for solution in solutions:
        if not all(variable in solution for variable in variables):
            return CentralJSolveResult(CentralJSolveStatus.UNSUPPORTED)
        coefficients = tuple(sp.simplify(solution[variable]) for variable in variables)
        matrix = sp.zeros(profile.dimension)
        for coefficient, basis_matrix in zip(coefficients, center_basis, strict=True):
            matrix += coefficient * basis_matrix
        matrix = matrix.applyfunc(sp.simplify)
        if any(sp.simplify(value.as_real_imag()[1]) != 0 for value in matrix):
            continue
        key = tuple(sp.simplify(value) for value in matrix)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            ComplexStructureCandidate(
                expression=tuple(
                    (f"center_{index}", coefficient)
                    for index, coefficient in enumerate(coefficients)
                    if coefficient != 0
                ),
                matrix=matrix,
                source="local_compatible_center",
            )
        )
    if not candidates:
        return CentralJSolveResult(CentralJSolveStatus.NO_SOLUTIONS)
    return CentralJSolveResult(CentralJSolveStatus.SOLVED, tuple(candidates))


def lab_a_idempotent_policy(
    profile: VerdictProfile,
    idempotents: Sequence[CentralIdempotent],
) -> IdempotentPolicyResult:
    allowed_ranks = {0, profile.dimension}
    if all(item.rank in allowed_ranks for item in idempotents):
        return IdempotentPolicyResult.PASSED
    return IdempotentPolicyResult.FORBIDDEN_PRESENT


def strict_split_idempotent_policy(
    profile: VerdictProfile,
    idempotents: Sequence[CentralIdempotent],
) -> IdempotentPolicyResult:
    one = identity(profile.dimension)
    zero = sp.zeros(profile.dimension)
    allowed = (zero, *profile.target_projectors, one)
    for idempotent in idempotents:
        if not any(_same_matrix(idempotent.matrix, candidate) for candidate in allowed):
            return IdempotentPolicyResult.FORBIDDEN_PRESENT

    for projector in profile.target_projectors:
        if not any(_same_matrix(idempotent.matrix, projector) for idempotent in idempotents):
            return IdempotentPolicyResult.MISSING_TARGET
    return IdempotentPolicyResult.PASSED


def domain_wall_trivial_center_policy(
    profile: VerdictProfile,
    idempotents: Sequence[CentralIdempotent],
) -> IdempotentPolicyResult:
    nontrivial = [item for item in idempotents if item.rank not in (0, profile.dimension)]
    if nontrivial:
        return IdempotentPolicyResult.FORBIDDEN_PRESENT
    return IdempotentPolicyResult.PASSED


def _unique_pm_or_multiple(
    candidates: Sequence[ComplexStructureCandidate],
) -> CommutantPolicyResult:
    if len(candidates) == 2:
        return CommutantPolicyResult.PASSED_UNIQUE_PM
    if len(candidates) > 2:
        return CommutantPolicyResult.PASSED_MULTIPLE_ALIGNED
    return CommutantPolicyResult.GAUGE_MISALIGNMENT


def no_commutant_check(
    profile: VerdictProfile,
    algebra_basis: Sequence[sp.Matrix],
    center_basis: Sequence[sp.Matrix],
    target_projectors: Sequence[sp.Matrix],
    j_candidates: Sequence[ComplexStructureCandidate],
    wall_context: object | None = None,
) -> CommutantPolicyResult:
    del profile, algebra_basis, center_basis, target_projectors, wall_context
    return _unique_pm_or_multiple(j_candidates)


def j_centralizes_gauge_all_candidates(
    profile: VerdictProfile,
    algebra_basis: Sequence[sp.Matrix],
    center_basis: Sequence[sp.Matrix],
    target_projectors: Sequence[sp.Matrix],
    j_candidates: Sequence[ComplexStructureCandidate],
    wall_context: object | None = None,
) -> CommutantPolicyResult:
    del algebra_basis, center_basis, target_projectors, wall_context
    if not profile.gauge_generators:
        return _unique_pm_or_multiple(j_candidates)

    passing = []
    zero = sp.zeros(profile.dimension)
    for candidate in j_candidates:
        if all(
            (commutator(candidate.matrix, generator)).applyfunc(sp.simplify) == zero
            for generator in profile.gauge_generators
        ):
            passing.append(candidate)
    return _unique_pm_or_multiple(passing)


def side_local_gauge_with_wall_transition(
    profile: VerdictProfile,
    algebra_basis: Sequence[sp.Matrix],
    center_basis: Sequence[sp.Matrix],
    target_projectors: Sequence[sp.Matrix],
    j_candidates: Sequence[ComplexStructureCandidate],
    wall_context: object | None = None,
) -> CommutantPolicyResult:
    """Check side-local gauge/J compatibility for a domain-wall context."""

    del profile, algebra_basis, center_basis, target_projectors
    if not isinstance(wall_context, WallContext):
        return CommutantPolicyResult.GAUGE_MISALIGNMENT
    if not wall_context.consistency_certified():
        return CommutantPolicyResult.GAUGE_MISALIGNMENT

    left_projector = wall_context.left_side_projector
    right_projector = wall_context.right_side_projector
    left_j = wall_context.left_complex_structure
    right_j = wall_context.right_complex_structure
    passing = []
    for candidate in j_candidates:
        matrix = candidate.matrix
        left_restriction = (left_projector * matrix * left_projector).applyfunc(sp.simplify)
        right_restriction = (right_projector * matrix * right_projector).applyfunc(sp.simplify)

        left_positive = (left_projector * left_j * left_projector).applyfunc(sp.simplify)
        left_negative = (-left_projector * left_j * left_projector).applyfunc(sp.simplify)
        right_positive = (right_projector * right_j * right_projector).applyfunc(sp.simplify)
        right_negative = (-right_projector * right_j * right_projector).applyfunc(sp.simplify)

        left_sign = 1 if left_restriction == left_positive else -1 if left_restriction == left_negative else 0
        right_sign = (
            1
            if right_restriction == right_positive
            else -1
            if right_restriction == right_negative
            else 0
        )
        if left_sign == 0 or right_sign == 0 or left_sign != right_sign:
            continue

        ok_gauge = all(
            _same_matrix(commutator(matrix, left_generator), sp.zeros(matrix.rows))
            for left_generator, _ in wall_context.gauge_pairs
        )
        if ok_gauge:
            passing.append(candidate)

    return _unique_pm_or_multiple(passing)
