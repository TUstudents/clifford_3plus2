"""Complex-linear split-first verdict path.

This path assumes the complex structure is fundamental. The question is not
whether a rule derives ``J``; it is whether a complex-linear rule algebra
derives a target central split without extra central locking.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import exact_matrix_span


class ComplexSplitVerdict(StrEnum):
    SPLIT_CANDIDATE = "split_candidate"
    SEEDED_SPLIT_CONTROL = "seeded_split_control"
    FALSIFIED_NO_SPLIT = "falsified_no_split"
    FALSIFIED_RANK_ONE_LOCKING = "falsified_rank_one_locking"
    FALSIFIED_FORBIDDEN_IDEMPOTENT = "falsified_forbidden_idempotent"
    FALSIFIED_COMMUTATIVE = "falsified_commutative"
    NOT_SOLVED = "not_solved"


@dataclass(frozen=True)
class ComplexLayerInput:
    name: str
    matrix: sp.Matrix
    is_seeded_split_layer: bool = False
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ComplexAlgebraClosure:
    basis: tuple[sp.Matrix, ...]
    closed: bool


@dataclass(frozen=True)
class ComplexCentralIdempotent:
    expression: tuple[tuple[str, sp.Expr], ...]
    matrix: sp.Matrix
    rank: int


@dataclass(frozen=True)
class ComplexSplitProfile:
    name: str
    dimension: int
    target_projectors: tuple[sp.Matrix, ...]
    max_algebra_dimension: int = 64
    max_center_dimension: int = 6
    require_noncommutative: bool = True
    expected_block_dimensions: tuple[int, ...] | None = None
    expected_block_commutativity: tuple[bool, ...] | None = None
    idempotent_policy: Callable[
        ["ComplexSplitProfile", Sequence[ComplexCentralIdempotent]], str
    ] | None = None
    target_ranks: tuple[int, int] | None = None
    allow_discovered_split: bool = False


@dataclass(frozen=True)
class ComplexSplitResult:
    profile_name: str
    verdict: ComplexSplitVerdict
    reason: str
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int
    central_idempotents: tuple[ComplexCentralIdempotent, ...]
    target_split_present: bool
    seeded_split_present: bool
    forbidden_rank_one_count: int
    forbidden_idempotent_count: int
    block_dimensions: tuple[int, ...]
    block_commutativity: tuple[bool, ...]
    discovered_projectors: tuple[sp.Matrix, ...] = ()
    canonical_split_matched: bool = False
    discovered_projector_ranks: tuple[int, ...] = ()


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _identity(dimension: int) -> sp.Matrix:
    return sp.eye(dimension)


def _span_rank(matrices: Sequence[sp.Matrix], *, dimension: int) -> int:
    if not matrices:
        return 0
    return exact_matrix_span(matrices, rows=dimension, cols=dimension).rank


def generated_complex_algebra_closure(
    generators: Sequence[sp.Matrix],
    *,
    dimension: int,
    max_dimension: int,
) -> ComplexAlgebraClosure:
    span = exact_matrix_span(rows=dimension, cols=dimension, add_matrices=False)
    span.add(_identity(dimension))
    for generator in generators:
        if generator.shape != (dimension, dimension):
            raise ValueError("generator shape does not match profile dimension")
        span.add(generator.applyfunc(sp.simplify))

    index = 0
    while index < len(span.basis_matrices):
        left = span.basis_matrices[index]
        current_basis = tuple(span.basis_matrices)
        for right in current_basis:
            for product in (left * right, right * left):
                span.add(product.applyfunc(sp.simplify))
                if span.rank >= max_dimension:
                    return ComplexAlgebraClosure(tuple(span.basis_matrices), False)
        index += 1
    return ComplexAlgebraClosure(tuple(span.basis_matrices), True)


def complex_center_basis(
    algebra_basis: Sequence[sp.Matrix],
    *,
    dimension: int,
) -> tuple[sp.Matrix, ...]:
    variables = sp.symbols(f"x0:{dimension * dimension}")
    candidate = sp.Matrix(dimension, dimension, variables)
    equations = [
        value
        for algebra_matrix in algebra_basis
        for value in candidate * algebra_matrix - algebra_matrix * candidate
    ]
    coefficient_matrix, _ = sp.linear_eq_to_matrix(equations, variables)
    return tuple(
        sp.Matrix(dimension, dimension, vector).applyfunc(sp.simplify)
        for vector in coefficient_matrix.nullspace()
    )


def solve_complex_central_idempotents(
    center_basis: Sequence[sp.Matrix],
    *,
    dimension: int,
    max_center_dimension: int,
) -> tuple[bool, tuple[ComplexCentralIdempotent, ...]]:
    if len(center_basis) > max_center_dimension:
        return False, ()

    variables = sp.symbols(f"c0:{len(center_basis)}")
    candidate = sp.zeros(dimension)
    for variable, basis_matrix in zip(variables, center_basis, strict=True):
        candidate += variable * basis_matrix

    equations = []
    for value in (candidate * candidate - candidate).applyfunc(sp.expand):
        if value != 0 and value not in equations:
            equations.append(value)

    try:
        solutions = sp.solve(equations, variables, dict=True, simplify=False)
    except (NotImplementedError, TypeError, ValueError):
        return False, ()

    idempotents: list[ComplexCentralIdempotent] = []
    seen: set[tuple[sp.Expr, ...]] = set()
    for solution in solutions:
        if not all(variable in solution for variable in variables):
            return False, ()
        coefficients = tuple(sp.simplify(solution[variable]) for variable in variables)
        matrix = sp.zeros(dimension)
        for coefficient, basis_matrix in zip(coefficients, center_basis, strict=True):
            matrix += coefficient * basis_matrix
        matrix = matrix.applyfunc(sp.simplify)
        key = tuple(sp.simplify(value) for value in matrix)
        if key in seen:
            continue
        seen.add(key)
        idempotents.append(
            ComplexCentralIdempotent(
                expression=tuple(
                    (f"center_{index}", coefficient)
                    for index, coefficient in enumerate(coefficients)
                    if coefficient != 0
                ),
                matrix=matrix,
                rank=matrix.rank(),
            )
        )
    return True, tuple(sorted(idempotents, key=lambda item: (item.rank, tuple(item.matrix))))


def is_commutative(algebra_basis: Sequence[sp.Matrix]) -> bool:
    for left in algebra_basis:
        for right in algebra_basis:
            if not _same_matrix(left * right, right * left):
                return False
    return True


def _contains_matrix(matrices: Sequence[sp.Matrix], target: sp.Matrix) -> bool:
    return any(_same_matrix(matrix, target) for matrix in matrices)


def _target_split_present(
    profile: ComplexSplitProfile,
    idempotents: Sequence[ComplexCentralIdempotent],
) -> bool:
    matrices = tuple(idempotent.matrix for idempotent in idempotents)
    return all(_contains_matrix(matrices, projector) for projector in profile.target_projectors)


def _projectors_match(
    left_projectors: Sequence[sp.Matrix],
    right_projectors: Sequence[sp.Matrix],
) -> bool:
    if len(left_projectors) != len(right_projectors):
        return False
    return all(_contains_matrix(tuple(right_projectors), projector) for projector in left_projectors)


def canonical_split_matched(
    discovered_projectors: Sequence[sp.Matrix],
    canonical_projectors: Sequence[sp.Matrix],
) -> bool:
    return _projectors_match(discovered_projectors, canonical_projectors)


def _ordered_pair_by_target_ranks(
    pair: tuple[ComplexCentralIdempotent, ComplexCentralIdempotent],
    target_ranks: tuple[int, int],
) -> tuple[ComplexCentralIdempotent, ComplexCentralIdempotent] | None:
    left, right = pair
    if (left.rank, right.rank) == target_ranks:
        return left, right
    if (right.rank, left.rank) == target_ranks:
        return right, left
    return None


def _complementary_rank_pairs(
    idempotents: Sequence[ComplexCentralIdempotent],
    *,
    target_ranks: tuple[int, int],
    dimension: int,
) -> tuple[tuple[ComplexCentralIdempotent, ComplexCentralIdempotent], ...]:
    one = _identity(dimension)
    pairs = []
    for left in idempotents:
        if left.rank not in target_ranks:
            continue
        for right in idempotents:
            if left is right or right.rank not in target_ranks:
                continue
            ordered = _ordered_pair_by_target_ranks((left, right), target_ranks)
            if ordered is None:
                continue
            first, second = ordered
            if not _same_matrix(first.matrix + second.matrix, one):
                continue
            if not _same_matrix(first.matrix * second.matrix, _zero(dimension)):
                continue
            if not any(
                _same_matrix(first.matrix, old_first.matrix)
                and _same_matrix(second.matrix, old_second.matrix)
                for old_first, old_second in pairs
            ):
                pairs.append((first, second))
    return tuple(pairs)


def discover_complementary_rank_pair(
    idempotents: Sequence[ComplexCentralIdempotent],
    target_ranks: tuple[int, int],
    dimension: int,
) -> tuple[ComplexCentralIdempotent, ComplexCentralIdempotent] | None:
    pairs = _complementary_rank_pairs(
        idempotents,
        target_ranks=target_ranks,
        dimension=dimension,
    )
    return pairs[0] if len(pairs) == 1 else None


def split_seeded_by_layer(
    layers: Sequence[ComplexLayerInput],
    profile: ComplexSplitProfile,
) -> bool:
    if any(layer.is_seeded_split_layer for layer in layers):
        return True
    for layer in layers:
        if any(_same_matrix(layer.matrix, projector) for projector in profile.target_projectors):
            return True
    return False


def split_idempotent_policy(
    profile: ComplexSplitProfile,
    idempotents: Sequence[ComplexCentralIdempotent],
) -> str:
    zero = _zero(profile.dimension)
    one = _identity(profile.dimension)
    allowed = (zero, *profile.target_projectors, one)
    for idempotent in idempotents:
        if any(_same_matrix(idempotent.matrix, allowed_matrix) for allowed_matrix in allowed):
            continue
        if idempotent.rank == 1:
            return "rank_one_locking"
        return "forbidden_idempotent"
    return "passed"


def _split_idempotent_policy_for_projectors(
    profile: ComplexSplitProfile,
    idempotents: Sequence[ComplexCentralIdempotent],
    projectors: Sequence[sp.Matrix],
) -> str:
    zero = _zero(profile.dimension)
    one = _identity(profile.dimension)
    allowed = (zero, *projectors, one)
    for idempotent in idempotents:
        if any(_same_matrix(idempotent.matrix, allowed_matrix) for allowed_matrix in allowed):
            continue
        if idempotent.rank == 1:
            return "rank_one_locking"
        return "forbidden_idempotent"
    return "passed"


def block_invariant_chain(
    algebra_basis: Sequence[sp.Matrix],
    projectors: Sequence[sp.Matrix],
    *,
    dimension: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    block_dimensions = []
    block_commutativity = []
    for projector in projectors:
        projected = [
            (projector * algebra_matrix * projector).applyfunc(sp.simplify)
            for algebra_matrix in algebra_basis
        ]
        span = exact_matrix_span(projected, rows=dimension, cols=dimension)
        basis = tuple(span.basis_matrices)
        block_dimensions.append(span.rank)
        block_commutativity.append(is_commutative(basis))
    return tuple(block_dimensions), tuple(block_commutativity)


def _empty_result(
    profile: ComplexSplitProfile,
    *,
    verdict: ComplexSplitVerdict,
    reason: str,
    generated_algebra_dimension: int = 0,
    generated_algebra_closed: bool = False,
    center_dimension: int = 0,
    central_idempotents: tuple[ComplexCentralIdempotent, ...] = (),
    target_split_present: bool = False,
    seeded_split_present: bool = False,
    forbidden_rank_one_count: int = 0,
    forbidden_idempotent_count: int = 0,
    block_dimensions: tuple[int, ...] = (),
    block_commutativity: tuple[bool, ...] = (),
    discovered_projectors: tuple[sp.Matrix, ...] = (),
    canonical_split_matched: bool = False,
    discovered_projector_ranks: tuple[int, ...] = (),
) -> ComplexSplitResult:
    return ComplexSplitResult(
        profile_name=profile.name,
        verdict=verdict,
        reason=reason,
        generated_algebra_dimension=generated_algebra_dimension,
        generated_algebra_closed=generated_algebra_closed,
        center_dimension=center_dimension,
        central_idempotents=central_idempotents,
        target_split_present=target_split_present,
        seeded_split_present=seeded_split_present,
        forbidden_rank_one_count=forbidden_rank_one_count,
        forbidden_idempotent_count=forbidden_idempotent_count,
        block_dimensions=block_dimensions,
        block_commutativity=block_commutativity,
        discovered_projectors=discovered_projectors,
        canonical_split_matched=canonical_split_matched,
        discovered_projector_ranks=discovered_projector_ranks,
    )


def rule_to_complex_split_verdict(
    layers: Sequence[ComplexLayerInput],
    profile: ComplexSplitProfile,
) -> ComplexSplitResult:
    seeded_split = split_seeded_by_layer(layers, profile)
    closure = generated_complex_algebra_closure(
        tuple(layer.matrix for layer in layers),
        dimension=profile.dimension,
        max_dimension=profile.max_algebra_dimension,
    )
    if not closure.closed:
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.NOT_SOLVED,
            reason="algebra_not_closed",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=False,
            seeded_split_present=seeded_split,
        )

    center_basis = complex_center_basis(closure.basis, dimension=profile.dimension)
    if len(center_basis) > profile.max_center_dimension:
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.NOT_SOLVED,
            reason="center_too_large",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            seeded_split_present=seeded_split,
        )

    idempotents_solved, idempotents = solve_complex_central_idempotents(
        center_basis,
        dimension=profile.dimension,
        max_center_dimension=profile.max_center_dimension,
    )
    if not idempotents_solved:
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.NOT_SOLVED,
            reason="idempotents_not_solved",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            seeded_split_present=seeded_split,
        )

    discovered_projectors: tuple[sp.Matrix, ...] = ()
    discovered_projector_ranks: tuple[int, ...] = ()
    canonical_match = False
    active_projectors = profile.target_projectors
    target_present = _target_split_present(profile, idempotents)
    if profile.allow_discovered_split:
        if profile.target_ranks is None:
            return _empty_result(
                profile,
                verdict=ComplexSplitVerdict.NOT_SOLVED,
                reason="missing_target_ranks",
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=True,
                center_dimension=len(center_basis),
                central_idempotents=idempotents,
                seeded_split_present=seeded_split,
            )
        rank_one_nontrivial = [
            idempotent
            for idempotent in idempotents
            if idempotent.rank == 1 and idempotent.rank not in (0, profile.dimension)
        ]
        if rank_one_nontrivial:
            return _empty_result(
                profile,
                verdict=ComplexSplitVerdict.FALSIFIED_RANK_ONE_LOCKING,
                reason="rank_one_locking",
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=True,
                center_dimension=len(center_basis),
                central_idempotents=idempotents,
                target_split_present=True,
                seeded_split_present=seeded_split,
                forbidden_rank_one_count=len(rank_one_nontrivial),
                forbidden_idempotent_count=1,
            )
        discovered_pairs = _complementary_rank_pairs(
            idempotents,
            target_ranks=profile.target_ranks,
            dimension=profile.dimension,
        )
        if len(discovered_pairs) > 1:
            return _empty_result(
                profile,
                verdict=ComplexSplitVerdict.NOT_SOLVED,
                reason="multiple_discovered_splits",
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=True,
                center_dimension=len(center_basis),
                central_idempotents=idempotents,
                seeded_split_present=seeded_split,
            )
        if not discovered_pairs:
            target_present = False
        else:
            active_idempotents = discovered_pairs[0]
            active_projectors = tuple(item.matrix for item in active_idempotents)
            discovered_projectors = active_projectors
            discovered_projector_ranks = tuple(item.rank for item in active_idempotents)
            canonical_match = canonical_split_matched(active_projectors, profile.target_projectors)
            target_present = True
    if not target_present:
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_NO_SPLIT,
            reason="target_split_missing",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=False,
            seeded_split_present=seeded_split,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )

    if profile.allow_discovered_split:
        policy = _split_idempotent_policy_for_projectors(profile, idempotents, active_projectors)
    else:
        policy = (profile.idempotent_policy or split_idempotent_policy)(profile, idempotents)
    forbidden_rank_one_count = sum(
        1
        for idempotent in idempotents
        if idempotent.rank == 1
        and not any(_same_matrix(idempotent.matrix, projector) for projector in active_projectors)
        and idempotent.rank not in (0, profile.dimension)
    )
    forbidden_idempotent_count = 0 if policy == "passed" else 1
    if policy == "rank_one_locking":
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_RANK_ONE_LOCKING,
            reason="rank_one_locking",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=seeded_split,
            forbidden_rank_one_count=max(1, forbidden_rank_one_count),
            forbidden_idempotent_count=forbidden_idempotent_count,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )
    if policy != "passed":
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_FORBIDDEN_IDEMPOTENT,
            reason=policy,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=seeded_split,
            forbidden_rank_one_count=forbidden_rank_one_count,
            forbidden_idempotent_count=forbidden_idempotent_count,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )

    block_dimensions, block_commutativity = block_invariant_chain(
        closure.basis,
        active_projectors,
        dimension=profile.dimension,
    )
    if profile.expected_block_dimensions is not None and (
        block_dimensions != profile.expected_block_dimensions
    ):
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_FORBIDDEN_IDEMPOTENT,
            reason="block_dimensions_mismatch",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=seeded_split,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )
    if profile.expected_block_commutativity is not None and (
        block_commutativity != profile.expected_block_commutativity
    ):
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_FORBIDDEN_IDEMPOTENT,
            reason="block_commutativity_mismatch",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=seeded_split,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )
    if profile.require_noncommutative and is_commutative(closure.basis):
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.FALSIFIED_COMMUTATIVE,
            reason="commutative",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=seeded_split,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )
    if seeded_split:
        return _empty_result(
            profile,
            verdict=ComplexSplitVerdict.SEEDED_SPLIT_CONTROL,
            reason="seeded_split",
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=True,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            target_split_present=True,
            seeded_split_present=True,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            discovered_projectors=discovered_projectors,
            canonical_split_matched=canonical_match,
            discovered_projector_ranks=discovered_projector_ranks,
        )
    return _empty_result(
        profile,
        verdict=ComplexSplitVerdict.SPLIT_CANDIDATE,
        reason="passed",
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=True,
        center_dimension=len(center_basis),
        central_idempotents=idempotents,
        target_split_present=True,
        seeded_split_present=False,
        block_dimensions=block_dimensions,
        block_commutativity=block_commutativity,
        discovered_projectors=discovered_projectors,
        canonical_split_matched=canonical_match,
        discovered_projector_ranks=discovered_projector_ranks,
    )


def complex_result_to_dict(result: ComplexSplitResult) -> dict[str, object]:
    return {
        "profile_name": result.profile_name,
        "verdict": result.verdict.value,
        "reason": result.reason,
        "generated_algebra_dimension": result.generated_algebra_dimension,
        "generated_algebra_closed": result.generated_algebra_closed,
        "center_dimension": result.center_dimension,
        "central_idempotent_ranks": tuple(
            sorted(idempotent.rank for idempotent in result.central_idempotents)
        ),
        "target_split_present": result.target_split_present,
        "seeded_split_present": result.seeded_split_present,
        "forbidden_rank_one_count": result.forbidden_rank_one_count,
        "forbidden_idempotent_count": result.forbidden_idempotent_count,
        "block_dimensions": result.block_dimensions,
        "block_commutativity": result.block_commutativity,
        "canonical_split_matched": result.canonical_split_matched,
        "discovered_projector_ranks": result.discovered_projector_ranks,
    }
