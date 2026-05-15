"""Block-preserving noncommuting Floquet-alpha twist family."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaCandidate,
    floquet_alpha_candidates,
    floquet_alpha_canonical_j,
    floquet_alpha_layer,
    floquet_alpha_operator,
    floquet_alpha_spectral_projectors,
)
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    RuleToVerdictResult,
    _complementary_rank_6_4_pairs,
    _lower_rank_idempotents_inside_pairs,
    center_basis_of_algebra,
    centralizer_basis,
    generated_algebra_basis,
    rule_to_verdict,
    solve_central_idempotents,
    solve_complex_structures_in_basis,
)


RouteLabel = str


@dataclass(frozen=True)
class FloquetAlphaNoncommutingCandidate:
    pattern: FloquetAlphaCandidate
    orientation_signs: tuple[int, ...]
    mode_mapping: tuple[tuple[int, int], ...] | None = None

    @property
    def name(self) -> str:
        signs = "_".join("p" if sign > 0 else "m" for sign in self.orientation_signs)
        if self.mode_mapping is None:
            return f"{self.pattern.name}_noncommuting_signed_twist_{signs}"
        mapping = "_".join(f"{source}{target}" for source, target in self.mode_mapping)
        return f"{self.pattern.name}_noncommuting_signed_twist_{signs}_perm_{mapping}"


@dataclass(frozen=True)
class FloquetAlphaNoncommutingCertificate:
    candidate_name: str
    base_candidate_name: str
    twist_layer_name: str
    orientation_signs: tuple[int, ...]
    mode_mapping: tuple[tuple[int, int], ...]
    u1_u2_commute: bool
    u2_real_orthogonal: bool
    u2_preserves_alpha_projector: bool
    u2_preserves_eta_projector: bool
    alpha_orientation_nonconstant: bool
    eta_orientation_nonconstant: bool
    generated_algebra_dimension: int
    center_dimension: int
    center_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    compatible_centralizer_dimension: int
    compatible_j_solved: bool
    compatible_j_moduli_dimension: int | None
    compatible_complex_structure_count: int
    generated_j_solved: bool
    generated_j_moduli_dimension: int | None
    generated_complex_structure_count: int
    local_compatible_operator_dimension: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    forced_j_found: bool
    pass_strict_rule_to_bridge: bool
    rule_verdict: str
    route_label: RouteLabel
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaNoncommutingJDiagnostic:
    index: int
    expression: tuple[tuple[str, sp.Expr], ...]
    pair_orientation_signs: tuple[int, ...]
    in_generated_algebra: bool
    in_rule_local_center: bool
    equals_spectral_polarization_j: bool
    equals_negative_spectral_polarization_j: bool
    commutes_with_u1: bool
    commutes_with_u2: bool
    squares_to_minus_identity: bool
    orthogonal: bool
    matrix: sp.Matrix


@dataclass(frozen=True)
class FloquetAlphaNoncommutingJGapCertificate:
    candidate_name: str
    compatible_j_count: int
    generated_algebra_dimension: int
    center_dimension: int
    compatible_centralizer_dimension: int
    compatible_j_solved: bool
    compatible_j_moduli_dimension: int | None
    generated_j_solved: bool
    generated_complex_structure_count: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    compatible_j_in_generated_algebra_count: int
    compatible_j_in_rule_local_center_count: int
    spectral_polarization_j_matched_count: int
    forced_j_found: bool
    reason_for_forced_j_failure: str
    compatible_j_diagnostics: tuple[FloquetAlphaNoncommutingJDiagnostic, ...]
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaNoncommutingCompletionCertificate:
    candidate_name: str
    completed_j_index: int
    completed_j_pair_orientation_signs: tuple[int, ...]
    w_in_previous_generated_algebra: bool
    w_in_completed_generated_algebra: bool
    w_in_completed_center: bool
    w_commutes_with_u1: bool
    w_commutes_with_u2: bool
    w_squares_to_minus_identity: bool
    w_orthogonal: bool
    generated_algebra_dimension: int
    center_dimension: int
    center_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    compatible_centralizer_dimension: int
    compatible_j_solved: bool
    compatible_j_moduli_dimension: int | None
    compatible_complex_structure_count: int
    local_compatible_operator_dimension: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    declared_w_is_local_compatible_j: bool
    strict_unique_j_found: bool
    pass_completion_to_bridge: bool
    completion_label: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaNoncommutingExhaustiveSummary:
    candidate_count: int
    evaluated_symmetry_classes: int
    sign_patterns_per_alpha_pattern: int
    permutation_choices_per_alpha_pattern: int
    noncommuting_candidates: int
    commuting_candidates: int
    oversized_algebra_rejections: int
    generated_algebra_dimension_counts: tuple[tuple[int, int], ...]
    compatible_j_count_distribution: tuple[tuple[int, int], ...]
    minimal_four_j_candidates: int
    no_locking_shape_candidates: int
    compatible_j_in_generated_algebra_candidates: int
    compatible_j_in_generated_algebra_total: int
    minimal_four_j_in_generated_algebra_candidates: int
    no_locking_shape_j_in_generated_algebra_candidates: int
    bridge_candidate_count: int
    hit_candidate_names: tuple[str, ...]
    route_label: str
    load_bearing_qca_bridge: bool = False


Qsqrt3Entry = tuple[sp.Rational, sp.Rational]
Qsqrt3Matrix = tuple[Qsqrt3Entry, ...]


def _mode_count(dimension: int) -> int:
    if dimension % 2:
        raise ValueError("real carrier dimension must be even")
    return dimension // 2


def _validate_signs(signs: tuple[int, ...], *, dimension: int = 10) -> None:
    if len(signs) != _mode_count(dimension):
        raise ValueError("orientation sign count must match the number of mode-pairs")
    if any(sign not in {-1, 1} for sign in signs):
        raise ValueError("orientation signs must be +/-1")


def floquet_alpha_noncommuting_mode_mapping(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> dict[int, int]:
    """Return the block-preserving cycle/swap support of the signed twist."""

    if candidate.mode_mapping is not None:
        return dict(candidate.mode_mapping)

    alpha_modes = candidate.pattern.alpha_modes
    eta_modes = candidate.pattern.eta_modes
    mode_mapping: dict[int, int] = {}
    for source, target in zip(alpha_modes, (*alpha_modes[1:], alpha_modes[0]), strict=True):
        mode_mapping[source] = target
    for source, target in zip(eta_modes, reversed(eta_modes), strict=True):
        mode_mapping[source] = target
    return mode_mapping


def signed_orientation_twist_operator(
    mode_mapping: dict[int, int],
    orientation_signs: tuple[int, ...],
    *,
    dimension: int = 10,
) -> sp.Matrix:
    """Return a signed pair permutation preserving a hidden orientation.

    A mode edge with equal endpoint signs is complex-linear with respect to the
    standard pair orientation.  A sign-changing edge is complex-antilinear:
    ``(x, y) -> (x, -y)``.  The resulting matrix is still real orthogonal and
    block-preserving when the mode permutation preserves the alpha/eta sectors.
    """

    _validate_signs(orientation_signs, dimension=dimension)
    mode_count = _mode_count(dimension)
    if set(mode_mapping) != set(range(mode_count)):
        raise ValueError("mode mapping must be a full mode-pair permutation")
    if set(mode_mapping.values()) != set(range(mode_count)):
        raise ValueError("mode mapping must be a full mode-pair permutation")

    matrix = sp.zeros(dimension)
    for source, target in mode_mapping.items():
        matrix[target, source] = 1
        y_sign = 1 if orientation_signs[source] == orientation_signs[target] else -1
        matrix[target + mode_count, source + mode_count] = y_sign
    return matrix


def floquet_alpha_noncommuting_twist_operator(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> sp.Matrix:
    return signed_orientation_twist_operator(
        floquet_alpha_noncommuting_mode_mapping(candidate),
        candidate.orientation_signs,
    )


def floquet_alpha_noncommuting_twist_layer(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> RuleLayerInput:
    return RuleLayerInput(
        name=f"{candidate.name}_layer",
        matrix=floquet_alpha_noncommuting_twist_operator(candidate),
        support=(0,),
        locality_radius=0,
    )


def floquet_alpha_noncommuting_candidates(
    *,
    pattern_index: int | None = None,
) -> tuple[FloquetAlphaNoncommutingCandidate, ...]:
    """Return the first exact block-preserving noncommuting twist per pattern."""

    candidates: list[FloquetAlphaNoncommutingCandidate] = []
    for pattern in floquet_alpha_candidates():
        if pattern_index is not None and pattern.pattern_index != pattern_index:
            continue
        signs = [1] * 5
        signs[pattern.alpha_modes[-1]] = -1
        signs[pattern.eta_modes[-1]] = -1
        candidates.append(
            FloquetAlphaNoncommutingCandidate(
                pattern=pattern,
                orientation_signs=tuple(signs),
            )
        )
    return tuple(candidates)


def _block_preserving_pair_permutation_mappings(
    pattern: FloquetAlphaCandidate,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    mappings = []
    for alpha_targets in permutations(pattern.alpha_modes):
        alpha_mapping = tuple(zip(pattern.alpha_modes, alpha_targets, strict=True))
        for eta_targets in permutations(pattern.eta_modes):
            eta_mapping = tuple(zip(pattern.eta_modes, eta_targets, strict=True))
            mappings.append(tuple(sorted((*alpha_mapping, *eta_mapping))))
    return tuple(mappings)


def floquet_alpha_noncommuting_exhaustive_candidates(
    *,
    pattern_index: int | None = None,
) -> tuple[FloquetAlphaNoncommutingCandidate, ...]:
    """Enumerate all signed block-preserving pair-permutation twists.

    This is the finite Route-1 discrete class: ten Floquet-alpha resonance
    patterns, all ``2^5`` hidden pair-orientation sign choices, and all
    block-preserving pair permutations ``S_3 x S_2``.
    """

    candidates: list[FloquetAlphaNoncommutingCandidate] = []
    for pattern in floquet_alpha_candidates():
        if pattern_index is not None and pattern.pattern_index != pattern_index:
            continue
        for mode_mapping in _block_preserving_pair_permutation_mappings(pattern):
            for signs in product((-1, 1), repeat=5):
                candidates.append(
                    FloquetAlphaNoncommutingCandidate(
                        pattern=pattern,
                        orientation_signs=tuple(signs),
                        mode_mapping=mode_mapping,
                    )
                )
    return tuple(candidates)


def floquet_alpha_noncommuting_rule_to_verdict(
    candidate: FloquetAlphaNoncommutingCandidate,
    *,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (
            floquet_alpha_layer(candidate.pattern),
            floquet_alpha_noncommuting_twist_layer(candidate),
        ),
        rule_name=candidate.name,
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def _orientation_nonconstant(
    signs: tuple[int, ...],
    modes: tuple[int, ...],
) -> bool:
    return len({signs[mode] for mode in modes}) > 1


def _matrix_in_span(matrix: sp.Matrix, basis: tuple[sp.Matrix, ...]) -> bool:
    return matrix_span_rank((*basis, matrix)) == matrix_span_rank(basis)


def _qzero() -> Qsqrt3Entry:
    return sp.Rational(0), sp.Rational(0)


def _qone() -> Qsqrt3Entry:
    return sp.Rational(1), sp.Rational(0)


def _qadd(left: Qsqrt3Entry, right: Qsqrt3Entry) -> Qsqrt3Entry:
    return left[0] + right[0], left[1] + right[1]


def _qsub(left: Qsqrt3Entry, right: Qsqrt3Entry) -> Qsqrt3Entry:
    return left[0] - right[0], left[1] - right[1]


def _qmul(left: Qsqrt3Entry, right: Qsqrt3Entry) -> Qsqrt3Entry:
    return left[0] * right[0] + 3 * left[1] * right[1], left[0] * right[1] + left[1] * right[0]


def _qinv(value: Qsqrt3Entry) -> Qsqrt3Entry:
    denominator = value[0] * value[0] - 3 * value[1] * value[1]
    return value[0] / denominator, -value[1] / denominator


def _qis_zero(value: Qsqrt3Entry) -> bool:
    return value == _qzero()


def _qsqrt3_entry(value: sp.Expr) -> Qsqrt3Entry:
    sqrt3 = sp.sqrt(3)
    expanded = sp.expand(value)
    sqrt_part = sp.Rational(sp.expand(expanded).coeff(sqrt3))
    rational_part = sp.Rational(sp.simplify(expanded - sqrt_part * sqrt3))
    return rational_part, sqrt_part


def _qmatrix_from_sympy(matrix: sp.Matrix) -> Qsqrt3Matrix:
    return tuple(_qsqrt3_entry(value) for value in matrix)


def _qidentity(*, dimension: int = 10) -> Qsqrt3Matrix:
    entries = []
    for row in range(dimension):
        for column in range(dimension):
            entries.append(_qone() if row == column else _qzero())
    return tuple(entries)


def _qmatrix_mul(
    left: Qsqrt3Matrix,
    right: Qsqrt3Matrix,
    *,
    dimension: int = 10,
) -> Qsqrt3Matrix:
    entries = []
    for row in range(dimension):
        for column in range(dimension):
            value = _qzero()
            for index in range(dimension):
                value = _qadd(
                    value,
                    _qmul(left[row * dimension + index], right[index * dimension + column]),
                )
            entries.append(value)
    return tuple(entries)


def _qmatrix_rank(vectors: tuple[Qsqrt3Matrix, ...]) -> int:
    if not vectors:
        return 0
    rows = [list(row) for row in zip(*vectors, strict=True)]
    row_count = len(rows)
    column_count = len(vectors)
    rank = 0
    for column in range(column_count):
        pivot = None
        for row in range(rank, row_count):
            if not _qis_zero(rows[row][column]):
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = _qinv(rows[rank][column])
        for active_column in range(column, column_count):
            rows[rank][active_column] = _qmul(rows[rank][active_column], inverse)
        for row in range(row_count):
            if row == rank or _qis_zero(rows[row][column]):
                continue
            factor = rows[row][column]
            for active_column in range(column, column_count):
                rows[row][active_column] = _qsub(
                    rows[row][active_column],
                    _qmul(factor, rows[rank][active_column]),
                )
        rank += 1
        if rank == row_count:
            break
    return rank


def _qmatrix_in_span(matrix: Qsqrt3Matrix, basis: tuple[Qsqrt3Matrix, ...]) -> bool:
    return _qmatrix_rank((*basis, matrix)) == _qmatrix_rank(basis)


def _append_qindependent(
    basis: list[Qsqrt3Matrix],
    candidate: Qsqrt3Matrix,
) -> bool:
    if candidate in basis:
        return False
    if _qmatrix_rank((*basis, candidate)) > len(basis):
        basis.append(candidate)
        return True
    return False


def _qgenerated_algebra_basis(
    generators: tuple[Qsqrt3Matrix, ...],
    *,
    max_dimension: int = 100,
) -> tuple[bool, tuple[Qsqrt3Matrix, ...]]:
    basis: list[Qsqrt3Matrix] = []
    for matrix in (_qidentity(), *generators):
        _append_qindependent(basis, matrix)
    changed = True
    while changed:
        changed = False
        current = tuple(basis)
        for left, right in product(current, current):
            changed = _append_qindependent(basis, _qmatrix_mul(left, right)) or changed
            if len(basis) > max_dimension:
                return False, tuple(basis)
    return True, tuple(basis)


def _pair_orientation_signs(matrix: sp.Matrix, *, dimension: int = 10) -> tuple[int, ...]:
    mode_count = _mode_count(dimension)
    signs: list[int] = []
    for mode in range(mode_count):
        top_right = sp.simplify(matrix[mode, mode + mode_count])
        bottom_left = sp.simplify(matrix[mode + mode_count, mode])
        if top_right == -1 and bottom_left == 1:
            signs.append(1)
        elif top_right == 1 and bottom_left == -1:
            signs.append(-1)
        else:
            return ()
    return tuple(signs)


def pair_orientation_j_operator(
    orientation_signs: tuple[int, ...],
    *,
    dimension: int = 10,
) -> sp.Matrix:
    """Return the local pair-orientation complex structure for the signs."""

    _validate_signs(orientation_signs, dimension=dimension)
    mode_count = _mode_count(dimension)
    matrix = sp.zeros(dimension)
    for mode, sign in enumerate(orientation_signs):
        matrix[mode, mode + mode_count] = -sign
        matrix[mode + mode_count, mode] = sign
    return matrix


def floquet_alpha_noncommuting_completion_j_signs(
    candidate: FloquetAlphaNoncommutingCandidate,
    *,
    j_index: int = 0,
) -> tuple[int, ...]:
    alpha_flip, eta_flip = ((1, 1), (1, -1), (-1, 1), (-1, -1))[j_index]
    signs = list(candidate.orientation_signs)
    for mode in candidate.pattern.alpha_modes:
        signs[mode] *= alpha_flip
    for mode in candidate.pattern.eta_modes:
        signs[mode] *= eta_flip
    return tuple(signs)


def _compatible_pair_orientation_sign_choices(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> tuple[tuple[int, ...], ...]:
    mapping = floquet_alpha_noncommuting_mode_mapping(candidate)
    edge_signs = {
        source: (
            1
            if candidate.orientation_signs[source] == candidate.orientation_signs[target]
            else -1
        )
        for source, target in mapping.items()
    }
    base = [0] * _mode_count(10)
    components: list[tuple[int, ...]] = []
    for start in range(_mode_count(10)):
        if base[start]:
            continue
        component = []
        base[start] = 1
        current = start
        while current not in component:
            component.append(current)
            target = mapping[current]
            propagated = edge_signs[current] * base[current]
            if base[target] and base[target] != propagated:
                raise ValueError("inconsistent signed permutation orientation")
            base[target] = propagated
            current = target
        components.append(tuple(component))

    choices = []
    for flips in product((-1, 1), repeat=len(components)):
        signs = list(base)
        for flip, component in zip(flips, components, strict=True):
            for mode in component:
                signs[mode] *= flip
        choices.append(tuple(signs))
    return tuple(sorted(set(choices)))


def _relative_exhaustive_key(candidate: FloquetAlphaNoncommutingCandidate) -> tuple[object, ...]:
    ordered_modes = (*candidate.pattern.alpha_modes, *candidate.pattern.eta_modes)
    index_by_mode = {mode: index for index, mode in enumerate(ordered_modes)}
    mapping = floquet_alpha_noncommuting_mode_mapping(candidate)
    relative_mapping = tuple(index_by_mode[mapping[mode]] for mode in ordered_modes)
    signs = tuple(candidate.orientation_signs[mode] for mode in ordered_modes)
    alpha_anchor = signs[0]
    eta_anchor = signs[len(candidate.pattern.alpha_modes)]
    relative_signs = tuple(
        sign * (alpha_anchor if index < len(candidate.pattern.alpha_modes) else eta_anchor)
        for index, sign in enumerate(signs)
    )
    return relative_mapping, relative_signs


def _exhaustive_candidate_generated_j_count(
    candidate: FloquetAlphaNoncommutingCandidate,
    *,
    max_algebra_dimension: int,
) -> tuple[bool, int, int, int]:
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    if is_zero_matrix(commutator(u1, u2)):
        return True, 0, 0, 0

    solved, basis = _qgenerated_algebra_basis(
        (_qmatrix_from_sympy(u1), _qmatrix_from_sympy(u2)),
        max_dimension=max_algebra_dimension,
    )
    if not solved:
        return False, len(basis), 0, 0

    compatible_signs = _compatible_pair_orientation_sign_choices(candidate)
    compatible_j_count = len(compatible_signs)
    generated_j_count = 0
    for signs in compatible_signs:
        matrix = _qmatrix_from_sympy(pair_orientation_j_operator(signs))
        if _qmatrix_in_span(matrix, basis):
            generated_j_count += 1
    return True, len(basis), compatible_j_count, generated_j_count


def _is_single_cycle(mapping: dict[int, int], modes: tuple[int, ...]) -> bool:
    if not modes:
        return False
    visited = []
    current = modes[0]
    while current not in visited:
        visited.append(current)
        current = mapping[current]
    return current == modes[0] and set(visited) == set(modes)


def _has_no_locking_shape(candidate: FloquetAlphaNoncommutingCandidate) -> bool:
    mapping = floquet_alpha_noncommuting_mode_mapping(candidate)
    return (
        _is_single_cycle(mapping, candidate.pattern.alpha_modes)
        and _is_single_cycle(mapping, candidate.pattern.eta_modes)
        and _orientation_nonconstant(candidate.orientation_signs, candidate.pattern.alpha_modes)
        and _orientation_nonconstant(candidate.orientation_signs, candidate.pattern.eta_modes)
    )


def floquet_alpha_noncommuting_exhaustive_summary(
    *,
    pattern_index: int | None = None,
    max_algebra_dimension: int = 100,
) -> FloquetAlphaNoncommutingExhaustiveSummary:
    candidates = floquet_alpha_noncommuting_exhaustive_candidates(pattern_index=pattern_index)
    cache: dict[tuple[object, ...], tuple[bool, int, int, int]] = {}
    dimension_counts: dict[int, int] = {}
    compatible_j_counts: dict[int, int] = {}
    noncommuting = 0
    commuting = 0
    oversized = 0
    minimal_four_j_candidates = 0
    no_locking_shape_candidates = 0
    generated_j_hit_candidates = 0
    generated_j_hit_total = 0
    minimal_four_j_hits = 0
    no_locking_shape_hits = 0
    hit_names: list[str] = []

    for candidate in candidates:
        key = _relative_exhaustive_key(candidate)
        result = cache.get(key)
        if result is None:
            result = _exhaustive_candidate_generated_j_count(
                candidate,
                max_algebra_dimension=max_algebra_dimension,
            )
            cache[key] = result
        solved, algebra_dimension, compatible_j_count, generated_j_count = result
        if algebra_dimension == 0:
            commuting += 1
            continue
        noncommuting += 1
        if not solved:
            oversized += 1
            continue
        dimension_counts[algebra_dimension] = dimension_counts.get(algebra_dimension, 0) + 1
        compatible_j_counts[compatible_j_count] = compatible_j_counts.get(compatible_j_count, 0) + 1
        if compatible_j_count == 4:
            minimal_four_j_candidates += 1
        no_locking_shape = _has_no_locking_shape(candidate)
        if no_locking_shape:
            no_locking_shape_candidates += 1
        if compatible_j_count and generated_j_count:
            generated_j_hit_candidates += 1
            generated_j_hit_total += generated_j_count
            if compatible_j_count == 4:
                minimal_four_j_hits += 1
            if no_locking_shape:
                no_locking_shape_hits += 1
            if len(hit_names) < 10:
                hit_names.append(candidate.name)

    bridge_candidates = 0
    if no_locking_shape_hits:
        route_label = "discrete_signed_twist_no_locking_shape_generated_j_hit"
    elif generated_j_hit_candidates:
        route_label = "discrete_signed_twist_generated_j_hits_fail_no_locking_shape"
    elif oversized:
        route_label = "discrete_signed_twist_oversized_algebra_rejections"
    else:
        route_label = "discrete_signed_twist_no_generated_compatible_j"

    return FloquetAlphaNoncommutingExhaustiveSummary(
        candidate_count=len(candidates),
        evaluated_symmetry_classes=len(cache),
        sign_patterns_per_alpha_pattern=32,
        permutation_choices_per_alpha_pattern=12,
        noncommuting_candidates=noncommuting,
        commuting_candidates=commuting,
        oversized_algebra_rejections=oversized,
        generated_algebra_dimension_counts=tuple(sorted(dimension_counts.items())),
        compatible_j_count_distribution=tuple(sorted(compatible_j_counts.items())),
        minimal_four_j_candidates=minimal_four_j_candidates,
        no_locking_shape_candidates=no_locking_shape_candidates,
        compatible_j_in_generated_algebra_candidates=generated_j_hit_candidates,
        compatible_j_in_generated_algebra_total=generated_j_hit_total,
        minimal_four_j_in_generated_algebra_candidates=minimal_four_j_hits,
        no_locking_shape_j_in_generated_algebra_candidates=no_locking_shape_hits,
        bridge_candidate_count=bridge_candidates,
        hit_candidate_names=tuple(hit_names),
        route_label=route_label,
        load_bearing_qca_bridge=bool(bridge_candidates),
    )


def _route_label(
    result: RuleToVerdictResult,
    *,
    preserves_alpha: bool,
    preserves_eta: bool,
    commutes: bool,
) -> RouteLabel:
    if result.pass_rule_to_bridge:
        return "bridge_candidate"
    if commutes:
        return "commuting_not_route_1"
    if not preserves_alpha or not preserves_eta:
        return "coarse_split_not_preserved"
    if not result.center_solved:
        return "center_not_solved"
    if not result.complementary_rank_6_4_pairs:
        return "coarse_center_missing"
    if result.lower_rank_central_idempotents:
        return "lower_rank_center_generated"
    if (
        result.compatible_j_solved
        and result.compatible_j_moduli_dimension == 0
        and result.compatible_complex_structures
    ):
        return "coarse_center_preserved_compatible_j_not_rule_generated"
    return "coarse_center_preserved_j_not_forced"


def _j_gap_failure_reason(
    result: RuleToVerdictResult,
    diagnostics: tuple[FloquetAlphaNoncommutingJDiagnostic, ...],
) -> str:
    if result.forced_j_found:
        return "forced_j_found"
    if not result.compatible_j_solved:
        return "compatible_j_not_solved"
    if result.compatible_j_moduli_dimension != 0:
        return "compatible_j_not_zero_dimensional"
    if not diagnostics:
        return "no_compatible_j_candidates"
    if not any(item.in_generated_algebra for item in diagnostics) and not any(
        item.in_rule_local_center for item in diagnostics
    ):
        return "compatible_j_finite_but_not_generated_or_rule_local"
    if not result.generated_j_solved:
        return "generated_j_not_solved"
    if not result.local_compatible_j_solved:
        return "rule_local_j_not_solved"
    return "compatible_j_not_matched_by_forced_pair"


def floquet_alpha_noncommuting_certificate(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> FloquetAlphaNoncommutingCertificate:
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate.pattern)
    result = floquet_alpha_noncommuting_rule_to_verdict(candidate)
    one = identity(10)
    zero = sp.zeros(10)
    commutes = is_zero_matrix(commutator(u1, u2))
    preserves_alpha = sp.simplify(u2 * alpha_projector * u2.T - alpha_projector) == zero
    preserves_eta = sp.simplify(u2 * eta_projector * u2.T - eta_projector) == zero

    return FloquetAlphaNoncommutingCertificate(
        candidate_name=candidate.name,
        base_candidate_name=candidate.pattern.name,
        twist_layer_name=floquet_alpha_noncommuting_twist_layer(candidate).name,
        orientation_signs=candidate.orientation_signs,
        mode_mapping=tuple(sorted(floquet_alpha_noncommuting_mode_mapping(candidate).items())),
        u1_u2_commute=commutes,
        u2_real_orthogonal=u2.T * u2 == one,
        u2_preserves_alpha_projector=preserves_alpha,
        u2_preserves_eta_projector=preserves_eta,
        alpha_orientation_nonconstant=_orientation_nonconstant(
            candidate.orientation_signs,
            candidate.pattern.alpha_modes,
        ),
        eta_orientation_nonconstant=_orientation_nonconstant(
            candidate.orientation_signs,
            candidate.pattern.eta_modes,
        ),
        generated_algebra_dimension=result.generated_algebra_dimension,
        center_dimension=result.center_dimension,
        center_solved=result.center_solved,
        central_idempotent_ranks=tuple(item.rank for item in result.central_idempotents),
        complementary_rank_6_4_pairs=result.complementary_rank_6_4_pairs,
        lower_rank_central_idempotents=len(result.lower_rank_central_idempotents),
        compatible_centralizer_dimension=result.compatible_centralizer_dimension,
        compatible_j_solved=result.compatible_j_solved,
        compatible_j_moduli_dimension=result.compatible_j_moduli_dimension,
        compatible_complex_structure_count=len(result.compatible_complex_structures),
        generated_j_solved=result.generated_j_solved,
        generated_j_moduli_dimension=result.generated_j_moduli_dimension,
        generated_complex_structure_count=len(result.generated_complex_structures),
        local_compatible_operator_dimension=result.local_compatible_operator_dimension,
        local_compatible_j_solved=result.local_compatible_j_solved,
        local_compatible_j_moduli_dimension=result.local_compatible_j_moduli_dimension,
        local_compatible_complex_structure_count=len(result.local_compatible_complex_structures),
        forced_j_found=result.forced_j_found,
        pass_strict_rule_to_bridge=result.pass_rule_to_bridge,
        rule_verdict=result.verdict,
        route_label=_route_label(
            result,
            preserves_alpha=preserves_alpha,
            preserves_eta=preserves_eta,
            commutes=commutes,
        ),
    )


def floquet_alpha_noncommuting_j_gap_certificate(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> FloquetAlphaNoncommutingJGapCertificate:
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    result = floquet_alpha_noncommuting_rule_to_verdict(candidate)
    generated_basis = generated_algebra_basis((u1, u2))
    local_center_basis = center_basis_of_algebra(generated_basis)
    spectral_j = floquet_alpha_canonical_j(candidate.pattern)
    zero = sp.zeros(10)
    one = identity(10)

    diagnostics: list[FloquetAlphaNoncommutingJDiagnostic] = []
    for index, compatible_j in enumerate(result.compatible_complex_structures):
        matrix = compatible_j.matrix
        diagnostics.append(
            FloquetAlphaNoncommutingJDiagnostic(
                index=index,
                expression=compatible_j.expression,
                pair_orientation_signs=_pair_orientation_signs(matrix),
                in_generated_algebra=_matrix_in_span(matrix, generated_basis),
                in_rule_local_center=_matrix_in_span(matrix, local_center_basis),
                equals_spectral_polarization_j=sp.simplify(matrix - spectral_j) == zero,
                equals_negative_spectral_polarization_j=(
                    sp.simplify(matrix + spectral_j) == zero
                ),
                commutes_with_u1=is_zero_matrix(commutator(matrix, u1)),
                commutes_with_u2=is_zero_matrix(commutator(matrix, u2)),
                squares_to_minus_identity=sp.simplify(matrix * matrix + one) == zero,
                orthogonal=sp.simplify(matrix.T * matrix - one) == zero,
                matrix=matrix,
            )
        )

    diagnostic_tuple = tuple(diagnostics)
    return FloquetAlphaNoncommutingJGapCertificate(
        candidate_name=candidate.name,
        compatible_j_count=len(result.compatible_complex_structures),
        generated_algebra_dimension=result.generated_algebra_dimension,
        center_dimension=result.center_dimension,
        compatible_centralizer_dimension=result.compatible_centralizer_dimension,
        compatible_j_solved=result.compatible_j_solved,
        compatible_j_moduli_dimension=result.compatible_j_moduli_dimension,
        generated_j_solved=result.generated_j_solved,
        generated_complex_structure_count=len(result.generated_complex_structures),
        local_compatible_j_solved=result.local_compatible_j_solved,
        local_compatible_j_moduli_dimension=result.local_compatible_j_moduli_dimension,
        local_compatible_complex_structure_count=len(result.local_compatible_complex_structures),
        compatible_j_in_generated_algebra_count=sum(
            item.in_generated_algebra for item in diagnostic_tuple
        ),
        compatible_j_in_rule_local_center_count=sum(
            item.in_rule_local_center for item in diagnostic_tuple
        ),
        spectral_polarization_j_matched_count=sum(
            item.equals_spectral_polarization_j
            or item.equals_negative_spectral_polarization_j
            for item in diagnostic_tuple
        ),
        forced_j_found=result.forced_j_found,
        reason_for_forced_j_failure=_j_gap_failure_reason(result, diagnostic_tuple),
        compatible_j_diagnostics=diagnostic_tuple,
    )


def floquet_alpha_noncommuting_completion_certificate(
    candidate: FloquetAlphaNoncommutingCandidate,
    *,
    j_index: int = 0,
) -> FloquetAlphaNoncommutingCompletionCertificate:
    """Declare one finite compatible J as a diagnostic third layer.

    This is a completion experiment, not a physical claim.  It asks whether
    adding the missing local pair-orientation structure would close the strict
    bridge or create a new no-locking obstruction.
    """

    try:
        w_signs = floquet_alpha_noncommuting_completion_j_signs(
            candidate,
            j_index=j_index,
        )
    except IndexError as exc:
        raise ValueError(f"unknown compatible J index: {j_index}") from exc

    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    w = pair_orientation_j_operator(w_signs)
    previous_generated_basis = generated_algebra_basis((u1, u2))
    completed_generated_basis = generated_algebra_basis((u1, u2, w))
    completed_center_basis = center_basis_of_algebra(completed_generated_basis)
    center_solved, idempotents = solve_central_idempotents(
        completed_center_basis,
        max_center_dimension=8,
    )
    rank_pairs = _complementary_rank_6_4_pairs(idempotents)
    lower_rank = _lower_rank_idempotents_inside_pairs(idempotents, rank_pairs)
    compatible_basis = centralizer_basis((u1, u2, w))
    (
        compatible_j_solved,
        compatible_j_moduli_dimension,
        compatible_j,
    ) = solve_complex_structures_in_basis(
        compatible_basis,
        source="compatible_centralizer",
        max_basis_dimension=8,
    )
    (
        local_j_solved,
        local_j_moduli_dimension,
        local_j,
    ) = solve_complex_structures_in_basis(
        completed_center_basis,
        source="local_compatible_center",
        max_basis_dimension=8,
    )
    declared_w_is_local_compatible_j = any(item.matrix == w for item in local_j)
    strict_unique_j_found = (
        local_j_solved
        and local_j_moduli_dimension == 0
        and len(local_j) == 2
        and declared_w_is_local_compatible_j
    )
    pass_completion = bool(
        center_solved
        and rank_pairs
        and not lower_rank
        and strict_unique_j_found
    )
    if pass_completion:
        label = "completion_bridge_candidate"
    elif lower_rank:
        label = "completion_generates_lower_rank_center"
    elif local_j_solved and len(local_j) > 2:
        label = "completion_no_lower_rank_but_j_still_block_sign_ambiguous"
    else:
        label = "completion_j_not_forced"

    zero = sp.zeros(10)
    one = identity(10)
    return FloquetAlphaNoncommutingCompletionCertificate(
        candidate_name=candidate.name,
        completed_j_index=j_index,
        completed_j_pair_orientation_signs=w_signs,
        w_in_previous_generated_algebra=_matrix_in_span(w, previous_generated_basis),
        w_in_completed_generated_algebra=_matrix_in_span(w, completed_generated_basis),
        w_in_completed_center=_matrix_in_span(w, completed_center_basis),
        w_commutes_with_u1=is_zero_matrix(commutator(w, u1)),
        w_commutes_with_u2=is_zero_matrix(commutator(w, u2)),
        w_squares_to_minus_identity=sp.simplify(w * w + one) == zero,
        w_orthogonal=sp.simplify(w.T * w - one) == zero,
        generated_algebra_dimension=len(completed_generated_basis),
        center_dimension=len(completed_center_basis),
        center_solved=center_solved,
        central_idempotent_ranks=tuple(item.rank for item in idempotents),
        complementary_rank_6_4_pairs=len(rank_pairs),
        lower_rank_central_idempotents=len(lower_rank),
        compatible_centralizer_dimension=len(compatible_basis),
        compatible_j_solved=compatible_j_solved,
        compatible_j_moduli_dimension=compatible_j_moduli_dimension,
        compatible_complex_structure_count=len(compatible_j),
        local_compatible_operator_dimension=len(completed_center_basis),
        local_compatible_j_solved=local_j_solved,
        local_compatible_j_moduli_dimension=local_j_moduli_dimension,
        local_compatible_complex_structure_count=len(local_j),
        declared_w_is_local_compatible_j=declared_w_is_local_compatible_j,
        strict_unique_j_found=strict_unique_j_found,
        pass_completion_to_bridge=pass_completion,
        completion_label=label,
    )
