"""Experimental profile-driven verdict path for the leptonic bridge labs.

This module is intentionally separate from the legacy ``qca.rule_verdict``
contract. It reuses the existing exact algebra kernels, but keeps the new
profile and verdict semantics inside the lepton lab while the design is still
experimental.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import exact_matrix_span
from clifford_3plus2_d5.algebra.matrices import commutator, identity
from clifford_3plus2_d5.qca.gates import is_real_matrix
from clifford_3plus2_d5.qca.rule_verdict import (
    CentralIdempotent,
    ComplexStructureCandidate,
    RuleLayerInput,
    center_basis_of_algebra,
    generated_algebra_closure,
    solve_central_idempotents,
)


class V2Verdict(StrEnum):
    """Verdict labels with global-bridge and domain-wall standards separated."""

    BRIDGE_CANDIDATE = "bridge_candidate"
    BRIDGE_CANDIDATE_J_BLIND = "bridge_candidate_j_blind"
    CLOCK_PLANE_CLOSURE_CANDIDATE = "clock_plane_closure_candidate"
    DOMAIN_WALL_CANDIDATE = "domain_wall_candidate"
    CANDIDATE_ONLY_J_NOT_FORCED = "candidate_only_j_not_forced"
    STRUCTURAL_CANDIDATE_ORBIT = "structural_candidate_orbit"
    NOT_SOLVED = "not_solved"
    FALSIFIED = "falsified"


class PrimitiveClass(StrEnum):
    """Layer provenance relative to the clock complex-structure basis."""

    ARBITRARY_REAL_ORTHOGONAL = "arbitrary_real_orthogonal"
    PER_MODE_CLOCK_ROTATION = "per_mode_clock_rotation"
    GLOBALLY_SYNCHRONIZED_CLOCK = "globally_synchronized_clock"
    EXPLICIT_J_LAYER = "explicit_j_layer"


class CentralJSolveStatus(StrEnum):
    SOLVED = "solved"
    TOO_LARGE = "too_large"
    UNSUPPORTED = "unsupported"
    NO_SOLUTIONS = "no_solutions"


class IdempotentPolicyResult(StrEnum):
    PASSED = "passed"
    MISSING_TARGET = "missing_target"
    FORBIDDEN_PRESENT = "forbidden_present"
    NO_NONTRIVIAL = "no_nontrivial"


class CommutantPolicyResult(StrEnum):
    PASSED_UNIQUE_PM = "passed_unique_pm"
    PASSED_MULTIPLE_ALIGNED = "passed_multiple_aligned"
    GAUGE_MISALIGNMENT = "gauge_misalignment"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class CentralJSolveResult:
    status: CentralJSolveStatus
    candidates: tuple[ComplexStructureCandidate, ...] = ()


CentralJCallable = Callable[["VerdictProfile", Sequence[sp.Matrix]], CentralJSolveResult]
IdempotentPolicyCallable = Callable[
    ["VerdictProfile", Sequence[CentralIdempotent]], IdempotentPolicyResult
]
CommutantPolicyCallable = Callable[
    [
        "VerdictProfile",
        Sequence[sp.Matrix],
        Sequence[sp.Matrix],
        Sequence[sp.Matrix],
        Sequence[ComplexStructureCandidate],
        object | None,
    ],
    CommutantPolicyResult,
]


@dataclass(frozen=True)
class VerdictProfile:
    name: str
    dimension: int
    central_j_candidates: CentralJCallable
    idempotent_policy: IdempotentPolicyCallable
    commutant_policy: CommutantPolicyCallable
    target_projectors: tuple[sp.Matrix, ...] = ()
    require_real_orthogonal_layers: bool = True
    require_finite_locality_radius: bool = True
    require_closed_generated_algebra: bool = True
    require_solved_center: bool = True
    require_noncommutative: bool = True
    minimum_algebra_dimension: int = 0
    expected_center_dimension: int | None = None
    expected_block_dimensions: tuple[int, ...] | None = None
    expected_block_commutativity: tuple[bool, ...] | None = None
    j_search_basis: Literal["center", "algebra"] = "center"
    j_uniqueness_required: Literal["unique_pm", "structural_orbit", "any_finite"] = "unique_pm"
    explicit_j_layer_detection: bool = True
    compatible_j_solver: Literal["nullspace", "splitting", "skip"] = "skip"
    max_algebra_dimension: int = 64
    max_center_dimension: int = 16
    max_compatible_basis_dimension: int = 16
    wall_context_required: bool = False
    gauge_generators: tuple[sp.Matrix, ...] = ()
    known_center_basis: tuple[sp.Matrix, ...] = ()
    verify_known_center_basis_exact: bool = False
    positive_verdict: V2Verdict = V2Verdict.BRIDGE_CANDIDATE
    use_target_projector_idempotents: bool = False


@dataclass(frozen=True)
class RuleToVerdictV2Result:
    profile_name: str
    dimension: int
    verdict: V2Verdict
    reason: str
    all_layers_real_orthogonal: bool
    all_layers_local: bool
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    primitive_classes: tuple[PrimitiveClass, ...]
    max_primitive_class: PrimitiveClass
    explicit_j_layer_present: bool
    seeded_j_in_strict_sense: bool
    center_dimension: int
    central_idempotents: tuple[CentralIdempotent, ...]
    idempotent_verdict: IdempotentPolicyResult
    central_j_solve_status: CentralJSolveStatus
    central_j_candidates: tuple[ComplexStructureCandidate, ...]
    forced_j: bool
    primitive_central_idempotents: tuple[CentralIdempotent, ...]
    block_dimensions: tuple[int, ...]
    block_commutativity: tuple[bool, ...]
    commutant_verdict: CommutantPolicyResult
    pass_rule_to_bridge: bool
    load_bearing_qca_bridge: bool = False
    load_bearing_domain_wall_candidate: bool = False


def _zero(dimension: int) -> sp.Matrix:
    return sp.zeros(dimension)


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _layer_is_local(layer: RuleLayerInput) -> bool:
    if not layer.support or any(site < 0 for site in layer.support):
        return False
    return layer.locality_radius >= max(layer.support) - min(layer.support)


def is_explicit_j_layer(matrix: sp.Matrix) -> bool:
    """Return true when a layer is itself an orthogonal complex structure."""

    if matrix.rows != matrix.cols:
        return False
    one = identity(matrix.rows)
    return _same_matrix(matrix.T * matrix, one) and _same_matrix(matrix * matrix, -one)


def _is_real_orthogonal(matrix: sp.Matrix) -> bool:
    return (
        matrix.rows == matrix.cols
        and is_real_matrix(matrix)
        and _same_matrix(matrix.T * matrix, identity(matrix.rows))
    )


def _mode_block(matrix: sp.Matrix, mode: int, mode_count: int) -> sp.Matrix:
    return sp.Matrix(
        [
            [matrix[mode, mode], matrix[mode, mode_count + mode]],
            [matrix[mode_count + mode, mode], matrix[mode_count + mode, mode_count + mode]],
        ]
    )


def _is_clock_pair_block(block: sp.Matrix) -> bool:
    a = block[0, 0]
    b = block[1, 0]
    return _same_matrix(block, sp.Matrix([[a, -b], [b, a]]))


def _is_pairwise_clock_rotation(matrix: sp.Matrix) -> tuple[bool, bool]:
    """Return (is_pairwise_clock_rotation, globally_synchronized)."""

    if matrix.rows != matrix.cols or matrix.rows % 2:
        return False, False
    if not _is_real_orthogonal(matrix):
        return False, False
    mode_count = matrix.rows // 2
    blocks = []
    for row_mode in range(mode_count):
        for col_mode in range(mode_count):
            if row_mode == col_mode:
                continue
            entries = (
                matrix[row_mode, col_mode],
                matrix[row_mode, mode_count + col_mode],
                matrix[mode_count + row_mode, col_mode],
                matrix[mode_count + row_mode, mode_count + col_mode],
            )
            if any(sp.simplify(entry) != 0 for entry in entries):
                return False, False
        block = _mode_block(matrix, row_mode, mode_count)
        if not _is_clock_pair_block(block):
            return False, False
        blocks.append(block)

    has_rotation = any(sp.simplify(block[1, 0]) != 0 for block in blocks)
    if not has_rotation:
        return False, False
    first = blocks[0]
    synchronized = all(_same_matrix(block, first) for block in blocks[1:])
    return True, synchronized


def classify_primitive(matrix: sp.Matrix) -> PrimitiveClass:
    if is_explicit_j_layer(matrix):
        return PrimitiveClass.EXPLICIT_J_LAYER
    is_clock_rotation, synchronized = _is_pairwise_clock_rotation(matrix)
    if synchronized:
        return PrimitiveClass.GLOBALLY_SYNCHRONIZED_CLOCK
    if is_clock_rotation:
        return PrimitiveClass.PER_MODE_CLOCK_ROTATION
    return PrimitiveClass.ARBITRARY_REAL_ORTHOGONAL


def primitive_class_severity(primitive_class: PrimitiveClass) -> int:
    return {
        PrimitiveClass.ARBITRARY_REAL_ORTHOGONAL: 0,
        PrimitiveClass.PER_MODE_CLOCK_ROTATION: 1,
        PrimitiveClass.GLOBALLY_SYNCHRONIZED_CLOCK: 2,
        PrimitiveClass.EXPLICIT_J_LAYER: 3,
    }[primitive_class]


def is_commutative(basis: Sequence[sp.Matrix]) -> bool:
    return all(
        _same_matrix(commutator(left, right), _zero(left.rows)) for left in basis for right in basis
    )


def _nonzero_idempotents(
    idempotents: Sequence[CentralIdempotent], *, dimension: int
) -> tuple[CentralIdempotent, ...]:
    zero = _zero(dimension)
    return tuple(item for item in idempotents if not _same_matrix(item.matrix, zero))


def _idempotent_leq(left: sp.Matrix, right: sp.Matrix) -> bool:
    return _same_matrix(left * right, left) and _same_matrix(right * left, left)


def compute_primitive_central_idempotents(
    idempotents: Sequence[CentralIdempotent], *, dimension: int
) -> tuple[CentralIdempotent, ...]:
    nonzero = _nonzero_idempotents(idempotents, dimension=dimension)
    primitive = []
    for candidate in nonzero:
        has_smaller = any(
            not _same_matrix(other.matrix, candidate.matrix)
            and _idempotent_leq(other.matrix, candidate.matrix)
            for other in nonzero
        )
        if not has_smaller:
            primitive.append(candidate)
    return tuple(sorted(primitive, key=lambda item: (item.rank, str(item.expression))))


def block_invariant_chain_reduced(
    algebra_basis: Sequence[sp.Matrix],
    primitive_idempotents: Sequence[CentralIdempotent],
    *,
    dimension: int,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    block_dimensions: list[int] = []
    block_commutativity: list[bool] = []
    for idempotent in primitive_idempotents:
        projected = [
            (idempotent.matrix * matrix * idempotent.matrix).applyfunc(sp.simplify)
            for matrix in algebra_basis
        ]
        span = exact_matrix_span(rows=dimension, cols=dimension, add_matrices=False)
        independent_basis = []
        for matrix in projected:
            if span.add(matrix):
                independent_basis.append(matrix)
        block_dimensions.append(len(independent_basis))
        block_commutativity.append(is_commutative(independent_basis))
    return tuple(block_dimensions), tuple(block_commutativity)


def _default_result(
    profile: VerdictProfile,
    *,
    verdict: V2Verdict,
    reason: str,
    all_layers_real_orthogonal: bool = False,
    all_layers_local: bool = False,
    generated_algebra_dimension: int = 0,
    generated_algebra_closed: bool = False,
    primitive_classes: Sequence[PrimitiveClass] = (),
    center_dimension: int = 0,
    central_idempotents: Sequence[CentralIdempotent] = (),
    idempotent_verdict: IdempotentPolicyResult = IdempotentPolicyResult.MISSING_TARGET,
    central_j_solve_status: CentralJSolveStatus = CentralJSolveStatus.UNSUPPORTED,
    central_j_candidates: Sequence[ComplexStructureCandidate] = (),
    primitive_central_idempotents: Sequence[CentralIdempotent] = (),
    block_dimensions: Sequence[int] = (),
    block_commutativity: Sequence[bool] = (),
    commutant_verdict: CommutantPolicyResult = CommutantPolicyResult.NOT_APPLICABLE,
) -> RuleToVerdictV2Result:
    primitive_tuple = tuple(primitive_classes)
    max_primitive = (
        max(primitive_tuple, key=primitive_class_severity)
        if primitive_tuple
        else PrimitiveClass.ARBITRARY_REAL_ORTHOGONAL
    )
    explicit_j = any(item == PrimitiveClass.EXPLICIT_J_LAYER for item in primitive_tuple)
    forced_j = bool(central_j_candidates) and not explicit_j
    load_bearing_qca_bridge = verdict in (
        V2Verdict.BRIDGE_CANDIDATE,
        V2Verdict.BRIDGE_CANDIDATE_J_BLIND,
    )
    return RuleToVerdictV2Result(
        profile_name=profile.name,
        dimension=profile.dimension,
        verdict=verdict,
        reason=reason,
        all_layers_real_orthogonal=all_layers_real_orthogonal,
        all_layers_local=all_layers_local,
        generated_algebra_dimension=generated_algebra_dimension,
        generated_algebra_closed=generated_algebra_closed,
        primitive_classes=primitive_tuple,
        max_primitive_class=max_primitive,
        explicit_j_layer_present=explicit_j,
        seeded_j_in_strict_sense=explicit_j,
        center_dimension=center_dimension,
        central_idempotents=tuple(central_idempotents),
        idempotent_verdict=idempotent_verdict,
        central_j_solve_status=central_j_solve_status,
        central_j_candidates=tuple(central_j_candidates),
        forced_j=forced_j,
        primitive_central_idempotents=tuple(primitive_central_idempotents),
        block_dimensions=tuple(block_dimensions),
        block_commutativity=tuple(block_commutativity),
        commutant_verdict=commutant_verdict,
        pass_rule_to_bridge=load_bearing_qca_bridge,
        load_bearing_qca_bridge=load_bearing_qca_bridge,
        load_bearing_domain_wall_candidate=verdict == V2Verdict.DOMAIN_WALL_CANDIDATE,
    )


def _positive_verdict(profile: VerdictProfile, max_primitive_class: PrimitiveClass) -> V2Verdict:
    if (
        profile.positive_verdict == V2Verdict.CLOCK_PLANE_CLOSURE_CANDIDATE
        and max_primitive_class == PrimitiveClass.ARBITRARY_REAL_ORTHOGONAL
    ):
        return V2Verdict.BRIDGE_CANDIDATE_J_BLIND
    return profile.positive_verdict


def rule_to_verdict_v2(
    layers: Sequence[RuleLayerInput],
    profile: VerdictProfile,
    *,
    wall_context: object | None = None,
) -> RuleToVerdictV2Result:
    if not layers:
        return _default_result(profile, verdict=V2Verdict.FALSIFIED, reason="no_layers")
    if any(layer.matrix.shape != (profile.dimension, profile.dimension) for layer in layers):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="wrong_layer_dimension",
        )
    if profile.wall_context_required and wall_context is None:
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="missing_wall_context",
        )

    all_layers_real_orthogonal = all(
        _is_real_orthogonal(layer.matrix) for layer in layers
    )
    all_layers_local = all(_layer_is_local(layer) for layer in layers)
    primitive_classes = tuple(classify_primitive(layer.matrix) for layer in layers)

    if profile.require_real_orthogonal_layers and not all_layers_real_orthogonal:
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="not_real_orthogonal",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            primitive_classes=primitive_classes,
        )
    if profile.require_finite_locality_radius and not all_layers_local:
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="nonlocal",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            primitive_classes=primitive_classes,
        )
    if profile.explicit_j_layer_detection and any(
        item == PrimitiveClass.EXPLICIT_J_LAYER for item in primitive_classes
    ):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="explicit_j_seeded",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            primitive_classes=primitive_classes,
        )

    closure = generated_algebra_closure(
        tuple(layer.matrix for layer in layers),
        dimension=profile.dimension,
        max_dimension=profile.max_algebra_dimension,
    )
    if profile.require_closed_generated_algebra and not closure.closed:
        return _default_result(
            profile,
            verdict=V2Verdict.NOT_SOLVED,
            reason="algebra_not_closed",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=False,
            primitive_classes=primitive_classes,
        )

    center_basis = (
        profile.known_center_basis
        if profile.known_center_basis
        else center_basis_of_algebra(closure.basis, dimension=profile.dimension)
    )
    if profile.known_center_basis and profile.verify_known_center_basis_exact:
        actual_center_basis = center_basis_of_algebra(closure.basis, dimension=profile.dimension)
        known_span = exact_matrix_span(rows=profile.dimension, cols=profile.dimension, add_matrices=False)
        for center_matrix in center_basis:
            known_span.add(center_matrix)
        known_center_is_exact = len(actual_center_basis) == len(center_basis) and not any(
            known_span.add(actual_center_matrix) for actual_center_matrix in actual_center_basis
        )
        if not known_center_is_exact:
            return _default_result(
                profile,
                verdict=V2Verdict.FALSIFIED,
                reason="known_center_basis_not_exact",
                all_layers_real_orthogonal=all_layers_real_orthogonal,
                all_layers_local=all_layers_local,
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=closure.closed,
                primitive_classes=primitive_classes,
                center_dimension=len(actual_center_basis),
            )
    if profile.known_center_basis:
        zero = sp.zeros(profile.dimension)
        known_center_is_central = all(
            _same_matrix(commutator(center_matrix, algebra_matrix), zero)
            for center_matrix in center_basis
            for algebra_matrix in closure.basis
        )
        if not known_center_is_central:
            return _default_result(
                profile,
                verdict=V2Verdict.FALSIFIED,
                reason="known_center_basis_not_central",
                all_layers_real_orthogonal=all_layers_real_orthogonal,
                all_layers_local=all_layers_local,
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=closure.closed,
                primitive_classes=primitive_classes,
                center_dimension=len(center_basis),
            )
    if len(center_basis) > profile.max_center_dimension:
        return _default_result(
            profile,
            verdict=V2Verdict.NOT_SOLVED,
            reason="center_too_large",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
        )

    if profile.use_target_projector_idempotents:
        zero = sp.zeros(profile.dimension)
        one = identity(profile.dimension)
        idempotents = (
            CentralIdempotent(expression=(("zero", sp.Integer(1)),), matrix=zero, rank=0),
            *(
                CentralIdempotent(
                    expression=((f"target_{index}", sp.Integer(1)),),
                    matrix=projector,
                    rank=projector.rank(),
                )
                for index, projector in enumerate(profile.target_projectors)
            ),
            CentralIdempotent(expression=(("identity", sp.Integer(1)),), matrix=one, rank=profile.dimension),
        )
        center_solved = True
    else:
        center_solved, idempotents = solve_central_idempotents(
            center_basis,
            max_center_dimension=profile.max_center_dimension,
            dimension=profile.dimension,
        )
    if profile.require_solved_center and not center_solved:
        return _default_result(
            profile,
            verdict=V2Verdict.NOT_SOLVED,
            reason="idempotents_not_solved",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
        )

    idempotent_verdict = profile.idempotent_policy(profile, idempotents)
    if idempotent_verdict != IdempotentPolicyResult.PASSED:
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason=f"idempotent_{idempotent_verdict.value}",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
        )

    primitive_idempotents = compute_primitive_central_idempotents(
        idempotents, dimension=profile.dimension
    )
    block_dimensions, block_commutativity = block_invariant_chain_reduced(
        closure.basis,
        primitive_idempotents,
        dimension=profile.dimension,
    )
    if (
        profile.expected_block_dimensions is not None
        and block_dimensions != profile.expected_block_dimensions
    ):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="block_dimensions_mismatch",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )
    if (
        profile.expected_block_commutativity is not None
        and block_commutativity != profile.expected_block_commutativity
    ):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="block_commutativity_mismatch",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )

    j_result = profile.central_j_candidates(profile, center_basis)
    if j_result.status == CentralJSolveStatus.TOO_LARGE:
        verdict = V2Verdict.NOT_SOLVED
        reason = "center_j_solve_too_large"
    elif j_result.status == CentralJSolveStatus.UNSUPPORTED:
        verdict = V2Verdict.NOT_SOLVED
        reason = "center_j_solve_unsupported"
    elif j_result.status == CentralJSolveStatus.NO_SOLUTIONS or not j_result.candidates:
        verdict = V2Verdict.FALSIFIED
        reason = "no_central_j"
    else:
        verdict = None
        reason = ""
    if verdict is not None:
        return _default_result(
            profile,
            verdict=verdict,
            reason=reason,
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )

    if profile.require_noncommutative and is_commutative(closure.basis):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="commutative",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )
    if len(closure.basis) < profile.minimum_algebra_dimension:
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="algebra_too_small",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )
    if (
        profile.expected_center_dimension is not None
        and len(center_basis) != profile.expected_center_dimension
    ):
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="center_dimension_mismatch",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
        )

    commutant_verdict = profile.commutant_policy(
        profile,
        closure.basis,
        center_basis,
        profile.target_projectors,
        j_result.candidates,
        wall_context,
    )

    if profile.j_uniqueness_required == "unique_pm":
        if commutant_verdict == CommutantPolicyResult.PASSED_UNIQUE_PM:
            positive = _positive_verdict(profile, max(primitive_classes, key=primitive_class_severity))
            return _default_result(
                profile,
                verdict=positive,
                reason="passed",
                all_layers_real_orthogonal=all_layers_real_orthogonal,
                all_layers_local=all_layers_local,
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=closure.closed,
                primitive_classes=primitive_classes,
                center_dimension=len(center_basis),
                central_idempotents=idempotents,
                idempotent_verdict=idempotent_verdict,
                central_j_solve_status=j_result.status,
                central_j_candidates=j_result.candidates,
                primitive_central_idempotents=primitive_idempotents,
                block_dimensions=block_dimensions,
                block_commutativity=block_commutativity,
                commutant_verdict=commutant_verdict,
            )
        if commutant_verdict == CommutantPolicyResult.PASSED_MULTIPLE_ALIGNED:
            return _default_result(
                profile,
                verdict=V2Verdict.CANDIDATE_ONLY_J_NOT_FORCED,
                reason="j_not_forced",
                all_layers_real_orthogonal=all_layers_real_orthogonal,
                all_layers_local=all_layers_local,
                generated_algebra_dimension=len(closure.basis),
                generated_algebra_closed=closure.closed,
                primitive_classes=primitive_classes,
                center_dimension=len(center_basis),
                central_idempotents=idempotents,
                idempotent_verdict=idempotent_verdict,
                central_j_solve_status=j_result.status,
                central_j_candidates=j_result.candidates,
                primitive_central_idempotents=primitive_idempotents,
                block_dimensions=block_dimensions,
                block_commutativity=block_commutativity,
                commutant_verdict=commutant_verdict,
            )
        return _default_result(
            profile,
            verdict=V2Verdict.FALSIFIED,
            reason="gauge_misalignment",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            commutant_verdict=commutant_verdict,
        )
    if profile.j_uniqueness_required == "structural_orbit":
        return _default_result(
            profile,
            verdict=V2Verdict.STRUCTURAL_CANDIDATE_ORBIT,
            reason="structural_orbit",
            all_layers_real_orthogonal=all_layers_real_orthogonal,
            all_layers_local=all_layers_local,
            generated_algebra_dimension=len(closure.basis),
            generated_algebra_closed=closure.closed,
            primitive_classes=primitive_classes,
            center_dimension=len(center_basis),
            central_idempotents=idempotents,
            idempotent_verdict=idempotent_verdict,
            central_j_solve_status=j_result.status,
            central_j_candidates=j_result.candidates,
            primitive_central_idempotents=primitive_idempotents,
            block_dimensions=block_dimensions,
            block_commutativity=block_commutativity,
            commutant_verdict=commutant_verdict,
        )
    return _default_result(
        profile,
        verdict=V2Verdict.STRUCTURAL_CANDIDATE_ORBIT,
        reason="any_finite",
        all_layers_real_orthogonal=all_layers_real_orthogonal,
        all_layers_local=all_layers_local,
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=closure.closed,
        primitive_classes=primitive_classes,
        center_dimension=len(center_basis),
        central_idempotents=idempotents,
        idempotent_verdict=idempotent_verdict,
        central_j_solve_status=j_result.status,
        central_j_candidates=j_result.candidates,
        primitive_central_idempotents=primitive_idempotents,
        block_dimensions=block_dimensions,
        block_commutativity=block_commutativity,
        commutant_verdict=commutant_verdict,
    )


def _idempotent_to_dict(idempotent: CentralIdempotent) -> dict[str, object]:
    return {
        "expression": [
            {"basis": name, "coefficient": str(coefficient)}
            for name, coefficient in idempotent.expression
        ],
        "rank": idempotent.rank,
    }


def _complex_structure_to_dict(candidate: ComplexStructureCandidate) -> dict[str, object]:
    return {
        "expression": [
            {"basis": name, "coefficient": str(coefficient)}
            for name, coefficient in candidate.expression
        ],
        "source": candidate.source,
    }


def result_to_dict_v2(result: RuleToVerdictV2Result) -> dict[str, object]:
    return {
        "profile_name": result.profile_name,
        "dimension": result.dimension,
        "verdict": result.verdict.value,
        "reason": result.reason,
        "all_layers_real_orthogonal": result.all_layers_real_orthogonal,
        "all_layers_local": result.all_layers_local,
        "generated_algebra_dimension": result.generated_algebra_dimension,
        "generated_algebra_closed": result.generated_algebra_closed,
        "primitive_classes": [item.value for item in result.primitive_classes],
        "max_primitive_class": result.max_primitive_class.value,
        "explicit_j_layer_present": result.explicit_j_layer_present,
        "seeded_j_in_strict_sense": result.seeded_j_in_strict_sense,
        "center_dimension": result.center_dimension,
        "central_idempotent_ranks": [item.rank for item in result.central_idempotents],
        "central_idempotents": [
            _idempotent_to_dict(idempotent) for idempotent in result.central_idempotents
        ],
        "idempotent_verdict": result.idempotent_verdict.value,
        "central_j_solve_status": result.central_j_solve_status.value,
        "central_j_candidate_count": len(result.central_j_candidates),
        "central_j_candidates": [
            _complex_structure_to_dict(candidate) for candidate in result.central_j_candidates
        ],
        "forced_j": result.forced_j,
        "primitive_central_idempotent_ranks": [
            item.rank for item in result.primitive_central_idempotents
        ],
        "block_dimensions": list(result.block_dimensions),
        "block_commutativity": list(result.block_commutativity),
        "commutant_verdict": result.commutant_verdict.value,
        "pass_rule_to_bridge": result.pass_rule_to_bridge,
        "load_bearing_qca_bridge": result.load_bearing_qca_bridge,
        "load_bearing_domain_wall_candidate": result.load_bearing_domain_wall_candidate,
    }
