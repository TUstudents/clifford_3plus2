"""Unseeded projector discovery for E2 exploration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.explore.primitives import (
    block_reflection_primitive,
    global_clock_primitive,
    identity_primitive,
    minus_identity_primitive,
    mode_swap_primitive,
    rank_one_pair_primitive,
)
from clifford_3plus2_d5.obstruction_r10.explore.rule_space import ExplorationPrimitive, PrimitiveSet
from clifford_3plus2_d5.obstruction_r10.search.addressability import (
    AddressableOperator,
    rank_one_color_projector_controls,
    standard_block_projectors,
)
from clifford_3plus2_d5.obstruction_r10.search.gate_words import PrimitiveGate, scan_gate_words


ProjectorDiscoveryMode = Literal[
    "unseeded",
    "sanity-seeded",
    "block-reflection-candidate",
]
ProjectorDiscoveryVerdict = Literal["projector_pair_found", "not_found"]


@dataclass(frozen=True)
class ProjectorDiscoveryBounds:
    max_word_depth: int = 2
    max_basis_size: int = 6
    coefficient_values: tuple[sp.Rational, ...] = (
        sp.Rational(-1),
        sp.Rational(-1, 2),
        sp.Rational(0),
        sp.Rational(1, 2),
        sp.Rational(1),
    )
    max_candidates: int = 200


@dataclass(frozen=True)
class BasisElement:
    name: str
    matrix: sp.Matrix
    source: str


@dataclass(frozen=True)
class ProjectorCandidate:
    mode: ProjectorDiscoveryMode
    primitive_set_name: str
    expression: tuple[tuple[str, sp.Rational], ...]
    rank: int
    commutes_with_j: bool
    central_in_rule_data: bool
    unsafe_rank_one_projector: bool
    equals_standard_p3: bool
    equals_standard_p2: bool


@dataclass(frozen=True)
class PrimitiveSetDiscovery:
    mode: ProjectorDiscoveryMode
    primitive_set_name: str
    basis_dimension: int
    coefficient_combinations_considered: int
    candidate_projectors: int
    rank_2_projectors: int
    rank_6_projectors: int
    rank_4_projectors: int
    complementary_pairs: int
    unsafe_rank_one_projectors: int
    verdict: ProjectorDiscoveryVerdict


@dataclass(frozen=True)
class ProjectorDiscoverySummary:
    mode: ProjectorDiscoveryMode
    primitive_sets_scanned: int
    algebra_elements_considered: int
    candidate_projectors: int
    rank_2_projectors: int
    rank_6_projectors: int
    rank_4_projectors: int
    complementary_pairs: int
    unsafe_rank_one_projectors: int
    unseeded_projector_pairs_found: int
    seeded_projector_pairs_found: int
    block_reflection_pairs_found: int
    discovery_verdict: ProjectorDiscoveryVerdict
    discovery_check_passed: bool
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class ProjectorDiscoveryRun:
    summary: ProjectorDiscoverySummary
    primitive_results: tuple[PrimitiveSetDiscovery, ...]
    candidates: tuple[ProjectorCandidate, ...]
    rejections: tuple[dict[str, object], ...]
    bounds: ProjectorDiscoveryBounds


def _empty_controls() -> tuple[AddressableOperator, ...]:
    return ()


def _primitive_set(
    name: str,
    primitives: tuple[ExplorationPrimitive, ...],
    addressable_operators: tuple[AddressableOperator, ...] = (),
) -> PrimitiveSet:
    return PrimitiveSet(name, primitives, addressable_operators)


def primitive_sets_for_mode(mode: ProjectorDiscoveryMode) -> tuple[PrimitiveSet, ...]:
    clock = global_clock_primitive()
    if mode == "sanity-seeded":
        return (
            _primitive_set(
                "sanity_seeded_standard_projectors",
                (identity_primitive(),),
                standard_block_projectors(),
            ),
        )
    if mode == "block-reflection-candidate":
        return (
            _primitive_set(
                "block_reflection_candidate",
                (clock, block_reflection_primitive()),
                _empty_controls(),
            ),
        )
    return (
        _primitive_set("identity_only_unseeded", (identity_primitive(),)),
        _primitive_set("clock_only_unseeded", (clock,)),
        _primitive_set(
            "clock_plus_minus_identity_unseeded",
            (clock, minus_identity_primitive()),
        ),
        _primitive_set(
            "clock_plus_color_swap_unseeded",
            (clock, mode_swap_primitive(0, 1)),
        ),
        _primitive_set(
            "clock_plus_weak_swap_unseeded",
            (clock, mode_swap_primitive(3, 4)),
        ),
        _primitive_set(
            "rank_one_pair_falsifier_unseeded",
            (clock, rank_one_pair_primitive(0)),
        ),
    )


def rank_one_color_control_set() -> PrimitiveSet:
    return _primitive_set(
        "rank_one_color_control_falsifier",
        (global_clock_primitive(),),
        rank_one_color_projector_controls(),
    )


def _append_if_independent(basis: list[BasisElement], candidate: BasisElement) -> None:
    matrices = [element.matrix for element in basis]
    if not basis or matrix_span_rank([*matrices, candidate.matrix]) > matrix_span_rank(matrices):
        basis.append(candidate)


def _primitive_gate_projection(
    primitives: tuple[ExplorationPrimitive, ...],
) -> tuple[PrimitiveGate, ...]:
    return tuple(
        PrimitiveGate(
            name=primitive.name,
            matrix=primitive.matrix,
            independently_addressable=primitive.independently_addressable_pair,
        )
        for primitive in primitives
    )


def generated_basis(
    primitive_set: PrimitiveSet,
    *,
    bounds: ProjectorDiscoveryBounds,
) -> tuple[BasisElement, ...]:
    basis: list[BasisElement] = []
    _append_if_independent(
        basis,
        BasisElement("I", identity(10), "identity"),
    )
    for primitive in primitive_set.primitives:
        _append_if_independent(
            basis,
            BasisElement(primitive.name, primitive.matrix, "primitive"),
        )
    for operator in primitive_set.addressable_operators:
        _append_if_independent(
            basis,
            BasisElement(operator.name, operator.matrix, "addressable_operator"),
        )

    primitive_gates = _primitive_gate_projection(primitive_set.primitives)
    if primitive_gates:
        for word, matrix in scan_gate_words(primitive_gates, bounds.max_word_depth):
            _append_if_independent(
                basis,
                BasisElement("*".join(word), matrix, "word_product"),
            )
            if len(basis) >= bounds.max_basis_size:
                return tuple(basis[: bounds.max_basis_size])
    return tuple(basis[: bounds.max_basis_size])


def _expression(
    coefficients: Sequence[sp.Rational],
    basis: Sequence[BasisElement],
) -> tuple[tuple[str, sp.Rational], ...]:
    return tuple(
        (element.name, coefficient)
        for coefficient, element in zip(coefficients, basis, strict=True)
        if coefficient != 0
    )


def _rule_matrices(primitive_set: PrimitiveSet) -> tuple[sp.Matrix, ...]:
    return tuple(primitive.matrix for primitive in primitive_set.primitives) + tuple(
        operator.matrix for operator in primitive_set.addressable_operators
    )


def _is_central(matrix: sp.Matrix, rule_matrices: tuple[sp.Matrix, ...]) -> bool:
    return all(is_zero_matrix(commutator(matrix, rule_matrix)) for rule_matrix in rule_matrices)


def _candidate_to_dict(candidate: ProjectorCandidate) -> dict[str, object]:
    return {
        "mode": candidate.mode,
        "primitive_set_name": candidate.primitive_set_name,
        "expression": [
            {"basis": name, "coefficient": str(coefficient)}
            for name, coefficient in candidate.expression
        ],
        "rank": candidate.rank,
        "commutes_with_j": candidate.commutes_with_j,
        "central_in_rule_data": candidate.central_in_rule_data,
        "unsafe_rank_one_projector": candidate.unsafe_rank_one_projector,
        "equals_standard_p3": candidate.equals_standard_p3,
        "equals_standard_p2": candidate.equals_standard_p2,
    }


def _matrix_from_expression(
    expression: tuple[tuple[str, sp.Rational], ...],
    basis: Sequence[BasisElement],
    dimension: int,
) -> sp.Matrix:
    matrix = sp.zeros(dimension)
    for basis_name, coefficient in expression:
        for element in basis:
            if element.name == basis_name:
                matrix += coefficient * element.matrix
                break
    return matrix


def primitive_result_to_dict(result: PrimitiveSetDiscovery) -> dict[str, object]:
    return {
        "mode": result.mode,
        "primitive_set_name": result.primitive_set_name,
        "basis_dimension": result.basis_dimension,
        "coefficient_combinations_considered": (
            result.coefficient_combinations_considered
        ),
        "candidate_projectors": result.candidate_projectors,
        "rank_2_projectors": result.rank_2_projectors,
        "rank_6_projectors": result.rank_6_projectors,
        "rank_4_projectors": result.rank_4_projectors,
        "complementary_pairs": result.complementary_pairs,
        "unsafe_rank_one_projectors": result.unsafe_rank_one_projectors,
        "verdict": result.verdict,
    }


def bounds_to_dict(bounds: ProjectorDiscoveryBounds) -> dict[str, object]:
    return {
        "max_word_depth": bounds.max_word_depth,
        "max_basis_size": bounds.max_basis_size,
        "coefficient_values": [str(value) for value in bounds.coefficient_values],
        "max_candidates": bounds.max_candidates,
    }


def summary_to_dict(
    summary: ProjectorDiscoverySummary,
    *,
    bounds: ProjectorDiscoveryBounds | None = None,
    primitive_results: tuple[PrimitiveSetDiscovery, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": summary.mode,
        "primitive_sets_scanned": summary.primitive_sets_scanned,
        "algebra_elements_considered": summary.algebra_elements_considered,
        "candidate_projectors": summary.candidate_projectors,
        "rank_2_projectors": summary.rank_2_projectors,
        "rank_6_projectors": summary.rank_6_projectors,
        "rank_4_projectors": summary.rank_4_projectors,
        "complementary_pairs": summary.complementary_pairs,
        "unsafe_rank_one_projectors": summary.unsafe_rank_one_projectors,
        "unseeded_projector_pairs_found": summary.unseeded_projector_pairs_found,
        "seeded_projector_pairs_found": summary.seeded_projector_pairs_found,
        "block_reflection_pairs_found": summary.block_reflection_pairs_found,
        "discovery_verdict": summary.discovery_verdict,
        "discovery_check_passed": summary.discovery_check_passed,
        "load_bearing_qca_bridge": summary.load_bearing_qca_bridge,
    }
    if bounds is not None:
        payload["bounds"] = bounds_to_dict(bounds)
    if primitive_results:
        payload["primitive_results"] = [
            primitive_result_to_dict(result) for result in primitive_results
        ]
    return payload


def discover_projectors_in_primitive_set(
    *,
    mode: ProjectorDiscoveryMode,
    primitive_set: PrimitiveSet,
    bounds: ProjectorDiscoveryBounds,
) -> tuple[PrimitiveSetDiscovery, tuple[ProjectorCandidate, ...], dict[str, object] | None]:
    carrier = standard_real_carrier()
    basis = generated_basis(primitive_set, bounds=bounds)
    rule_matrices = _rule_matrices(primitive_set)
    seen: list[sp.Matrix] = []
    candidates: list[ProjectorCandidate] = []
    combinations_considered = 0
    for coefficients in product(bounds.coefficient_values, repeat=len(basis)):
        if all(coefficient == 0 for coefficient in coefficients):
            continue
        combinations_considered += 1

        matrix = sp.zeros(carrier.dimension)
        for coefficient, element in zip(coefficients, basis, strict=True):
            matrix += coefficient * element.matrix
        if matrix * matrix != matrix:
            continue
        if any(matrix == seen_matrix for seen_matrix in seen):
            continue

        rank = matrix.rank()
        if rank not in {2, 4, 6}:
            continue
        commutes_with_j = is_zero_matrix(commutator(matrix, carrier.complex_structure))
        central = _is_central(matrix, rule_matrices)
        if not commutes_with_j or not central:
            continue

        seen.append(matrix)
        candidates.append(
            ProjectorCandidate(
                mode=mode,
                primitive_set_name=primitive_set.name,
                expression=_expression(coefficients, basis),
                rank=rank,
                commutes_with_j=commutes_with_j,
                central_in_rule_data=central,
                unsafe_rank_one_projector=rank == 2,
                equals_standard_p3=matrix == carrier.projector_3,
                equals_standard_p2=matrix == carrier.projector_2,
            )
        )
        if len(candidates) >= bounds.max_candidates:
            break

    rank_2 = tuple(candidate for candidate in candidates if candidate.rank == 2)
    rank_6 = tuple(candidate for candidate in candidates if candidate.rank == 6)
    rank_4 = tuple(candidate for candidate in candidates if candidate.rank == 4)
    matrices_by_expression = {
        candidate.expression: _matrix_from_expression(
            candidate.expression,
            basis,
            carrier.dimension,
        )
        for candidate in candidates
    }
    complementary_pairs = 0
    for p6 in rank_6:
        p6_matrix = matrices_by_expression[p6.expression]
        for p4 in rank_4:
            p4_matrix = matrices_by_expression[p4.expression]
            if (
                p6_matrix + p4_matrix == identity(carrier.dimension)
                and p6_matrix * p4_matrix == sp.zeros(carrier.dimension)
            ):
                complementary_pairs += 1

    result = PrimitiveSetDiscovery(
        mode=mode,
        primitive_set_name=primitive_set.name,
        basis_dimension=len(basis),
        coefficient_combinations_considered=combinations_considered,
        candidate_projectors=len(candidates),
        rank_2_projectors=len(rank_2),
        rank_6_projectors=len(rank_6),
        rank_4_projectors=len(rank_4),
        complementary_pairs=complementary_pairs,
        unsafe_rank_one_projectors=len(rank_2),
        verdict="projector_pair_found" if complementary_pairs else "not_found",
    )
    rejection = None
    if complementary_pairs == 0:
        rejection = {
            "mode": mode,
            "primitive_set_name": primitive_set.name,
            "reason": "no_complementary_6_4_projector_pair",
            "basis_dimension": len(basis),
            "candidate_projectors": len(candidates),
            "unsafe_rank_one_projectors": len(rank_2),
        }
    elif rank_2:
        rejection = {
            "mode": mode,
            "primitive_set_name": primitive_set.name,
            "reason": "unsafe_rank_one_projector_detected",
            "basis_dimension": len(basis),
            "candidate_projectors": len(candidates),
            "unsafe_rank_one_projectors": len(rank_2),
        }
    return result, tuple(candidates), rejection


def run_projector_discovery(
    *,
    mode: ProjectorDiscoveryMode,
    bounds: ProjectorDiscoveryBounds | None = None,
    primitive_sets: tuple[PrimitiveSet, ...] | None = None,
) -> ProjectorDiscoveryRun:
    bounds = bounds or ProjectorDiscoveryBounds()
    primitive_sets = primitive_sets or primitive_sets_for_mode(mode)
    primitive_results: list[PrimitiveSetDiscovery] = []
    candidates: list[ProjectorCandidate] = []
    rejections: list[dict[str, object]] = []
    for primitive_set in primitive_sets:
        result, set_candidates, rejection = discover_projectors_in_primitive_set(
            mode=mode,
            primitive_set=primitive_set,
            bounds=bounds,
        )
        primitive_results.append(result)
        candidates.extend(set_candidates)
        if rejection is not None:
            rejections.append(rejection)

    complementary_pairs = sum(result.complementary_pairs for result in primitive_results)
    unsafe_rank_one = sum(result.unsafe_rank_one_projectors for result in primitive_results)
    if mode == "unseeded":
        unseeded_pairs = complementary_pairs
        seeded_pairs = 0
        block_reflection_pairs = 0
    elif mode == "sanity-seeded":
        unseeded_pairs = 0
        seeded_pairs = complementary_pairs
        block_reflection_pairs = 0
    else:
        unseeded_pairs = 0
        seeded_pairs = 0
        block_reflection_pairs = complementary_pairs

    discovery_check_passed = (mode == "unseeded" and complementary_pairs == 0) or (
        mode in {"sanity-seeded", "block-reflection-candidate"}
        and complementary_pairs > 0
    )
    summary = ProjectorDiscoverySummary(
        mode=mode,
        primitive_sets_scanned=len(primitive_sets),
        algebra_elements_considered=sum(
            result.coefficient_combinations_considered for result in primitive_results
        ),
        candidate_projectors=sum(result.candidate_projectors for result in primitive_results),
        rank_2_projectors=sum(result.rank_2_projectors for result in primitive_results),
        rank_6_projectors=sum(result.rank_6_projectors for result in primitive_results),
        rank_4_projectors=sum(result.rank_4_projectors for result in primitive_results),
        complementary_pairs=complementary_pairs,
        unsafe_rank_one_projectors=unsafe_rank_one,
        unseeded_projector_pairs_found=unseeded_pairs,
        seeded_projector_pairs_found=seeded_pairs,
        block_reflection_pairs_found=block_reflection_pairs,
        discovery_verdict="projector_pair_found" if complementary_pairs else "not_found",
        discovery_check_passed=discovery_check_passed,
    )
    return ProjectorDiscoveryRun(
        summary=summary,
        primitive_results=tuple(primitive_results),
        candidates=tuple(candidates),
        rejections=tuple(rejections),
        bounds=bounds,
    )


def write_projector_discovery_artifacts(
    run: ProjectorDiscoveryRun,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "e2_summary.json"
    candidates_path = output_dir / "e2_projector_candidates.jsonl"
    rejections_path = output_dir / "e2_rejections.jsonl"

    summary_path.write_text(
        json.dumps(
            summary_to_dict(
                run.summary,
                bounds=run.bounds,
                primitive_results=run.primitive_results,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path.write_text(
        "".join(
            json.dumps(_candidate_to_dict(candidate), sort_keys=True) + "\n"
            for candidate in run.candidates
        ),
        encoding="utf-8",
    )
    rejections_path.write_text(
        "".join(json.dumps(rejection, sort_keys=True) + "\n" for rejection in run.rejections),
        encoding="utf-8",
    )
    return summary_path, candidates_path, rejections_path
