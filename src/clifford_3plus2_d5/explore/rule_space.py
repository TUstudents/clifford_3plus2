"""Data models for bounded microscopic rule-space exploration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.search.addressability import AddressableOperator


PrimitiveClass = Literal[
    "identity",
    "global_clock",
    "whole_block",
    "mode_permutation",
    "rank_one_pair",
]
RejectionReason = Literal[
    "no_j",
    "no_period_four",
    "no_split_candidate",
    "rank_one_addressability",
    "off_block_addressability",
    "unsafe_addressability",
    "normalizer_too_large",
    "normalizer_falsified",
]


@dataclass(frozen=True)
class ExplorationPrimitive:
    name: str
    matrix: sp.Matrix
    primitive_class: PrimitiveClass
    independently_addressable_pair: bool = False
    source: str = "candidate"


@dataclass(frozen=True)
class PrimitiveFamily:
    name: str
    primitives: tuple[ExplorationPrimitive, ...]


@dataclass(frozen=True)
class PrimitiveSet:
    name: str
    primitives: tuple[ExplorationPrimitive, ...]
    addressable_operators: tuple[AddressableOperator, ...]


@dataclass(frozen=True)
class RuleSpace:
    name: str
    primitive_families: tuple[PrimitiveFamily, ...]
    primitive_sets: tuple[PrimitiveSet, ...]


@dataclass(frozen=True)
class SearchBounds:
    max_depth: int = 4
    max_primitive_set_size: int = 4
    max_primitive_sets: int = 200
    max_words: int = 50_000
    max_survivors: int = 50
    max_rejections: int = 200


@dataclass(frozen=True)
class CandidateEvaluation:
    primitive_set_name: str
    word: tuple[str, ...]
    depth: int
    is_real_orthogonal: bool
    determinant: int | None
    j_squared_minus_identity: bool
    period_four_check_passed: bool
    equals_standard_j: bool
    projector_pair_found: bool
    addressability_algebra_safe: bool
    normalizer_verdict: str
    rejection_reasons: tuple[RejectionReason, ...]
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class ExplorationSummary:
    rule_space_name: str
    primitive_family_count: int
    primitive_set_count: int
    primitive_sets_scanned: int
    words_scanned: int
    j_hits: int
    period_four_hits: int
    split_candidates: int
    addressability_safe_hits: int
    normalizer_candidate_hits: int
    forced_candidate_hits: int
    rank_one_rejections: int
    off_block_rejections: int
    normalizer_too_large_rejections: int
    surviving_candidates: int
    top_rejection_reasons: tuple[tuple[str, int], ...]
    exploration_check_passed: bool
    load_bearing_qca_bridge: bool = False


def primitive_to_dict(primitive: ExplorationPrimitive) -> dict[str, object]:
    return {
        "name": primitive.name,
        "primitive_class": primitive.primitive_class,
        "independently_addressable_pair": primitive.independently_addressable_pair,
        "source": primitive.source,
    }


def primitive_set_to_dict(primitive_set: PrimitiveSet) -> dict[str, object]:
    return {
        "name": primitive_set.name,
        "primitives": [primitive_to_dict(primitive) for primitive in primitive_set.primitives],
        "addressable_operators": [
            operator.name for operator in primitive_set.addressable_operators
        ],
    }


def bounds_to_dict(bounds: SearchBounds) -> dict[str, int]:
    return {
        "max_depth": bounds.max_depth,
        "max_primitive_set_size": bounds.max_primitive_set_size,
        "max_primitive_sets": bounds.max_primitive_sets,
        "max_words": bounds.max_words,
        "max_survivors": bounds.max_survivors,
        "max_rejections": bounds.max_rejections,
    }


def evaluation_to_dict(evaluation: CandidateEvaluation) -> dict[str, object]:
    return {
        "primitive_set_name": evaluation.primitive_set_name,
        "word": list(evaluation.word),
        "depth": evaluation.depth,
        "is_real_orthogonal": evaluation.is_real_orthogonal,
        "determinant": evaluation.determinant,
        "j_squared_minus_identity": evaluation.j_squared_minus_identity,
        "period_four_check_passed": evaluation.period_four_check_passed,
        "equals_standard_j": evaluation.equals_standard_j,
        "projector_pair_found": evaluation.projector_pair_found,
        "addressability_algebra_safe": evaluation.addressability_algebra_safe,
        "normalizer_verdict": evaluation.normalizer_verdict,
        "rejection_reasons": list(evaluation.rejection_reasons),
        "load_bearing_qca_bridge": evaluation.load_bearing_qca_bridge,
    }


def summary_to_dict(
    summary: ExplorationSummary,
    *,
    bounds: SearchBounds | None = None,
    primitive_sets: Sequence[PrimitiveSet] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_space_name": summary.rule_space_name,
        "primitive_family_count": summary.primitive_family_count,
        "primitive_set_count": summary.primitive_set_count,
        "primitive_sets_scanned": summary.primitive_sets_scanned,
        "words_scanned": summary.words_scanned,
        "j_hits": summary.j_hits,
        "period_four_hits": summary.period_four_hits,
        "split_candidates": summary.split_candidates,
        "addressability_safe_hits": summary.addressability_safe_hits,
        "normalizer_candidate_hits": summary.normalizer_candidate_hits,
        "forced_candidate_hits": summary.forced_candidate_hits,
        "rank_one_rejections": summary.rank_one_rejections,
        "off_block_rejections": summary.off_block_rejections,
        "normalizer_too_large_rejections": (
            summary.normalizer_too_large_rejections
        ),
        "surviving_candidates": summary.surviving_candidates,
        "top_rejection_reasons": [
            {"reason": reason, "count": count}
            for reason, count in summary.top_rejection_reasons
        ],
        "exploration_check_passed": summary.exploration_check_passed,
        "load_bearing_qca_bridge": summary.load_bearing_qca_bridge,
    }
    if bounds is not None:
        payload["bounds"] = bounds_to_dict(bounds)
    if primitive_sets:
        payload["primitive_sets"] = [
            primitive_set_to_dict(primitive_set) for primitive_set in primitive_sets
        ]
    return payload
