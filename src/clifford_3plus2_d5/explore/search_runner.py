"""Bounded exact rule-space exploration runner."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.explore.projector_discovery import discover_projector_pair
from clifford_3plus2_d5.explore.primitives import default_e1_rule_space
from clifford_3plus2_d5.explore.rule_space import (
    CandidateEvaluation,
    ExplorationPrimitive,
    ExplorationSummary,
    PrimitiveSet,
    RejectionReason,
    RuleSpace,
    SearchBounds,
    evaluation_to_dict,
    summary_to_dict,
)
from clifford_3plus2_d5.search.addressability import structural_split_certificate
from clifford_3plus2_d5.search.forced_j import is_real_orthogonal
from clifford_3plus2_d5.search.gate_words import PrimitiveGate, scan_gate_words
from clifford_3plus2_d5.search.normalizer import normalizer_certificate


@dataclass(frozen=True)
class PrimitiveSetContext:
    primitive_set: PrimitiveSet
    projector_pair_found: bool
    addressability_algebra_safe: bool
    rank_one_addressability: bool
    off_block_addressability: bool
    normalizer_verdict: str
    normalizer_check_passed: bool
    normalizer_too_large: bool


@dataclass(frozen=True)
class ExplorationRun:
    summary: ExplorationSummary
    survivors: tuple[CandidateEvaluation, ...]
    rejections: tuple[CandidateEvaluation, ...]
    bounds: SearchBounds
    primitive_sets_scanned: tuple[PrimitiveSet, ...]


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


def _primitive_set_context(primitive_set: PrimitiveSet) -> PrimitiveSetContext:
    split = structural_split_certificate(operators=primitive_set.addressable_operators)
    normalizer = normalizer_certificate(
        rule_data_operators=primitive_set.addressable_operators,
        addressable_operators=primitive_set.addressable_operators,
    )
    projector_discovery = discover_projector_pair(
        rule_data_operators=primitive_set.addressable_operators
    )
    rank_one_addressability = (
        split.rank_one_color_projectors_addressable
        or split.rank_one_weak_projectors_addressable
        or any(
            primitive.independently_addressable_pair
            for primitive in primitive_set.primitives
        )
    )
    normalizer_too_large = not (
        normalizer.j_unique_or_forced and normalizer.split_unique_or_forced
    )
    return PrimitiveSetContext(
        primitive_set=primitive_set,
        projector_pair_found=projector_discovery.projector_pair_found,
        addressability_algebra_safe=split.addressability_algebra_safe
        and normalizer.addressability_algebra_safe,
        rank_one_addressability=rank_one_addressability,
        off_block_addressability=split.off_block_controls_addressable,
        normalizer_verdict=normalizer.forcedness_verdict,
        normalizer_check_passed=(
            normalizer.candidate_j_valid
            and normalizer.candidate_split_valid
            and normalizer.normalizer_preserves_declared_split
            and normalizer.addressability_algebra_safe
        ),
        normalizer_too_large=normalizer_too_large,
    )


def _evaluate_word(
    *,
    primitive_set_name: str,
    word: tuple[str, ...],
    matrix,
    context: PrimitiveSetContext,
) -> CandidateEvaluation:
    carrier = standard_real_carrier()
    dimension = carrier.dimension
    orthogonal = matrix.shape == (dimension, dimension) and is_real_orthogonal(
        matrix,
        carrier.metric,
    )
    determinant = int(matrix.det()) if matrix.shape == (dimension, dimension) else None
    squares_to_minus_identity = matrix * matrix == -identity(dimension)
    period_four = squares_to_minus_identity and matrix**4 == identity(dimension)
    equals_standard_j = matrix == carrier.complex_structure
    j_hit = orthogonal and determinant == 1 and squares_to_minus_identity

    reasons: list[RejectionReason] = []
    if not j_hit:
        reasons.append("no_j")
    if not period_four:
        reasons.append("no_period_four")
    if not context.projector_pair_found:
        reasons.append("no_split_candidate")
    if context.rank_one_addressability:
        reasons.append("rank_one_addressability")
    if context.off_block_addressability:
        reasons.append("off_block_addressability")
    if not context.addressability_algebra_safe:
        reasons.append("unsafe_addressability")
    if context.normalizer_verdict == "falsified":
        reasons.append("normalizer_falsified")
    elif context.normalizer_too_large:
        reasons.append("normalizer_too_large")

    return CandidateEvaluation(
        primitive_set_name=primitive_set_name,
        word=word,
        depth=len(word),
        is_real_orthogonal=orthogonal,
        determinant=determinant,
        j_squared_minus_identity=squares_to_minus_identity,
        period_four_check_passed=period_four,
        equals_standard_j=equals_standard_j,
        projector_pair_found=context.projector_pair_found,
        addressability_algebra_safe=context.addressability_algebra_safe,
        normalizer_verdict=context.normalizer_verdict,
        rejection_reasons=tuple(reasons),
    )


def run_rule_space_exploration(
    rule_space: RuleSpace | None = None,
    bounds: SearchBounds | None = None,
) -> ExplorationRun:
    rule_space = rule_space or default_e1_rule_space()
    bounds = bounds or SearchBounds()

    words_scanned = 0
    j_hits = 0
    period_four_hits = 0
    split_candidates = 0
    addressability_safe_hits = 0
    normalizer_candidate_hits = 0
    forced_candidate_hits = 0
    reason_counter: Counter[str] = Counter()
    survivors: list[CandidateEvaluation] = []
    rejections: list[CandidateEvaluation] = []
    scanned_sets: list[PrimitiveSet] = []

    for primitive_set in rule_space.primitive_sets[: bounds.max_primitive_sets]:
        if len(primitive_set.primitives) > bounds.max_primitive_set_size:
            continue
        if words_scanned >= bounds.max_words:
            break

        scanned_sets.append(primitive_set)
        context = _primitive_set_context(primitive_set)
        primitive_gates = _primitive_gate_projection(primitive_set.primitives)
        for word, matrix in scan_gate_words(primitive_gates, bounds.max_depth):
            if words_scanned >= bounds.max_words:
                break

            words_scanned += 1
            evaluation = _evaluate_word(
                primitive_set_name=primitive_set.name,
                word=word,
                matrix=matrix,
                context=context,
            )

            if (
                evaluation.is_real_orthogonal
                and evaluation.determinant == 1
                and evaluation.j_squared_minus_identity
            ):
                j_hits += 1
            if evaluation.period_four_check_passed:
                period_four_hits += 1
            if evaluation.projector_pair_found:
                split_candidates += 1
            if evaluation.addressability_algebra_safe:
                addressability_safe_hits += 1
            if context.normalizer_check_passed:
                normalizer_candidate_hits += 1
            if not evaluation.rejection_reasons:
                forced_candidate_hits += 1
                if len(survivors) < bounds.max_survivors:
                    survivors.append(evaluation)
            else:
                reason_counter.update(evaluation.rejection_reasons)
                if len(rejections) < bounds.max_rejections:
                    rejections.append(evaluation)

    summary = ExplorationSummary(
        rule_space_name=rule_space.name,
        primitive_family_count=len(rule_space.primitive_families),
        primitive_set_count=len(rule_space.primitive_sets),
        primitive_sets_scanned=len(scanned_sets),
        words_scanned=words_scanned,
        j_hits=j_hits,
        period_four_hits=period_four_hits,
        split_candidates=split_candidates,
        addressability_safe_hits=addressability_safe_hits,
        normalizer_candidate_hits=normalizer_candidate_hits,
        forced_candidate_hits=forced_candidate_hits,
        rank_one_rejections=reason_counter["rank_one_addressability"],
        off_block_rejections=reason_counter["off_block_addressability"],
        normalizer_too_large_rejections=reason_counter["normalizer_too_large"],
        surviving_candidates=forced_candidate_hits,
        top_rejection_reasons=tuple(reason_counter.most_common(8)),
        exploration_check_passed=(
            len(scanned_sets) > 1
            and words_scanned > 100
            and j_hits > 0
            and period_four_hits > 0
        ),
    )
    return ExplorationRun(
        summary=summary,
        survivors=tuple(survivors),
        rejections=tuple(rejections),
        bounds=bounds,
        primitive_sets_scanned=tuple(scanned_sets),
    )


def write_exploration_artifacts(
    run: ExplorationRun,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "e1_summary.json"
    survivors_path = output_dir / "e1_survivors.jsonl"
    rejections_path = output_dir / "e1_rejections.jsonl"

    summary_path.write_text(
        json.dumps(
            summary_to_dict(
                run.summary,
                bounds=run.bounds,
                primitive_sets=run.primitive_sets_scanned,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    survivors_path.write_text(
        "".join(
            json.dumps(evaluation_to_dict(evaluation), sort_keys=True) + "\n"
            for evaluation in run.survivors
        ),
        encoding="utf-8",
    )
    rejections_path.write_text(
        "".join(
            json.dumps(evaluation_to_dict(evaluation), sort_keys=True) + "\n"
            for evaluation in run.rejections
        ),
        encoding="utf-8",
    )
    return summary_path, survivors_path, rejections_path
