"""Scan runners for the complex-linear split lab."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from clifford_3plus2_d5.lepton.complex_primitives import (
    iter_complex_c3_candidates,
    iter_complex_c5_candidates,
    iter_complex_c5_discovered_candidates,
)
from clifford_3plus2_d5.lepton.complex_profiles import (
    complex_c3_split_profile,
    complex_c5_discovered_split_profile,
    complex_c5_split_profile,
)
from clifford_3plus2_d5.lepton.complex_verdict import (
    ComplexSplitResult,
    rule_to_complex_split_verdict,
)


@dataclass(frozen=True)
class ComplexSplitScanRow:
    rule_name: str
    verdict: str
    reason: str
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int
    central_idempotent_ranks: tuple[int, ...]
    target_split_present: bool
    seeded_split_present: bool
    forbidden_rank_one_count: int
    forbidden_idempotent_count: int
    block_dimensions: tuple[int, ...]
    block_commutativity: tuple[bool, ...]
    discovered_projector_ranks: tuple[int, ...] = ()
    canonical_split_matched: bool = False
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ComplexSplitSummary:
    name: str
    candidate_count: int
    verdict_counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]
    split_candidate_count: int
    non_synthetic_split_candidate_count: int
    canonical_split_candidate_count: int
    noncanonical_split_candidate_count: int
    seeded_control_count: int
    max_generated_algebra_dimension: int
    max_center_dimension: int


def _metadata_value(metadata: Sequence[tuple[str, str]], key: str) -> str | None:
    for item_key, item_value in metadata:
        if item_key == key:
            return item_value
    return None


def _is_non_synthetic(row: ComplexSplitScanRow) -> bool:
    return _metadata_value(row.metadata, "synthetic") == "false"


def complex_split_scan_row(
    rule_name: str,
    result: ComplexSplitResult,
    *,
    metadata: Sequence[tuple[str, str]] = (),
) -> ComplexSplitScanRow:
    return ComplexSplitScanRow(
        rule_name=rule_name,
        verdict=result.verdict.value,
        reason=result.reason,
        generated_algebra_dimension=result.generated_algebra_dimension,
        generated_algebra_closed=result.generated_algebra_closed,
        center_dimension=result.center_dimension,
        central_idempotent_ranks=tuple(
            sorted(idempotent.rank for idempotent in result.central_idempotents)
        ),
        target_split_present=result.target_split_present,
        seeded_split_present=result.seeded_split_present,
        forbidden_rank_one_count=result.forbidden_rank_one_count,
        forbidden_idempotent_count=result.forbidden_idempotent_count,
        block_dimensions=result.block_dimensions,
        block_commutativity=result.block_commutativity,
        discovered_projector_ranks=result.discovered_projector_ranks,
        canonical_split_matched=result.canonical_split_matched,
        metadata=tuple(metadata),
    )


def run_complex_c3_split_scan(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (3, 4),
) -> tuple[ComplexSplitScanRow, ...]:
    profile = complex_c3_split_profile()
    rows = []
    for candidate in iter_complex_c3_candidates(
        max_candidates=max_candidates,
        phase_orders=phase_orders,
    ):
        result = rule_to_complex_split_verdict(candidate.layers, profile)
        rows.append(complex_split_scan_row(candidate.name, result, metadata=candidate.metadata))
    return tuple(rows)


def run_complex_c5_split_scan(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (3, 4),
) -> tuple[ComplexSplitScanRow, ...]:
    profile = complex_c5_split_profile()
    rows = []
    for candidate in iter_complex_c5_candidates(
        max_candidates=max_candidates,
        phase_orders=phase_orders,
    ):
        result = rule_to_complex_split_verdict(candidate.layers, profile)
        rows.append(complex_split_scan_row(candidate.name, result, metadata=candidate.metadata))
    return tuple(rows)


def run_complex_c5_discovered_split_scan(
    *,
    max_candidates: int | None = None,
    phase_orders: tuple[int, ...] = (2,),
    include_conjugated: bool = True,
    family: str = "controls",
) -> tuple[ComplexSplitScanRow, ...]:
    profile = complex_c5_discovered_split_profile()
    rows = []
    for candidate in iter_complex_c5_discovered_candidates(
        max_candidates=max_candidates,
        phase_orders=phase_orders,
        include_conjugated=include_conjugated,
        family=family,
    ):
        result = rule_to_complex_split_verdict(candidate.layers, profile)
        rows.append(complex_split_scan_row(candidate.name, result, metadata=candidate.metadata))
    return tuple(rows)


def summarize_complex_rows(
    name: str,
    rows: Sequence[ComplexSplitScanRow],
) -> ComplexSplitSummary:
    verdict_counts = Counter(row.verdict for row in rows)
    reason_counts = Counter(row.reason for row in rows)
    return ComplexSplitSummary(
        name=name,
        candidate_count=len(rows),
        verdict_counts=tuple(sorted(verdict_counts.items(), key=lambda item: (-item[1], item[0]))),
        reason_counts=tuple(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        split_candidate_count=sum(1 for row in rows if row.verdict == "split_candidate"),
        non_synthetic_split_candidate_count=sum(
            1
            for row in rows
            if row.verdict == "split_candidate" and _is_non_synthetic(row)
        ),
        canonical_split_candidate_count=sum(
            1
            for row in rows
            if row.verdict == "split_candidate" and row.canonical_split_matched
        ),
        noncanonical_split_candidate_count=sum(
            1
            for row in rows
            if row.verdict == "split_candidate" and not row.canonical_split_matched
        ),
        seeded_control_count=sum(1 for row in rows if row.verdict == "seeded_split_control"),
        max_generated_algebra_dimension=max(
            (row.generated_algebra_dimension for row in rows),
            default=0,
        ),
        max_center_dimension=max((row.center_dimension for row in rows), default=0),
    )


def complex_c3_summary(
    *,
    max_candidates: int | None = None,
) -> ComplexSplitSummary:
    return summarize_complex_rows(
        "complex_c3_split",
        run_complex_c3_split_scan(max_candidates=max_candidates),
    )


def complex_c5_summary(
    *,
    max_candidates: int | None = None,
) -> ComplexSplitSummary:
    return summarize_complex_rows(
        "complex_c5_split",
        run_complex_c5_split_scan(max_candidates=max_candidates),
    )


def complex_c5_discovered_summary(
    *,
    max_candidates: int | None = None,
    family: str = "controls",
) -> ComplexSplitSummary:
    return summarize_complex_rows(
        f"complex_c5_discovered_split_{family}",
        run_complex_c5_discovered_split_scan(
            max_candidates=max_candidates,
            family=family,
        ),
    )
