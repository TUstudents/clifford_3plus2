"""Summary and audit helpers for the leptonic bridge laboratory."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Queue, get_context
from queue import Empty

from clifford_3plus2_d5.obstruction_r10.qca.rule_verdict import (
    center_basis_of_algebra,
    generated_algebra_closure,
    solve_central_idempotents,
)
from clifford_3plus2_d5.lepton.lab_b import (
    iter_lab_b_domain_wall_candidates,
    iter_lab_b_physical_domain_wall_candidates,
)
from clifford_3plus2_d5.lepton.profiles import (
    lab_b_domain_wall_profile,
    lab_b_physical_domain_wall_profile,
)
from clifford_3plus2_d5.lepton.scans import (
    LabAScanRow,
    LabBScanRow,
    run_lab_a_bloch_scan,
    run_lab_a_onsite_scan,
    run_lab_a_wall_scan,
    run_lab_b_domain_wall_scan,
    run_lab_b_physical_domain_wall_scan,
    run_lab_b_strict_scan,
    run_lab_b_structural_wall_scan,
)
from clifford_3plus2_d5.lepton.verdict import rule_to_verdict_v2


@dataclass(frozen=True)
class ScanSummary:
    name: str
    candidate_count: int
    verdict_counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]
    transition_tier_counts: tuple[tuple[str, int], ...]
    qca_bridge_count: int
    domain_wall_candidate_count: int
    max_generated_algebra_dimension: int
    max_center_dimension: int


@dataclass(frozen=True)
class DomainWallAudit:
    candidate_name: str
    wall_context_consistent: bool
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    actual_center_dimension: int
    center_idempotents_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    verdict: str
    reason: str
    central_j_candidate_count: int
    load_bearing_qca_bridge: bool
    load_bearing_domain_wall_candidate: bool


@dataclass(frozen=True)
class PhysicalDomainWallAudit:
    candidate_name: str
    wall_context_consistent: bool
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    exact_center_status: str
    actual_center_dimension: int | None
    center_idempotents_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    verdict: str
    reason: str
    central_j_candidate_count: int
    load_bearing_qca_bridge: bool
    load_bearing_domain_wall_candidate: bool


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _metadata_value(row: LabAScanRow | LabBScanRow, key: str) -> str | None:
    metadata = dict(row.metadata)
    return metadata.get(key)


def summarize_rows(name: str, rows: Sequence[LabAScanRow | LabBScanRow]) -> ScanSummary:
    verdict_counts = Counter(row.verdict for row in rows)
    reason_counts = Counter(row.reason for row in rows)
    transition_counts = Counter(
        tier for row in rows if (tier := _metadata_value(row, "transition_tier")) is not None
    )
    return ScanSummary(
        name=name,
        candidate_count=len(rows),
        verdict_counts=_sorted_counts(verdict_counts),
        reason_counts=_sorted_counts(reason_counts),
        transition_tier_counts=_sorted_counts(transition_counts),
        qca_bridge_count=sum(1 for row in rows if row.load_bearing_qca_bridge),
        domain_wall_candidate_count=sum(
            1
            for row in rows
            if isinstance(row, LabBScanRow) and row.load_bearing_domain_wall_candidate
        ),
        max_generated_algebra_dimension=max(
            (row.generated_algebra_dimension for row in rows),
            default=0,
        ),
        max_center_dimension=max((row.center_dimension for row in rows), default=0),
    )


def lab_a_summary(*, max_candidates_per_mechanism: int = 5) -> tuple[ScanSummary, ...]:
    return (
        summarize_rows(
            "lab_a_on_site",
            run_lab_a_onsite_scan(max_candidates=max_candidates_per_mechanism),
        ),
        summarize_rows(
            "lab_a_bloch",
            run_lab_a_bloch_scan(max_candidates=max_candidates_per_mechanism),
        ),
        summarize_rows(
            "lab_a_wall",
            run_lab_a_wall_scan(max_candidates=max_candidates_per_mechanism),
        ),
    )


def lab_b_strict_summary(*, max_candidates: int = 5) -> ScanSummary:
    return summarize_rows(
        "lab_b_strict",
        run_lab_b_strict_scan(max_candidates=max_candidates),
    )


def lab_b_structural_wall_summary(
    *,
    max_candidates: int = 3,
    max_pairs: int = 4,
) -> ScanSummary:
    return summarize_rows(
        "lab_b_structural_wall",
        run_lab_b_structural_wall_scan(max_candidates=max_candidates, max_pairs=max_pairs),
    )


def lab_b_domain_wall_summary(
    *,
    max_candidates: int = 1,
    max_pairs: int = 1,
    verify_center_exact: bool = False,
) -> ScanSummary:
    return summarize_rows(
        "lab_b_domain_wall",
        run_lab_b_domain_wall_scan(
            max_candidates=max_candidates,
            max_pairs=max_pairs,
            verify_center_exact=verify_center_exact,
        ),
    )


def lab_b_physical_domain_wall_summary(
    *,
    max_candidates: int = 1,
    max_pairs: int = 1,
) -> ScanSummary:
    return summarize_rows(
        "lab_b_physical_domain_wall",
        run_lab_b_physical_domain_wall_scan(
            max_candidates=max_candidates,
            max_pairs=max_pairs,
        ),
    )


def audit_first_domain_wall_candidate(
    *,
    max_pairs: int = 1,
) -> DomainWallAudit:
    candidate = next(iter_lab_b_domain_wall_candidates(max_candidates=1, max_pairs=max_pairs))
    profile = lab_b_domain_wall_profile()
    result = rule_to_verdict_v2(
        candidate.layers,
        profile,
        wall_context=candidate.wall_context,
    )
    closure = generated_algebra_closure(
        tuple(layer.matrix for layer in candidate.layers),
        dimension=profile.dimension,
        max_dimension=profile.max_algebra_dimension,
    )
    center_basis = center_basis_of_algebra(closure.basis, dimension=profile.dimension)
    idempotents_solved, idempotents = solve_central_idempotents(
        center_basis,
        max_center_dimension=profile.max_center_dimension,
        dimension=profile.dimension,
    )
    return DomainWallAudit(
        candidate_name=candidate.name,
        wall_context_consistent=bool(
            candidate.wall_context and candidate.wall_context.consistency_certified()
        ),
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=closure.closed,
        actual_center_dimension=len(center_basis),
        center_idempotents_solved=idempotents_solved,
        central_idempotent_ranks=tuple(sorted(item.rank for item in idempotents)),
        verdict=result.verdict.value,
        reason=result.reason,
        central_j_candidate_count=len(result.central_j_candidates),
        load_bearing_qca_bridge=result.load_bearing_qca_bridge,
        load_bearing_domain_wall_candidate=result.load_bearing_domain_wall_candidate,
    )


def _physical_domain_wall_exact_center_payload(max_pairs: int) -> dict[str, object]:
    candidate = next(iter_lab_b_physical_domain_wall_candidates(max_candidates=1, max_pairs=max_pairs))
    profile = lab_b_physical_domain_wall_profile()
    closure = generated_algebra_closure(
        tuple(layer.matrix for layer in candidate.layers),
        dimension=profile.dimension,
        max_dimension=profile.max_algebra_dimension,
    )
    center_basis = center_basis_of_algebra(closure.basis, dimension=profile.dimension)
    center_idempotents_solved, idempotents = solve_central_idempotents(
        center_basis,
        max_center_dimension=profile.max_center_dimension,
        dimension=profile.dimension,
    )
    ranks = tuple(sorted(item.rank for item in idempotents)) if center_idempotents_solved else ()
    status = "passed_exact"
    if not closure.closed or len(center_basis) != 2 or not center_idempotents_solved or ranks != (0, 12):
        status = "failed"
    return {
        "generated_algebra_dimension": len(closure.basis),
        "generated_algebra_closed": closure.closed,
        "exact_center_status": status,
        "actual_center_dimension": len(center_basis),
        "center_idempotents_solved": center_idempotents_solved,
        "central_idempotent_ranks": ranks,
    }


def _physical_domain_wall_exact_center_worker(max_pairs: int, queue: Queue) -> None:
    try:
        queue.put(("ok", _physical_domain_wall_exact_center_payload(max_pairs)))
    except BaseException as exc:  # pragma: no cover - exercised through parent timeout/error path.
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_physical_exact_center_with_timeout(
    *,
    max_pairs: int,
    timeout_seconds: int,
) -> dict[str, object]:
    context = get_context("fork")
    queue: Queue = context.Queue()
    process = context.Process(
        target=_physical_domain_wall_exact_center_worker,
        args=(max_pairs, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "exact_center_status": "not_solved_timeout",
            "actual_center_dimension": None,
            "center_idempotents_solved": False,
            "central_idempotent_ranks": (),
        }

    try:
        status, payload = queue.get_nowait()
    except Empty:
        return {
            "exact_center_status": "not_solved_no_result",
            "actual_center_dimension": None,
            "center_idempotents_solved": False,
            "central_idempotent_ranks": (),
        }
    if status != "ok":
        return {
            "exact_center_status": "not_solved_error",
            "actual_center_dimension": None,
            "center_idempotents_solved": False,
            "central_idempotent_ranks": (),
            "error": payload,
        }
    return payload


def audit_first_physical_domain_wall_candidate(
    *,
    max_pairs: int = 1,
    timeout_seconds: int = 30,
) -> PhysicalDomainWallAudit:
    candidate = next(iter_lab_b_physical_domain_wall_candidates(max_candidates=1, max_pairs=max_pairs))
    profile = lab_b_physical_domain_wall_profile()
    result = rule_to_verdict_v2(
        candidate.layers,
        profile,
        wall_context=candidate.wall_context,
    )
    exact = _run_physical_exact_center_with_timeout(
        max_pairs=max_pairs,
        timeout_seconds=timeout_seconds,
    )
    return PhysicalDomainWallAudit(
        candidate_name=candidate.name,
        wall_context_consistent=bool(
            candidate.wall_context and candidate.wall_context.consistency_certified()
        ),
        generated_algebra_dimension=int(
            exact.get("generated_algebra_dimension", result.generated_algebra_dimension)
        ),
        generated_algebra_closed=bool(
            exact.get("generated_algebra_closed", result.generated_algebra_closed)
        ),
        exact_center_status=str(exact["exact_center_status"]),
        actual_center_dimension=(
            int(exact["actual_center_dimension"])
            if exact.get("actual_center_dimension") is not None
            else None
        ),
        center_idempotents_solved=bool(exact["center_idempotents_solved"]),
        central_idempotent_ranks=tuple(int(rank) for rank in exact["central_idempotent_ranks"]),
        verdict=result.verdict.value,
        reason=result.reason,
        central_j_candidate_count=len(result.central_j_candidates),
        load_bearing_qca_bridge=result.load_bearing_qca_bridge,
        load_bearing_domain_wall_candidate=result.load_bearing_domain_wall_candidate,
    )
