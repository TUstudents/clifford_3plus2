from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations, permutations

from clifford_3plus2_d5.qca.bloch_rule import bloch_path_a_projector_free_layer
from clifford_3plus2_d5.qca.rule_verdict import (
    bloch_floquet_operators,
    center_basis_of_algebra,
    centralizer_basis,
    generated_algebra_closure,
    solve_central_idempotents,
)


@dataclass(frozen=True)
class StepwiseCandidate:
    pattern_index: int
    cycle: tuple[int, ...]
    source_shifts: tuple[int, ...]


@dataclass(frozen=True)
class StepwiseResult:
    name: str
    pattern_index: int
    cycle: tuple[int, ...]
    source_shifts: tuple[int, ...]
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int | None
    center_solved: bool | None
    compatible_centralizer_dimension: int | None
    central_idempotent_ranks: tuple[int, ...]
    route_label: str
    elapsed_seconds: float


def _five_cycles(limit: int) -> tuple[tuple[int, ...], ...]:
    cycles = []
    for tail in permutations((1, 2, 3, 4)):
        sequence = (0, *tail)
        target_by_source = [0] * 5
        for source, target in zip(sequence, (*sequence[1:], sequence[0]), strict=True):
            target_by_source[source] = target
        cycles.append(tuple(target_by_source))
        if len(cycles) >= limit:
            break
    return tuple(cycles)


def _shift_assignments(limit: int) -> tuple[tuple[int, ...], ...]:
    assignments = []
    for four_sources in combinations(range(5), 3):
        assignments.append(tuple(4 if source in four_sources else 3 for source in range(5)))
        if len(assignments) >= limit:
            break
    return tuple(assignments)


def stepwise_candidates(
    *,
    pattern_count: int,
    cycle_count: int,
    shift_count: int,
    max_candidates: int,
) -> tuple[StepwiseCandidate, ...]:
    candidates = []
    for pattern_index in range(pattern_count):
        for cycle in _five_cycles(cycle_count):
            for source_shifts in _shift_assignments(shift_count):
                candidates.append(
                    StepwiseCandidate(
                        pattern_index=pattern_index,
                        cycle=cycle,
                        source_shifts=source_shifts,
                    )
                )
                if len(candidates) >= max_candidates:
                    return tuple(candidates)
    return tuple(candidates)


def _label_result(
    *,
    closed: bool,
    algebra_dimension: int,
    center_dimension: int | None,
    idempotent_ranks: tuple[int, ...],
) -> str:
    if not closed:
        return "cap_exceeded_structured"
    if algebra_dimension >= 80:
        return "closes_full_matrix_like"
    if idempotent_ranks == (0, 4, 6, 10):
        return "closes_coarse_6_4_center"
    if center_dimension == 4:
        return "closes_four_dim_center_unresolved"
    if center_dimension == 1:
        return "closes_trivial_center"
    if center_dimension is None:
        return "closes_center_not_checked"
    return "closes_other_center"


def evaluate_candidate(
    candidate: StepwiseCandidate,
    *,
    max_algebra_dim: int,
    include_center: bool,
    include_centralizer: bool,
    include_idempotents: bool,
) -> StepwiseResult:
    start = time.perf_counter()
    layer = bloch_path_a_projector_free_layer(
        pattern_index=candidate.pattern_index,
        cycle=candidate.cycle,
        source_shifts=candidate.source_shifts,
    )
    samples = bloch_floquet_operators((layer,), bloch_period=12)
    closure = generated_algebra_closure(samples, max_dimension=max_algebra_dim)
    center_dimension = None
    center_solved = None
    compatible_centralizer_dimension = None
    idempotent_ranks: tuple[int, ...] = ()
    if include_center and closure.closed:
        center = center_basis_of_algebra(closure.basis)
        center_dimension = len(center)
        if include_idempotents:
            center_solved, idempotents = solve_central_idempotents(center)
            idempotent_ranks = tuple(item.rank for item in idempotents)
    if include_centralizer and closure.closed:
        compatible_centralizer_dimension = len(centralizer_basis(samples))

    return StepwiseResult(
        name=layer.name,
        pattern_index=candidate.pattern_index,
        cycle=candidate.cycle,
        source_shifts=candidate.source_shifts,
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=closure.closed,
        center_dimension=center_dimension,
        center_solved=center_solved,
        compatible_centralizer_dimension=compatible_centralizer_dimension,
        central_idempotent_ranks=idempotent_ranks,
        route_label=_label_result(
            closed=closure.closed,
            algebra_dimension=len(closure.basis),
            center_dimension=center_dimension,
            idempotent_ranks=idempotent_ranks,
        ),
        elapsed_seconds=time.perf_counter() - start,
    )


def _evaluate_for_pool(
    args: tuple[StepwiseCandidate, int, bool, bool, bool],
) -> StepwiseResult:
    candidate, max_algebra_dim, include_center, include_centralizer, include_idempotents = args
    return evaluate_candidate(
        candidate,
        max_algebra_dim=max_algebra_dim,
        include_center=include_center,
        include_centralizer=include_centralizer,
        include_idempotents=include_idempotents,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stepwise projector-free Bloch Path-A closure search."
    )
    parser.add_argument("--pattern-count", type=int, default=1)
    parser.add_argument("--cycle-count", type=int, default=3)
    parser.add_argument("--shift-count", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-algebra-dim", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--center-top", type=int, default=0)
    parser.add_argument("--centralizer", action="store_true")
    parser.add_argument("--idempotents", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    candidates = stepwise_candidates(
        pattern_count=args.pattern_count,
        cycle_count=args.cycle_count,
        shift_count=args.shift_count,
        max_candidates=args.max_candidates,
    )
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")

    closure_args = [
        (candidate, args.max_algebra_dim, False, False, False)
        for candidate in candidates
    ]
    if args.jobs == 1:
        closure_results = tuple(_evaluate_for_pool(item) for item in closure_args)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            closure_results = tuple(executor.map(_evaluate_for_pool, closure_args))

    detailed_by_name = {}
    for result in closure_results:
        if len(detailed_by_name) >= args.center_top:
            break
        if not result.generated_algebra_closed:
            continue
        candidate = StepwiseCandidate(
            pattern_index=result.pattern_index,
            cycle=result.cycle,
            source_shifts=result.source_shifts,
        )
        detailed = evaluate_candidate(
            candidate,
            max_algebra_dim=args.max_algebra_dim,
            include_center=True,
            include_centralizer=args.centralizer,
            include_idempotents=args.idempotents,
        )
        detailed_by_name[result.name] = detailed

    results = tuple(detailed_by_name.get(result.name, result) for result in closure_results)
    payload = {
        "candidate_count": len(results),
        "closed_count": sum(result.generated_algebra_closed for result in results),
        "coarse_6_4_count": sum(
            result.route_label == "closes_coarse_6_4_center" for result in results
        ),
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs a stepwise projector-free Bloch Path-A search.")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"closed_count: {payload['closed_count']}")
        print(f"coarse_6_4_count: {payload['coarse_6_4_count']}")
        for result in results:
            print(
                "candidate: "
                f"{result.name}, "
                f"dim={result.generated_algebra_dimension}, "
                f"closed={str(result.generated_algebra_closed).lower()}, "
                f"center={result.center_dimension}, "
                f"centralizer={result.compatible_centralizer_dimension}, "
                f"ranks={list(result.central_idempotent_ranks)}, "
                f"label={result.route_label}, "
                f"time={result.elapsed_seconds:.3f}s"
            )

    if args.check:
        if not results or any(result.generated_algebra_dimension <= 0 for result in results):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
