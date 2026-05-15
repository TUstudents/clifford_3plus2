from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations, permutations

import sympy as sp

from clifford_3plus2_d5.qca.bloch_rule import (
    BlochRuleLayerInput,
    bloch_layer_laurent_orthogonal,
    bloch_path_a_polynomial_hop_layer,
    bloch_path_a_projector_free_layer,
    bloch_seed_guardrail,
)
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    bloch_floquet_operators,
    center_basis_of_algebra,
    centralizer_basis,
    generated_algebra_closure,
    solve_central_idempotents,
    solve_complex_structures_from_idempotent_splitting,
)
from clifford_3plus2_d5.qca.spatial_1d import SpatialHoppingTerm


@dataclass(frozen=True)
class StepwiseCandidate:
    family: str
    pattern_index: int
    cycle: tuple[int, ...]
    source_shifts: tuple[int, ...]
    polynomial_terms: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = ()
    mixes_by_shift: tuple[tuple[int, tuple[int, int]], ...] = ()
    name_suffix: str = ""


@dataclass(frozen=True)
class StepwiseResult:
    name: str
    family: str
    pattern_index: int
    cycle: tuple[int, ...]
    source_shifts: tuple[int, ...]
    seed_guardrail_checked: bool
    seed_guardrail_passed: bool
    raw_seed_witnesses: tuple[str, ...]
    algebraic_seed_witnesses: tuple[str, ...]
    coefficient_algebra_dimension: int | None
    laurent_orthogonal: bool
    generated_algebra_dimension: int
    generated_algebra_closed: bool
    center_dimension: int | None
    center_solved: bool | None
    compatible_centralizer_dimension: int | None
    central_idempotent_ranks: tuple[int, ...]
    generated_j_solved: bool | None
    generated_j_moduli_dimension: int | None
    generated_j_count: int | None
    generated_j_sign_shape: str | None
    compatible_j_solved: bool | None
    compatible_j_moduli_dimension: int | None
    compatible_j_count: int | None
    compatible_j_sign_shape: str | None
    bridge_j_status: str | None
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
    family: str,
    pattern_count: int,
    cycle_count: int,
    shift_count: int,
    max_candidates: int,
) -> tuple[StepwiseCandidate, ...]:
    if family == "polynomial-hop":
        return polynomial_hop_candidates(
            pattern_count=pattern_count,
            cycle_count=cycle_count,
            shift_count=shift_count,
            max_candidates=max_candidates,
        )
    if family != "monomial-hop":
        raise ValueError(f"unknown family: {family}")
    candidates = []
    for pattern_index in range(pattern_count):
        for cycle in _five_cycles(cycle_count):
            for source_shifts in _shift_assignments(shift_count):
                candidates.append(
                    StepwiseCandidate(
                        family=family,
                        pattern_index=pattern_index,
                        cycle=cycle,
                        source_shifts=source_shifts,
                    )
                )
                if len(candidates) >= max_candidates:
                    return tuple(candidates)
    return tuple(candidates)


def _edges_by_shift(
    cycle: tuple[int, ...],
    source_shifts: tuple[int, ...],
) -> dict[int, tuple[tuple[int, int], ...]]:
    return {
        shift: tuple(
            (source, cycle[source])
            for source, source_shift in enumerate(source_shifts)
            if source_shift == shift
        )
        for shift in sorted(set(source_shifts))
    }


def polynomial_hop_candidates(
    *,
    pattern_count: int,
    cycle_count: int,
    shift_count: int,
    max_candidates: int,
) -> tuple[StepwiseCandidate, ...]:
    """Enumerate Laurent-orthogonal polynomial-hop candidates.

    Each candidate keeps the monomial source/target shift partition but replaces
    one shift coefficient by a finite-order rational reflection on two of its
    mode edges. This is the smallest projector-free class beyond partial
    permutations that still satisfies the exact Laurent orthogonality identity.
    """

    candidates = []
    for pattern_index in range(pattern_count):
        for cycle in _five_cycles(cycle_count):
            for source_shifts in _shift_assignments(shift_count):
                base_edges = _edges_by_shift(cycle, source_shifts)
                for shift in (3, 4):
                    edges = base_edges.get(shift, ())
                    if len(edges) < 2:
                        continue
                    for left, right in combinations(range(len(edges)), 2):
                        terms = tuple(
                            (item_shift, base_edges[item_shift])
                            for item_shift in sorted(base_edges)
                        )
                        mixes = ((shift, (left, right)),)
                        candidates.append(
                            StepwiseCandidate(
                                family="polynomial-hop",
                                pattern_index=pattern_index,
                                cycle=cycle,
                                source_shifts=source_shifts,
                                polynomial_terms=terms,
                                mixes_by_shift=mixes,
                                name_suffix=(
                                    f"c{''.join(str(item) for item in cycle)}_"
                                    f"s{''.join(str(item) for item in source_shifts)}_"
                                    f"m{shift}{left}{right}"
                                ),
                            )
                        )
                        if len(candidates) >= max_candidates:
                            return tuple(candidates)
    return tuple(candidates)


def _candidate_name(candidate: StepwiseCandidate) -> str:
    if candidate.family == "monomial-hop":
        default_candidate = (
            candidate.pattern_index == 0
            and candidate.cycle == (1, 2, 3, 4, 0)
            and candidate.source_shifts == (4, 4, 4, 3, 3)
        )
        if default_candidate:
            return "path_a_projector_free_cycle_combined"
        return (
            "path_a_projector_free_"
            f"p{candidate.pattern_index}_"
            f"c{''.join(str(item) for item in candidate.cycle)}_"
            f"s{''.join(str(item) for item in candidate.source_shifts)}"
        )
    if candidate.family == "polynomial-hop":
        return f"path_a_polynomial_hop_p{candidate.pattern_index}_{candidate.name_suffix}"
    raise ValueError(f"unknown candidate family: {candidate.family}")


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


def _layer_for_candidate(candidate: StepwiseCandidate) -> RuleLayerInput:
    if candidate.family == "monomial-hop":
        return bloch_path_a_projector_free_layer(
            pattern_index=candidate.pattern_index,
            cycle=candidate.cycle,
            source_shifts=candidate.source_shifts,
        )
    if candidate.family == "polynomial-hop":
        return bloch_path_a_polynomial_hop_layer(
            pattern_index=candidate.pattern_index,
            terms_by_shift=candidate.polynomial_terms,
            mixes_by_shift=candidate.mixes_by_shift,
            name_suffix=candidate.name_suffix,
        )
    raise ValueError(f"unknown candidate family: {candidate.family}")


def _as_bloch_layer(layer: RuleLayerInput) -> BlochRuleLayerInput:
    return BlochRuleLayerInput(
        name=layer.name,
        period=12,
        dimension=layer.matrix.rows,
        terms=tuple(
            SpatialHoppingTerm(shift=term.shift, matrix=term.matrix) for term in layer.bloch_terms
        ),
    )


def _seed_guardrail_for_layer(
    layer: RuleLayerInput,
) -> tuple[bool, tuple[str, ...], tuple[str, ...], int]:
    bloch_layer = _as_bloch_layer(layer)
    guardrail, coefficient_algebra_dimension = bloch_seed_guardrail((bloch_layer,))
    return (
        guardrail.passed,
        guardrail.raw_seed_witnesses,
        guardrail.algebraic_seed_witnesses,
        coefficient_algebra_dimension,
    )


def _zero_matrix_like(matrix: sp.Matrix) -> sp.Matrix:
    return sp.zeros(matrix.rows, matrix.cols)


def _negation_closed_count(matrices: tuple[sp.Matrix, ...]) -> int:
    count = 0
    for matrix in matrices:
        if any(
            (matrix + other).applyfunc(sp.simplify) == _zero_matrix_like(matrix)
            for other in matrices
        ):
            count += 1
    return count


def _j_sign_shape(matrices: tuple[sp.Matrix, ...]) -> str:
    if not matrices:
        return "none"
    if len(matrices) == 2 and _negation_closed_count(matrices) == 2:
        return "global_pm"
    if _negation_closed_count(matrices) == len(matrices):
        return f"{len(matrices)}_with_pm_pairs"
    return f"{len(matrices)}_not_pm_closed"


def _bridge_j_status(
    *,
    generated_solved: bool | None,
    generated_count: int | None,
    generated_shape: str | None,
    compatible_solved: bool | None,
    compatible_count: int | None,
    compatible_shape: str | None,
) -> str | None:
    if generated_solved is None and compatible_solved is None:
        return None
    if generated_solved is False or compatible_solved is False:
        return "j_not_solved"
    if generated_count == 2 and generated_shape == "global_pm":
        if compatible_count == 2 and compatible_shape == "global_pm":
            return "global_pm_bridge_j_candidate"
        return "rule_generated_pm_j_compatible_extra_unresolved"
    if generated_count == 0 and compatible_count:
        return "compatible_j_not_rule_generated"
    if generated_count == 0:
        return "no_rule_generated_j"
    if generated_count and generated_count > 2:
        return "too_many_rule_generated_j"
    return "j_status_unclassified"


def evaluate_candidate(
    candidate: StepwiseCandidate,
    *,
    max_algebra_dim: int,
    include_seed_guardrail: bool,
    include_center: bool,
    include_centralizer: bool,
    include_idempotents: bool,
    include_j_solve: bool,
) -> StepwiseResult:
    start = time.perf_counter()
    layer = _layer_for_candidate(candidate)
    seed_guardrail_passed = True
    raw_seed_witnesses: tuple[str, ...] = ()
    algebraic_seed_witnesses: tuple[str, ...] = ()
    coefficient_algebra_dimension = None
    if include_seed_guardrail:
        (
            seed_guardrail_passed,
            raw_seed_witnesses,
            algebraic_seed_witnesses,
            coefficient_algebra_dimension,
        ) = _seed_guardrail_for_layer(layer)
    laurent_orthogonal = bloch_layer_laurent_orthogonal(_as_bloch_layer(layer))
    samples = bloch_floquet_operators((layer,), bloch_period=12)
    closure = generated_algebra_closure(samples, max_dimension=max_algebra_dim)
    center_dimension = None
    center_solved = None
    compatible_centralizer_dimension = None
    idempotent_ranks: tuple[int, ...] = ()
    generated_j_solved = None
    generated_j_moduli_dimension = None
    generated_j_count = None
    generated_j_sign_shape = None
    compatible_j_solved = None
    compatible_j_moduli_dimension = None
    compatible_j_count = None
    compatible_j_sign_shape = None
    center = ()
    compatible_basis = ()
    if (include_center or include_j_solve) and closure.closed:
        center = center_basis_of_algebra(closure.basis)
        center_dimension = len(center)
        if include_idempotents or include_j_solve:
            center_solved, idempotents = solve_central_idempotents(center)
            idempotent_ranks = tuple(item.rank for item in idempotents)
    if (include_centralizer or include_j_solve) and closure.closed:
        compatible_basis = centralizer_basis(samples)
        compatible_centralizer_dimension = len(compatible_basis)
    if include_j_solve and closure.closed:
        generated_j_solved, generated_j_moduli_dimension, generated_j = (
            solve_complex_structures_from_idempotent_splitting(
                center,
                idempotents,
                source="local_compatible_center",
            )
        )
        generated_j_count = len(generated_j)
        generated_j_sign_shape = _j_sign_shape(tuple(item.matrix for item in generated_j))
        compatible_j_solved, compatible_j_moduli_dimension, compatible_j = (
            solve_complex_structures_from_idempotent_splitting(
                compatible_basis,
                idempotents,
                source="compatible_centralizer",
            )
        )
        compatible_j_count = len(compatible_j)
        compatible_j_sign_shape = _j_sign_shape(tuple(item.matrix for item in compatible_j))

    return StepwiseResult(
        name=layer.name,
        family=candidate.family,
        pattern_index=candidate.pattern_index,
        cycle=candidate.cycle,
        source_shifts=candidate.source_shifts,
        seed_guardrail_checked=include_seed_guardrail,
        seed_guardrail_passed=seed_guardrail_passed,
        raw_seed_witnesses=raw_seed_witnesses,
        algebraic_seed_witnesses=algebraic_seed_witnesses,
        coefficient_algebra_dimension=coefficient_algebra_dimension,
        laurent_orthogonal=laurent_orthogonal,
        generated_algebra_dimension=len(closure.basis),
        generated_algebra_closed=closure.closed,
        center_dimension=center_dimension,
        center_solved=center_solved,
        compatible_centralizer_dimension=compatible_centralizer_dimension,
        central_idempotent_ranks=idempotent_ranks,
        generated_j_solved=generated_j_solved,
        generated_j_moduli_dimension=generated_j_moduli_dimension,
        generated_j_count=generated_j_count,
        generated_j_sign_shape=generated_j_sign_shape,
        compatible_j_solved=compatible_j_solved,
        compatible_j_moduli_dimension=compatible_j_moduli_dimension,
        compatible_j_count=compatible_j_count,
        compatible_j_sign_shape=compatible_j_sign_shape,
        bridge_j_status=_bridge_j_status(
            generated_solved=generated_j_solved,
            generated_count=generated_j_count,
            generated_shape=generated_j_sign_shape,
            compatible_solved=compatible_j_solved,
            compatible_count=compatible_j_count,
            compatible_shape=compatible_j_sign_shape,
        ),
        route_label=_label_result(
            closed=closure.closed,
            algebra_dimension=len(closure.basis),
            center_dimension=center_dimension,
            idempotent_ranks=idempotent_ranks,
        ),
        elapsed_seconds=time.perf_counter() - start,
    )


def _evaluate_for_pool(
    args: tuple[StepwiseCandidate, int, bool, bool, bool, bool, bool],
) -> StepwiseResult:
    (
        candidate,
        max_algebra_dim,
        include_seed_guardrail,
        include_center,
        include_centralizer,
        include_idempotents,
        include_j_solve,
    ) = args
    return evaluate_candidate(
        candidate,
        max_algebra_dim=max_algebra_dim,
        include_seed_guardrail=include_seed_guardrail,
        include_center=include_center,
        include_centralizer=include_centralizer,
        include_idempotents=include_idempotents,
        include_j_solve=include_j_solve,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run stepwise projector-free Bloch Path-A closure search."
    )
    parser.add_argument("--pattern-count", type=int, default=1)
    parser.add_argument(
        "--family", choices=("monomial-hop", "polynomial-hop"), default="monomial-hop"
    )
    parser.add_argument("--cycle-count", type=int, default=3)
    parser.add_argument("--shift-count", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-algebra-dim", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-guardrail", action="store_true")
    parser.add_argument("--center-top", type=int, default=0)
    parser.add_argument("--centralizer", action="store_true")
    parser.add_argument("--idempotents", action="store_true")
    parser.add_argument("--j-solve", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    candidates = stepwise_candidates(
        family=args.family,
        pattern_count=args.pattern_count,
        cycle_count=args.cycle_count,
        shift_count=args.shift_count,
        max_candidates=args.max_candidates,
    )
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")

    closure_args = [
        (candidate, args.max_algebra_dim, args.seed_guardrail, False, False, False, False)
        for candidate in candidates
    ]
    if args.jobs == 1:
        closure_results = tuple(_evaluate_for_pool(item) for item in closure_args)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            closure_results = tuple(executor.map(_evaluate_for_pool, closure_args))

    candidate_by_name = {_candidate_name(candidate): candidate for candidate in candidates}
    detailed_targets = []
    for result in closure_results:
        if len(detailed_targets) >= args.center_top:
            break
        if not result.generated_algebra_closed:
            continue
        detailed_targets.append(candidate_by_name[result.name])

    detailed_args = [
        (
            candidate,
            args.max_algebra_dim,
            args.seed_guardrail,
            True,
            args.centralizer,
            args.idempotents,
            args.j_solve,
        )
        for candidate in detailed_targets
    ]
    if not detailed_args:
        detailed_results = ()
    elif args.jobs == 1:
        detailed_results = tuple(_evaluate_for_pool(item) for item in detailed_args)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            detailed_results = tuple(executor.map(_evaluate_for_pool, detailed_args))
    detailed_by_name = {result.name: result for result in detailed_results}

    results = tuple(detailed_by_name.get(result.name, result) for result in closure_results)
    payload = {
        "candidate_count": len(results),
        "closed_count": sum(result.generated_algebra_closed for result in results),
        "seed_guardrail_checked": args.seed_guardrail,
        "seed_guardrail_rejections": sum(
            result.seed_guardrail_checked and not result.seed_guardrail_passed for result in results
        ),
        "laurent_orthogonal_count": sum(result.laurent_orthogonal for result in results),
        "coarse_6_4_count": sum(
            result.route_label == "closes_coarse_6_4_center" for result in results
        ),
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs a stepwise projector-free Bloch Path-A search.")
        print(f"family: {args.family}")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"closed_count: {payload['closed_count']}")
        print(f"seed_guardrail_checked: {str(args.seed_guardrail).lower()}")
        print(f"seed_guardrail_rejections: {payload['seed_guardrail_rejections']}")
        print(f"laurent_orthogonal_count: {payload['laurent_orthogonal_count']}")
        print(f"coarse_6_4_count: {payload['coarse_6_4_count']}")
        for result in results:
            seeded = (
                str(not result.seed_guardrail_passed).lower()
                if result.seed_guardrail_checked
                else "unchecked"
            )
            print(
                "candidate: "
                f"{result.name}, "
                f"seeded={seeded}, "
                f"laurent={str(result.laurent_orthogonal).lower()}, "
                f"dim={result.generated_algebra_dimension}, "
                f"closed={str(result.generated_algebra_closed).lower()}, "
                f"center={result.center_dimension}, "
                f"centralizer={result.compatible_centralizer_dimension}, "
                f"ranks={list(result.central_idempotent_ranks)}, "
                f"generated_j_solved={result.generated_j_solved}, "
                f"generated_j={result.generated_j_count}, "
                f"compatible_j_solved={result.compatible_j_solved}, "
                f"compatible_j={result.compatible_j_count}, "
                f"j_status={result.bridge_j_status}, "
                f"label={result.route_label}, "
                f"time={result.elapsed_seconds:.3f}s"
            )

    if args.check:
        if not results or any(result.generated_algebra_dimension <= 0 for result in results):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
