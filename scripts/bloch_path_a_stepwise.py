from __future__ import annotations

import argparse
import json
import signal
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.qca.bloch_rule import (
    BlochRuleLayerInput,
    bloch_layer_laurent_orthogonal,
    bloch_path_a_polynomial_hop_layer,
    bloch_path_a_projector_free_layer,
    bloch_seed_guardrail,
)
from clifford_3plus2_d5.qca.floquet_alpha import floquet_alpha_candidates
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    bloch_floquet_operators,
    center_basis_of_generated_algebra,
    centralizer_basis,
    generated_algebra_closure,
    solve_central_idempotent_rank_profile,
    solve_central_idempotents,
    solve_complex_structures_from_idempotent_splitting,
)
from clifford_3plus2_d5.qca.projected_centralizer import (
    ProjectedCentralizerBlockDiagnostic,
    ProjectedCentralizerPairDiagnostic,
    projected_centralizer_pair_diagnostics,
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
    projected_centralizer_pairs: tuple[ProjectedCentralizerPairDiagnostic, ...]
    route_label: str
    elapsed_seconds: float
    stage_seconds: tuple[tuple[str, float], ...]


SCANNER_VERSION = "bloch_path_a_stepwise_v6"
DEFAULT_BLOCH_PERIOD = 12
MAX_CYCLES = 24
MAX_SHIFT_ASSIGNMENTS = 10
FLOQUET_ALPHA_PATTERNS = floquet_alpha_candidates()


class _WorkerTimeout(RuntimeError):
    pass


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


def _available_count(requested: int, maximum: int) -> int:
    return min(requested, maximum)


def candidate_space_total(
    *,
    family: str,
    pattern_count: int,
    cycle_count: int,
    shift_count: int,
) -> int:
    cycles = _five_cycles(_available_count(cycle_count, MAX_CYCLES))
    shifts = _shift_assignments(_available_count(shift_count, MAX_SHIFT_ASSIGNMENTS))
    if family == "monomial-hop":
        return pattern_count * len(cycles) * len(shifts)
    if family != "polynomial-hop":
        raise ValueError(f"unknown family: {family}")
    total = 0
    for _pattern_index in range(pattern_count):
        for cycle in cycles:
            for source_shifts in shifts:
                edges_by_shift = _edges_by_shift(cycle, source_shifts)
                for shift in (3, 4):
                    edge_count = len(edges_by_shift.get(shift, ()))
                    if edge_count >= 2:
                        total += edge_count * (edge_count - 1) // 2
    return total


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


def _candidate_identity(candidate: StepwiseCandidate) -> dict[str, object]:
    return {
        "family": candidate.family,
        "pattern_index": candidate.pattern_index,
        "cycle": list(candidate.cycle),
        "source_shifts": list(candidate.source_shifts),
        "polynomial_terms": [
            [shift, [[source, target] for source, target in edges]]
            for shift, edges in candidate.polynomial_terms
        ],
        "mixes_by_shift": [
            [shift, [left, right]] for shift, (left, right) in candidate.mixes_by_shift
        ],
        "name_suffix": candidate.name_suffix,
    }


def _candidate_from_identity(payload: dict[str, object]) -> StepwiseCandidate:
    return StepwiseCandidate(
        family=str(payload["family"]),
        pattern_index=int(payload["pattern_index"]),
        cycle=tuple(int(item) for item in payload["cycle"]),  # type: ignore[index]
        source_shifts=tuple(int(item) for item in payload["source_shifts"]),  # type: ignore[index]
        polynomial_terms=tuple(
            (
                int(shift),
                tuple((int(source), int(target)) for source, target in edges),
            )
            for shift, edges in payload.get("polynomial_terms", ())  # type: ignore[union-attr]
        ),
        mixes_by_shift=tuple(
            (int(shift), (int(pair[0]), int(pair[1])))
            for shift, pair in payload.get("mixes_by_shift", ())  # type: ignore[union-attr]
        ),
        name_suffix=str(payload.get("name_suffix", "")),
    )


def _cache_key(
    candidate: StepwiseCandidate,
    *,
    max_algebra_dim: int,
    include_seed_guardrail: bool,
    include_center: bool,
    include_centralizer: bool,
    include_idempotents: bool,
    include_j_solve: bool,
    include_projected_centralizer: bool,
    coarse_only_diagnostics: bool,
    generated_j_only: bool,
) -> str:
    payload = {
        "version": SCANNER_VERSION,
        "candidate": _candidate_identity(candidate),
        "bloch_period": DEFAULT_BLOCH_PERIOD,
        "max_algebra_dim": max_algebra_dim,
        "stages": {
            "seed_guardrail": include_seed_guardrail,
            "center": include_center,
            "centralizer": include_centralizer,
            "idempotents": include_idempotents,
            "j_solve": include_j_solve,
            "projected_centralizer": include_projected_centralizer,
            "coarse_only_diagnostics": coarse_only_diagnostics,
            "generated_j_only": generated_j_only,
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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


def _monomial_dim26_no_locking_rank_profile(
    candidate: StepwiseCandidate,
    *,
    generated_algebra_dimension: int,
    center_dimension: int | None,
) -> tuple[int, ...]:
    """Return the dim-26 monomial-hop no-locking rank profile when structural.

    In the monomial Path-A census, the dim-26/center-4 targets are exactly the
    five-cycle hops with one alpha-alpha edge and four alpha/eta crossing
    edges. They carry a central rank-2/rank-8 split, so they are non-coarse
    before any J search. Some orientations express the central involution in a
    poor center basis and make the generic idempotent solver stall; this
    combinatorial certificate avoids that algebraic-basis artifact.
    """

    if (
        candidate.family != "monomial-hop"
        or generated_algebra_dimension != 26
        or center_dimension != 4
        or not (0 <= candidate.pattern_index < len(FLOQUET_ALPHA_PATTERNS))
    ):
        return ()
    alpha_modes = set(FLOQUET_ALPHA_PATTERNS[candidate.pattern_index].alpha_modes)
    alpha_alpha_edges = 0
    eta_eta_edges = 0
    crossing_edges = 0
    for source, target in enumerate(candidate.cycle):
        source_is_alpha = source in alpha_modes
        target_is_alpha = target in alpha_modes
        if source_is_alpha and target_is_alpha:
            alpha_alpha_edges += 1
        elif not source_is_alpha and not target_is_alpha:
            eta_eta_edges += 1
        else:
            crossing_edges += 1
    if alpha_alpha_edges == 1 and eta_eta_edges == 0 and crossing_edges == 4:
        return (0, 2, 8, 10)
    return ()


def _monomial_alpha_eta_edge_counts(candidate: StepwiseCandidate) -> tuple[int, int, int] | None:
    if (
        candidate.family != "monomial-hop"
        or not (0 <= candidate.pattern_index < len(FLOQUET_ALPHA_PATTERNS))
    ):
        return None
    alpha_modes = set(FLOQUET_ALPHA_PATTERNS[candidate.pattern_index].alpha_modes)
    alpha_alpha_edges = 0
    eta_eta_edges = 0
    crossing_edges = 0
    for source, target in enumerate(candidate.cycle):
        source_is_alpha = source in alpha_modes
        target_is_alpha = target in alpha_modes
        if source_is_alpha and target_is_alpha:
            alpha_alpha_edges += 1
        elif not source_is_alpha and not target_is_alpha:
            eta_eta_edges += 1
        else:
            crossing_edges += 1
    return alpha_alpha_edges, eta_eta_edges, crossing_edges


def _monomial_dim34_coarse_rank_profile(
    candidate: StepwiseCandidate,
    *,
    generated_algebra_dimension: int,
    center_dimension: int | None,
) -> tuple[int, ...]:
    """Return the dim-34 monomial-hop coarse split when structural.

    The completed center census shows that dim-34/center-4 monomial-hop
    targets occur in exactly two alpha/eta cycle geometries: either one
    alpha-alpha edge plus four crossing edges, or two alpha-alpha edges, two
    crossing edges, and one eta-eta edge. In both cases the center is the
    coarse rank-(6,4) split. The generic idempotent equations can stall on
    some basis orientations of this same algebra; this certificate keeps the
    brute-force scan on the structural path.
    """

    if (
        candidate.family != "monomial-hop"
        or generated_algebra_dimension != 34
        or center_dimension != 4
    ):
        return ()
    edge_counts = _monomial_alpha_eta_edge_counts(candidate)
    if edge_counts in ((1, 0, 4), (2, 1, 2)):
        return (0, 4, 6, 10)
    return ()


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


def _reduce_analysis_generators(
    generators: Sequence[sp.Matrix],
    closure_basis: Sequence[sp.Matrix],
) -> tuple[sp.Matrix, ...]:
    """Find a small sample subset that still generates the closed algebra."""

    if not generators:
        return ()
    target_dimension = len(closure_basis)
    selected: list[sp.Matrix] = []
    selected_dimension = 1
    for generator in generators:
        trial = (*selected, generator)
        closure = generated_algebra_closure(
            trial,
            dimension=generator.rows,
            max_dimension=target_dimension + 1,
        )
        if closure.closed and len(closure.basis) == target_dimension:
            return trial
        if len(closure.basis) > selected_dimension:
            selected.append(generator)
            selected_dimension = len(closure.basis)
    return tuple(generators)


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


def _projected_centralizer_summary(
    pairs: tuple[ProjectedCentralizerPairDiagnostic, ...],
) -> str:
    if not pairs:
        return "not_checked"
    pair_summaries = []
    for pair in pairs:
        block_summaries = []
        for block in pair.blocks:
            block_summaries.append(
                "rank"
                f"{block.projector_rank}:"
                f"{block.classification}"
                f"/dim{block.projected_dimension}"
                f"/components{list(block.primitive_component_types)}"
            )
        pair_summaries.append(";".join(block_summaries))
    return "|".join(pair_summaries)


def _has_complex_projected_factor(result: StepwiseResult) -> bool:
    return any(
        block.contains_complex_factor
        for pair in result.projected_centralizer_pairs
        for block in pair.blocks
    )


def _is_split_real_projected(result: StepwiseResult) -> bool:
    if not result.projected_centralizer_pairs:
        return False
    return all(
        block.classification == "split_real"
        for pair in result.projected_centralizer_pairs
        for block in pair.blocks
    )


def _classify_result(result: StepwiseResult) -> str:
    if result.route_label == "candidate_timeout":
        return "timeout"
    if result.seed_guardrail_checked and not result.seed_guardrail_passed:
        return "seeded_rejected"
    if not result.generated_algebra_closed:
        return "cap_exceeded"
    if result.center_solved is False:
        return "solver_bottleneck"
    if result.central_idempotent_ranks and not (
        4 in result.central_idempotent_ranks and 6 in result.central_idempotent_ranks
    ):
        return "non_coarse"
    if result.center_dimension is not None and not result.central_idempotent_ranks:
        return "solver_bottleneck"
    if _has_complex_projected_factor(result):
        return "c_factor"
    if _is_split_real_projected(result):
        return "split_real"
    if result.projected_centralizer_pairs:
        return "projected_unclassified"
    if result.route_label == "closes_coarse_6_4_center":
        return "coarse_unclassified"
    return result.route_label


def _format_result_line(result: StepwiseResult) -> str:
    seeded = (
        str(not result.seed_guardrail_passed).lower()
        if result.seed_guardrail_checked
        else "unchecked"
    )
    return (
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
        f"projected={_projected_centralizer_summary(result.projected_centralizer_pairs)}, "
        f"class={_classify_result(result)}, "
        f"label={result.route_label}, "
        f"time={result.elapsed_seconds:.3f}s"
    )


def _block_from_dict(payload: dict[str, object]) -> ProjectedCentralizerBlockDiagnostic:
    return ProjectedCentralizerBlockDiagnostic(
        projector_rank=int(payload["projector_rank"]),
        projected_dimension=int(payload["projected_dimension"]),
        multiplication_table_solved=bool(payload["multiplication_table_solved"]),
        commutative=bool(payload["commutative"]),
        idempotents_solved=bool(payload["idempotents_solved"]),
        idempotent_ranks=tuple(int(item) for item in payload["idempotent_ranks"]),  # type: ignore[index]
        primitive_component_dimensions=tuple(
            int(item) for item in payload["primitive_component_dimensions"]  # type: ignore[index]
        ),
        primitive_component_types=tuple(
            str(item) for item in payload["primitive_component_types"]  # type: ignore[index]
        ),
        primitive_components_sum_to_projector=bool(
            payload["primitive_components_sum_to_projector"]
        ),
        contains_complex_factor=bool(payload["contains_complex_factor"]),
        classification=str(payload["classification"]),
    )


def _pair_from_dict(payload: dict[str, object]) -> ProjectedCentralizerPairDiagnostic:
    return ProjectedCentralizerPairDiagnostic(
        pair_ranks=tuple(int(item) for item in payload["pair_ranks"]),  # type: ignore[index]
        blocks=tuple(
            _block_from_dict(block) for block in payload["blocks"]  # type: ignore[index]
        ),
    )


def _result_from_dict(payload: dict[str, object]) -> StepwiseResult:
    return StepwiseResult(
        name=str(payload["name"]),
        family=str(payload["family"]),
        pattern_index=int(payload["pattern_index"]),
        cycle=tuple(int(item) for item in payload["cycle"]),  # type: ignore[index]
        source_shifts=tuple(int(item) for item in payload["source_shifts"]),  # type: ignore[index]
        seed_guardrail_checked=bool(payload["seed_guardrail_checked"]),
        seed_guardrail_passed=bool(payload["seed_guardrail_passed"]),
        raw_seed_witnesses=tuple(str(item) for item in payload["raw_seed_witnesses"]),  # type: ignore[index]
        algebraic_seed_witnesses=tuple(
            str(item) for item in payload["algebraic_seed_witnesses"]  # type: ignore[index]
        ),
        coefficient_algebra_dimension=(
            None
            if payload["coefficient_algebra_dimension"] is None
            else int(payload["coefficient_algebra_dimension"])
        ),
        laurent_orthogonal=bool(payload["laurent_orthogonal"]),
        generated_algebra_dimension=int(payload["generated_algebra_dimension"]),
        generated_algebra_closed=bool(payload["generated_algebra_closed"]),
        center_dimension=(
            None if payload["center_dimension"] is None else int(payload["center_dimension"])
        ),
        center_solved=(
            None if payload["center_solved"] is None else bool(payload["center_solved"])
        ),
        compatible_centralizer_dimension=(
            None
            if payload["compatible_centralizer_dimension"] is None
            else int(payload["compatible_centralizer_dimension"])
        ),
        central_idempotent_ranks=tuple(
            int(item) for item in payload["central_idempotent_ranks"]  # type: ignore[index]
        ),
        generated_j_solved=(
            None
            if payload["generated_j_solved"] is None
            else bool(payload["generated_j_solved"])
        ),
        generated_j_moduli_dimension=(
            None
            if payload["generated_j_moduli_dimension"] is None
            else int(payload["generated_j_moduli_dimension"])
        ),
        generated_j_count=(
            None if payload["generated_j_count"] is None else int(payload["generated_j_count"])
        ),
        generated_j_sign_shape=(
            None
            if payload["generated_j_sign_shape"] is None
            else str(payload["generated_j_sign_shape"])
        ),
        compatible_j_solved=(
            None
            if payload["compatible_j_solved"] is None
            else bool(payload["compatible_j_solved"])
        ),
        compatible_j_moduli_dimension=(
            None
            if payload["compatible_j_moduli_dimension"] is None
            else int(payload["compatible_j_moduli_dimension"])
        ),
        compatible_j_count=(
            None
            if payload["compatible_j_count"] is None
            else int(payload["compatible_j_count"])
        ),
        compatible_j_sign_shape=(
            None
            if payload["compatible_j_sign_shape"] is None
            else str(payload["compatible_j_sign_shape"])
        ),
        bridge_j_status=(
            None if payload["bridge_j_status"] is None else str(payload["bridge_j_status"])
        ),
        projected_centralizer_pairs=tuple(
            _pair_from_dict(pair) for pair in payload["projected_centralizer_pairs"]  # type: ignore[index]
        ),
        route_label=str(payload["route_label"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        stage_seconds=tuple(
            (str(stage), float(seconds))
            for stage, seconds in payload.get("stage_seconds", ())  # type: ignore[union-attr]
        ),
    )


def _load_cache(path: Path | None) -> dict[str, StepwiseResult]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, StepwiseResult] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            key = str(payload["cache_key"])
            cache[key] = _result_from_dict(payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return cache


def _target_candidates_from_cache(
    path: Path,
    *,
    family: str,
    center_dimensions: set[int] | None,
    generated_dimensions: set[int] | None,
    route_labels: set[str] | None,
    require_closed: bool,
) -> tuple[StepwiseCandidate, ...]:
    """Load detailed-scan targets from a previous center-census JSONL cache.

    The center census already paid the closure and center-basis cost for the
    whole candidate space. Reusing it as a target index avoids a second
    closure-only discovery pass before expensive idempotent/J diagnostics.
    """

    if not path.exists():
        raise FileNotFoundError(f"target cache file does not exist: {path}")

    candidates: list[StepwiseCandidate] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            key_payload = json.loads(str(payload["cache_key"]))
            candidate_payload = key_payload["candidate"]
            result = _result_from_dict(payload["result"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if result.family != family:
            continue
        if require_closed and not result.generated_algebra_closed:
            continue
        if center_dimensions is not None and result.center_dimension not in center_dimensions:
            continue
        if (
            generated_dimensions is not None
            and result.generated_algebra_dimension not in generated_dimensions
        ):
            continue
        if route_labels is not None and result.route_label not in route_labels:
            continue
        candidate = _candidate_from_identity(candidate_payload)
        identity_key = json.dumps(_candidate_identity(candidate), sort_keys=True)
        if identity_key in seen:
            continue
        seen.add(identity_key)
        candidates.append(candidate)
    return tuple(candidates)


def _interleaved_candidates(
    candidates: Sequence[StepwiseCandidate],
    *,
    stride: int,
) -> tuple[StepwiseCandidate, ...]:
    if stride <= 1:
        return tuple(candidates)
    return tuple(
        candidate
        for offset in range(stride)
        for candidate in candidates[offset::stride]
    )


def _append_cache(path: Path | None, cache_key: str, result: StepwiseResult) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "scanner_version": SCANNER_VERSION,
        "result": asdict(result),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _stage_average_seconds(results: Sequence[StepwiseResult]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for stage, seconds in result.stage_seconds:
            totals[stage] = totals.get(stage, 0.0) + seconds
            counts[stage] = counts.get(stage, 0) + 1
    return {
        stage: round(total / counts[stage], 6)
        for stage, total in sorted(totals.items())
        if counts[stage]
    }


def _summary_payload(
    results: Sequence[StepwiseResult],
    *,
    total_candidates: int,
    selected_count: int,
    cached_count: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    class_counter = Counter(_classify_result(result) for result in results)
    slowest = sorted(results, key=lambda item: item.elapsed_seconds, reverse=True)[:5]
    rate = len(results) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    pending = max(selected_count - len(results), 0)
    eta = pending / rate if rate > 0 else None
    return {
        "scanner_version": SCANNER_VERSION,
        "total_candidate_space": total_candidates,
        "selected_count": selected_count,
        "completed_count": len(results),
        "cached_count": cached_count,
        "pending_count": pending,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "candidates_per_second": round(rate, 6),
        "eta_seconds": None if eta is None else round(eta, 3),
        "class_counts": dict(sorted(class_counter.items())),
        "stage_average_seconds": _stage_average_seconds(results),
        "slowest_candidates": [
            {
                "name": item.name,
                "elapsed_seconds": round(item.elapsed_seconds, 3),
                "class": _classify_result(item),
                "ranks": list(item.central_idempotent_ranks),
            }
            for item in slowest
        ],
    }


def _format_progress(
    *,
    completed: int,
    total: int,
    cached_count: int,
    start: float,
    results: Sequence[StepwiseResult],
) -> str:
    elapsed = time.perf_counter() - start
    rate = completed / elapsed if elapsed > 0 else 0.0
    pending = max(total - completed, 0)
    eta = pending / rate if rate > 0 else None
    class_counts = Counter(_classify_result(result) for result in results)
    eta_text = "unknown" if eta is None else f"{eta:.1f}s"
    return (
        "progress: "
        f"completed={completed}/{total}, "
        f"cached={cached_count}, "
        f"pending={pending}, "
        f"rate={rate:.4f}/s, "
        f"eta={eta_text}, "
        f"classes={dict(sorted(class_counts.items()))}"
    )


def evaluate_candidate(
    candidate: StepwiseCandidate,
    *,
    max_algebra_dim: int,
    include_seed_guardrail: bool,
    include_center: bool,
    include_centralizer: bool,
    include_idempotents: bool,
    include_j_solve: bool,
    include_projected_centralizer: bool,
    coarse_only_diagnostics: bool,
    generated_j_only: bool,
) -> StepwiseResult:
    start = time.perf_counter()
    stage_seconds: list[tuple[str, float]] = []

    def record_stage(stage: str, stage_start: float) -> None:
        stage_seconds.append((stage, time.perf_counter() - stage_start))

    stage_start = time.perf_counter()
    layer = _layer_for_candidate(candidate)
    record_stage("layer", stage_start)

    seed_guardrail_passed = True
    raw_seed_witnesses: tuple[str, ...] = ()
    algebraic_seed_witnesses: tuple[str, ...] = ()
    coefficient_algebra_dimension = None
    if include_seed_guardrail:
        stage_start = time.perf_counter()
        (
            seed_guardrail_passed,
            raw_seed_witnesses,
            algebraic_seed_witnesses,
            coefficient_algebra_dimension,
        ) = _seed_guardrail_for_layer(layer)
        record_stage("seed_guardrail", stage_start)

    stage_start = time.perf_counter()
    bloch_layer = _as_bloch_layer(layer)
    record_stage("bloch_layer", stage_start)

    stage_start = time.perf_counter()
    laurent_orthogonal = bloch_layer_laurent_orthogonal(bloch_layer)
    record_stage("laurent", stage_start)

    stage_start = time.perf_counter()
    samples = bloch_floquet_operators((layer,), bloch_period=DEFAULT_BLOCH_PERIOD)
    record_stage("bloch_samples", stage_start)

    stage_start = time.perf_counter()
    closure = generated_algebra_closure(samples, max_dimension=max_algebra_dim)
    record_stage("closure", stage_start)

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
    projected_centralizer_pairs: tuple[ProjectedCentralizerPairDiagnostic, ...] = ()
    center = ()
    compatible_basis = ()
    idempotents = ()
    analysis_generators = samples
    has_rank_6_4_blocks = False
    structural_generated_j_absent = False
    if (
        include_center
        or include_centralizer
        or include_j_solve
        or include_projected_centralizer
    ) and closure.closed:
        stage_start = time.perf_counter()
        analysis_generators = _reduce_analysis_generators(samples, closure.basis)
        record_stage("generator_reduction", stage_start)

    if (include_center or include_j_solve or include_projected_centralizer) and closure.closed:
        stage_start = time.perf_counter()
        center = center_basis_of_generated_algebra(closure.basis, analysis_generators)
        center_dimension = len(center)
        record_stage("center_basis", stage_start)
        if include_idempotents or include_j_solve or include_projected_centralizer:
            stage_start = time.perf_counter()
            rank_profile_solved = False
            structural_rank_profile = ()
            if coarse_only_diagnostics:
                structural_rank_profile = _monomial_dim26_no_locking_rank_profile(
                    candidate,
                    generated_algebra_dimension=len(closure.basis),
                    center_dimension=center_dimension,
                )
                if (
                    not structural_rank_profile
                    and generated_j_only
                    and not include_projected_centralizer
                ):
                    structural_rank_profile = _monomial_dim34_coarse_rank_profile(
                        candidate,
                        generated_algebra_dimension=len(closure.basis),
                        center_dimension=center_dimension,
                    )
                    structural_generated_j_absent = bool(structural_rank_profile)
            if structural_rank_profile:
                center_solved = True
                idempotent_ranks = structural_rank_profile
            elif coarse_only_diagnostics and len(closure.basis) <= 26:
                rank_profile_solved, rank_profile = solve_central_idempotent_rank_profile(center)
                if rank_profile_solved and not (4 in rank_profile and 6 in rank_profile):
                    center_solved = True
                    idempotent_ranks = rank_profile
                else:
                    center_solved, idempotents = solve_central_idempotents(center)
                    idempotent_ranks = tuple(item.rank for item in idempotents)
            else:
                center_solved, idempotents = solve_central_idempotents(center)
                idempotent_ranks = tuple(item.rank for item in idempotents)
            has_rank_6_4_blocks = 4 in idempotent_ranks and 6 in idempotent_ranks
            record_stage("idempotents", stage_start)
    needs_compatible_basis = (
        (
            include_centralizer
            and not generated_j_only
            and (not coarse_only_diagnostics or has_rank_6_4_blocks)
        )
        or (
            include_j_solve
            and not generated_j_only
            and (not coarse_only_diagnostics or has_rank_6_4_blocks)
        )
        or (include_projected_centralizer and has_rank_6_4_blocks)
    )
    if needs_compatible_basis and closure.closed:
        stage_start = time.perf_counter()
        compatible_basis = centralizer_basis(analysis_generators)
        compatible_centralizer_dimension = len(compatible_basis)
        record_stage("centralizer", stage_start)
    if (
        include_projected_centralizer
        and closure.closed
        and center_solved
        and compatible_basis
    ):
        stage_start = time.perf_counter()
        projected_centralizer_pairs = projected_centralizer_pair_diagnostics(
            compatible_basis,
            idempotents,
        )
        record_stage("projected_centralizer", stage_start)
    if include_j_solve and closure.closed and (not coarse_only_diagnostics or has_rank_6_4_blocks):
        stage_start = time.perf_counter()
        if structural_generated_j_absent and generated_j_only:
            generated_j_solved = True
            generated_j_moduli_dimension = 0
            generated_j_count = 0
            generated_j_sign_shape = "none"
        else:
            generated_j_solved, generated_j_moduli_dimension, generated_j = (
                solve_complex_structures_from_idempotent_splitting(
                    center,
                    idempotents,
                    source="local_compatible_center",
                )
            )
            generated_j_count = len(generated_j)
            generated_j_sign_shape = _j_sign_shape(tuple(item.matrix for item in generated_j))
        if not generated_j_only:
            compatible_j_solved, compatible_j_moduli_dimension, compatible_j = (
                solve_complex_structures_from_idempotent_splitting(
                    compatible_basis,
                    idempotents,
                    source="compatible_centralizer",
                )
            )
            compatible_j_count = len(compatible_j)
            compatible_j_sign_shape = _j_sign_shape(tuple(item.matrix for item in compatible_j))
        record_stage("j_solve", stage_start)

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
        projected_centralizer_pairs=projected_centralizer_pairs,
        route_label=_label_result(
            closed=closure.closed,
            algebra_dimension=len(closure.basis),
            center_dimension=center_dimension,
            idempotent_ranks=idempotent_ranks,
        ),
        elapsed_seconds=time.perf_counter() - start,
        stage_seconds=tuple((stage, round(seconds, 6)) for stage, seconds in stage_seconds),
    )


def _timeout_result(
    args: EvalArgs,
    *,
    timeout_seconds: float,
    elapsed_seconds: float,
) -> StepwiseResult:
    (
        candidate,
        _max_algebra_dim,
        include_seed_guardrail,
        _include_center,
        _include_centralizer,
        _include_idempotents,
        _include_j_solve,
        _include_projected_centralizer,
        _coarse_only_diagnostics,
        _generated_j_only,
        _timeout_seconds,
    ) = args
    return StepwiseResult(
        name=_candidate_name(candidate),
        family=candidate.family,
        pattern_index=candidate.pattern_index,
        cycle=candidate.cycle,
        source_shifts=candidate.source_shifts,
        seed_guardrail_checked=include_seed_guardrail,
        seed_guardrail_passed=False,
        raw_seed_witnesses=(),
        algebraic_seed_witnesses=(),
        coefficient_algebra_dimension=None,
        laurent_orthogonal=False,
        generated_algebra_dimension=0,
        generated_algebra_closed=False,
        center_dimension=None,
        center_solved=None,
        compatible_centralizer_dimension=None,
        central_idempotent_ranks=(),
        generated_j_solved=None,
        generated_j_moduli_dimension=None,
        generated_j_count=None,
        generated_j_sign_shape=None,
        compatible_j_solved=None,
        compatible_j_moduli_dimension=None,
        compatible_j_count=None,
        compatible_j_sign_shape=None,
        bridge_j_status="timeout",
        projected_centralizer_pairs=(),
        route_label="candidate_timeout",
        elapsed_seconds=elapsed_seconds,
        stage_seconds=(("timeout", round(timeout_seconds, 6)),),
    )


EvalArgs = tuple[
    StepwiseCandidate,
    int,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    float | None,
]
CachedEvalArgs = tuple[str, EvalArgs]


def _evaluate_many(
    work_items: Sequence[CachedEvalArgs],
    *,
    jobs: int,
    chunk_size: int,
    timeout_seconds: float | None,
    stream: bool,
    stop_on_complex_factor: bool,
    cache: dict[str, StepwiseResult],
    cache_file: Path | None,
    use_cache: bool,
    progress_every: int,
) -> tuple[tuple[StepwiseResult, ...], int]:
    if not work_items:
        return (), 0

    start = time.perf_counter()
    cached_count = 0
    indexed_results: dict[int, StepwiseResult] = {}
    pending_items: list[tuple[int, str, EvalArgs]] = []
    for index, (cache_key, item) in enumerate(work_items):
        if use_cache and cache_key in cache:
            indexed_results[index] = cache[cache_key]
            cached_count += 1
        else:
            pending_items.append((index, cache_key, item))

    completed = len(indexed_results)
    if progress_every > 0 and completed:
        print(
            _format_progress(
                completed=completed,
                total=len(work_items),
                cached_count=cached_count,
                start=start,
                results=tuple(indexed_results.values()),
            ),
            flush=True,
        )

    def store_result(index: int, cache_key: str, result: StepwiseResult) -> bool:
        indexed_results[index] = result
        cache[cache_key] = result
        _append_cache(cache_file, cache_key, result)
        if stream:
            print(_format_result_line(result), flush=True)
        completed_now = len(indexed_results)
        if progress_every > 0 and (
            completed_now == len(work_items) or completed_now % progress_every == 0
        ):
            print(
                _format_progress(
                    completed=completed_now,
                    total=len(work_items),
                    cached_count=cached_count,
                    start=start,
                    results=tuple(indexed_results.values()),
                ),
                flush=True,
            )
        return stop_on_complex_factor and _has_complex_projected_factor(result)

    if jobs == 1 and timeout_seconds is None:
        for index, cache_key, item in pending_items:
            result = _evaluate_for_pool(item)
            if store_result(index, cache_key, result):
                break
        return tuple(indexed_results[index] for index in sorted(indexed_results)), cached_count

    if stream:
        pending_iterator = iter(pending_items)

        def submit_next(
            executor: ProcessPoolExecutor,
            future_by_index: dict[Future[StepwiseResult], int],
            cache_key_by_future: dict[Future[StepwiseResult], str],
            submitted_at: dict[Future[StepwiseResult], float],
        ) -> Future[StepwiseResult] | None:
            try:
                index, cache_key, item = next(pending_iterator)
            except StopIteration:
                return None
            future = executor.submit(_evaluate_for_pool, item)
            future_by_index[future] = index
            cache_key_by_future[future] = cache_key
            submitted_at[future] = time.perf_counter()
            return future

        stopped_early = False
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_by_index: dict[Future[StepwiseResult], int] = {}
            cache_key_by_future: dict[Future[StepwiseResult], str] = {}
            submitted_at: dict[Future[StepwiseResult], float] = {}
            for _ in range(min(jobs, len(pending_items))):
                submit_next(executor, future_by_index, cache_key_by_future, submitted_at)
            pending = set(future_by_index)
            while pending:
                done, pending = wait(
                    pending,
                    timeout=1.0 if timeout_seconds is not None else None,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    index = future_by_index.pop(future)
                    cache_key = cache_key_by_future.pop(future)
                    submitted_at.pop(future, None)
                    result = future.result()
                    if store_result(index, cache_key, result):
                        stopped_early = True
                    if stopped_early:
                        for pending_future in pending:
                            pending_future.cancel()
                        executor.shutdown(cancel_futures=True)
                        break
                    next_future = submit_next(
                        executor,
                        future_by_index,
                        cache_key_by_future,
                        submitted_at,
                    )
                    if next_future is not None:
                        pending.add(next_future)
                if stopped_early:
                    break
        return tuple(indexed_results[index] for index in sorted(indexed_results)), cached_count

    stopped_early = False
    effective_chunk_size = chunk_size if chunk_size > 0 else len(pending_items)
    for chunk_start in range(0, len(pending_items), effective_chunk_size):
        if stopped_early:
            break
        chunk = pending_items[chunk_start : chunk_start + effective_chunk_size]
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_by_index: dict[Future[StepwiseResult], int] = {}
            cache_key_by_future: dict[Future[StepwiseResult], str] = {}
            submitted_at: dict[Future[StepwiseResult], float] = {}
            for index, cache_key, item in chunk:
                future = executor.submit(_evaluate_for_pool, item)
                future_by_index[future] = index
                cache_key_by_future[future] = cache_key
                submitted_at[future] = time.perf_counter()
            pending = set(future_by_index)
            while pending:
                if stream or stop_on_complex_factor:
                    wait_timeout = 1.0 if timeout_seconds is not None else None
                    done, pending = wait(
                        pending,
                        timeout=wait_timeout,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    if timeout_seconds is None:
                        done = set(as_completed(pending))
                        pending = set()
                    else:
                        done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if timeout_seconds is not None and not done:
                    continue
                for future in done:
                    index = future_by_index[future]
                    cache_key = cache_key_by_future[future]
                    result = future.result()
                    if store_result(index, cache_key, result):
                        stopped_early = True
                    if stopped_early:
                        for pending_future in pending:
                            pending_future.cancel()
                        executor.shutdown(cancel_futures=True)
                        break
    return tuple(indexed_results[index] for index in sorted(indexed_results)), cached_count


def _evaluate_for_pool(
    args: EvalArgs,
) -> StepwiseResult:
    timeout_seconds = args[-1]
    start = time.perf_counter()
    old_handler = None
    old_timer: tuple[float, float] | None = None

    def handle_timeout(_signum: int, _frame: object) -> None:
        raise _WorkerTimeout()

    if timeout_seconds is not None:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, handle_timeout)
        old_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return _evaluate_for_pool_unbounded(args)
    except _WorkerTimeout:
        return _timeout_result(
            args,
            timeout_seconds=float(timeout_seconds or 0),
            elapsed_seconds=time.perf_counter() - start,
        )
    finally:
        if timeout_seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
            if old_timer is not None and old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _evaluate_for_pool_unbounded(
    args: EvalArgs,
) -> StepwiseResult:
    (
        candidate,
        max_algebra_dim,
        include_seed_guardrail,
        include_center,
        include_centralizer,
        include_idempotents,
        include_j_solve,
        include_projected_centralizer,
        coarse_only_diagnostics,
        generated_j_only,
        _timeout_seconds,
    ) = args
    return evaluate_candidate(
        candidate,
        max_algebra_dim=max_algebra_dim,
        include_seed_guardrail=include_seed_guardrail,
        include_center=include_center,
        include_centralizer=include_centralizer,
        include_idempotents=include_idempotents,
        include_j_solve=include_j_solve,
        include_projected_centralizer=include_projected_centralizer,
        coarse_only_diagnostics=coarse_only_diagnostics,
        generated_j_only=generated_j_only,
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--all-candidates", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-algebra-dim", type=int, default=48)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--seed-guardrail", action="store_true")
    parser.add_argument("--center-top", type=int, default=0)
    parser.add_argument("--centralizer", action="store_true")
    parser.add_argument("--idempotents", action="store_true")
    parser.add_argument("--j-solve", action="store_true")
    parser.add_argument("--projected-centralizer", action="store_true")
    parser.add_argument(
        "--coarse-only-diagnostics",
        action="store_true",
        help="Short-circuit detailed diagnostics once the central ranks are not 6+4.",
    )
    parser.add_argument(
        "--generated-j-only",
        action="store_true",
        help="For J diagnostics, solve only rule-generated local-center J candidates.",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stop-on-complex-factor", action="store_true")
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--target-cache-file",
        type=Path,
        default=None,
        help="Select detailed targets from a previous JSONL center-census cache.",
    )
    parser.add_argument(
        "--target-center-dim",
        type=int,
        action="append",
        default=None,
        help="Keep only cached targets with this center dimension; repeatable.",
    )
    parser.add_argument(
        "--target-generated-dim",
        type=int,
        action="append",
        default=None,
        help="Keep only cached targets with this generated algebra dimension; repeatable.",
    )
    parser.add_argument(
        "--target-route-label",
        action="append",
        default=None,
        help="Keep only cached targets with this route label; repeatable.",
    )
    parser.add_argument("--target-start-index", type=int, default=0)
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument(
        "--target-interleave-stride",
        type=int,
        default=1,
        help="Interleave selected cached targets by this stride before evaluation.",
    )
    parser.add_argument(
        "--target-include-open",
        action="store_true",
        help="Do not require cached targets to have generated_algebra_closed=true.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError("start-index must be nonnegative")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be nonnegative")
    if args.target_start_index < 0:
        raise ValueError("target-start-index must be nonnegative")
    if args.target_limit is not None and args.target_limit < 0:
        raise ValueError("target-limit must be nonnegative")
    if args.target_interleave_stride <= 0:
        raise ValueError("target-interleave-stride must be positive")
    total_candidates = candidate_space_total(
        family=args.family,
        pattern_count=args.pattern_count,
        cycle_count=args.cycle_count,
        shift_count=args.shift_count,
    )
    target_matching_count = None
    if args.target_cache_file is not None:
        target_candidates = _target_candidates_from_cache(
            args.target_cache_file,
            family=args.family,
            center_dimensions=(
                None if args.target_center_dim is None else set(args.target_center_dim)
            ),
            generated_dimensions=(
                None if args.target_generated_dim is None else set(args.target_generated_dim)
            ),
            route_labels=(
                None if args.target_route_label is None else set(args.target_route_label)
            ),
            require_closed=not args.target_include_open,
        )
        target_matching_count = len(target_candidates)
        target_requested = (
            max(target_matching_count - args.target_start_index, 0)
            if args.target_limit is None
            else args.target_limit
        )
        candidates = target_candidates[
            args.target_start_index : args.target_start_index + target_requested
        ]
        candidates = _interleaved_candidates(
            candidates,
            stride=args.target_interleave_stride,
        )
    else:
        if args.all_candidates:
            requested_count = max(total_candidates - args.start_index, 0)
        elif args.limit is not None:
            requested_count = args.limit
        else:
            requested_count = args.max_candidates
        candidates = stepwise_candidates(
            family=args.family,
            pattern_count=args.pattern_count,
            cycle_count=args.cycle_count,
            shift_count=args.shift_count,
            max_candidates=args.start_index + requested_count,
        )[args.start_index :]
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if args.chunk_size < 0:
        raise ValueError("chunk-size must be nonnegative")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")

    run_start = time.perf_counter()
    cache = _load_cache(args.cache_file) if args.resume else {}

    if args.count_only or args.dry_run:
        payload = {
            "scanner_version": SCANNER_VERSION,
            "family": args.family,
            "total_candidate_space": total_candidates,
            "start_index": args.start_index,
            "target_cache_file": None
            if args.target_cache_file is None
            else str(args.target_cache_file),
            "target_matching_count": target_matching_count,
            "target_start_index": args.target_start_index,
            "target_limit": args.target_limit,
            "target_interleave_stride": args.target_interleave_stride,
            "selected_count": len(candidates),
            "candidate_names": [_candidate_name(candidate) for candidate in candidates]
            if args.dry_run
            else [],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"family: {args.family}")
            print(f"total_candidate_space: {total_candidates}")
            print(f"start_index: {args.start_index}")
            if args.target_cache_file is not None:
                print(f"target_cache_file: {args.target_cache_file}")
                print(f"target_matching_count: {target_matching_count}")
                print(f"target_start_index: {args.target_start_index}")
            print(f"selected_count: {len(candidates)}")
            if args.dry_run:
                display_start = (
                    args.target_start_index
                    if args.target_cache_file is not None
                    else args.start_index
                )
                for index, candidate in enumerate(candidates, start=display_start):
                    print(f"candidate[{index}]: {_candidate_name(candidate)}")
        return 0

    if args.target_cache_file is not None:
        closure_results = ()
        detailed_targets = list(candidates)
    elif args.center_top >= len(candidates):
        closure_results = ()
        detailed_targets = list(candidates)
    else:
        closure_work_items = []
        for candidate in candidates:
            eval_args = (
                candidate,
                args.max_algebra_dim,
                args.seed_guardrail,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                args.timeout_seconds,
            )
            closure_work_items.append(
                (
                    _cache_key(
                        candidate,
                        max_algebra_dim=args.max_algebra_dim,
                        include_seed_guardrail=args.seed_guardrail,
                        include_center=False,
                        include_centralizer=False,
                        include_idempotents=False,
                        include_j_solve=False,
                        include_projected_centralizer=False,
                        coarse_only_diagnostics=False,
                        generated_j_only=False,
                    ),
                    eval_args,
                )
            )
        closure_results, _closure_cached_count = _evaluate_many(
            closure_work_items,
            jobs=args.jobs,
            chunk_size=args.chunk_size,
            timeout_seconds=args.timeout_seconds,
            stream=args.stream and not args.json,
            stop_on_complex_factor=False,
            cache=cache,
            cache_file=args.cache_file,
            use_cache=args.resume,
            progress_every=args.progress_every,
        )

        candidate_by_name = {_candidate_name(candidate): candidate for candidate in candidates}
        detailed_targets = []
        for result in closure_results:
            if len(detailed_targets) >= args.center_top:
                break
            if not result.generated_algebra_closed:
                continue
            detailed_targets.append(candidate_by_name[result.name])

    detailed_work_items = []
    for candidate in detailed_targets:
        eval_args = (
            candidate,
            args.max_algebra_dim,
            args.seed_guardrail,
            True,
            args.centralizer,
            args.idempotents,
            args.j_solve,
            args.projected_centralizer,
            args.coarse_only_diagnostics,
            args.generated_j_only,
            args.timeout_seconds,
        )
        detailed_work_items.append(
            (
                _cache_key(
                    candidate,
                    max_algebra_dim=args.max_algebra_dim,
                    include_seed_guardrail=args.seed_guardrail,
                    include_center=True,
                    include_centralizer=args.centralizer,
                    include_idempotents=args.idempotents,
                    include_j_solve=args.j_solve,
                    include_projected_centralizer=args.projected_centralizer,
                    coarse_only_diagnostics=args.coarse_only_diagnostics,
                    generated_j_only=args.generated_j_only,
                ),
                eval_args,
            )
        )
    detailed_results, detailed_cached_count = _evaluate_many(
        detailed_work_items,
        jobs=args.jobs,
        chunk_size=args.chunk_size,
        timeout_seconds=args.timeout_seconds,
        stream=args.stream and not args.json,
        stop_on_complex_factor=args.stop_on_complex_factor,
        cache=cache,
        cache_file=args.cache_file,
        use_cache=args.resume,
        progress_every=args.progress_every,
    )
    detailed_by_name = {result.name: result for result in detailed_results}

    if closure_results:
        results = tuple(detailed_by_name.get(result.name, result) for result in closure_results)
    else:
        results = detailed_results
    elapsed_seconds = time.perf_counter() - run_start
    summary = _summary_payload(
        results,
        total_candidates=total_candidates,
        selected_count=len(candidates),
        cached_count=detailed_cached_count,
        elapsed_seconds=elapsed_seconds,
    )
    payload = {
        "scanner_version": SCANNER_VERSION,
        "total_candidate_space": total_candidates,
        "start_index": args.start_index,
        "target_cache_file": (
            None if args.target_cache_file is None else str(args.target_cache_file)
        ),
        "target_matching_count": target_matching_count,
        "target_start_index": args.target_start_index,
        "target_limit": args.target_limit,
        "target_interleave_stride": args.target_interleave_stride,
        "selected_count": len(candidates),
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
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("This runs a stepwise projector-free Bloch Path-A search.")
        print(f"family: {args.family}")
        print(f"scanner_version: {SCANNER_VERSION}")
        print(f"total_candidate_space: {total_candidates}")
        print(f"start_index: {args.start_index}")
        if args.target_cache_file is not None:
            print(f"target_cache_file: {args.target_cache_file}")
            print(f"target_matching_count: {target_matching_count}")
            print(f"target_start_index: {args.target_start_index}")
        print(f"selected_count: {len(candidates)}")
        print(f"candidate_count: {payload['candidate_count']}")
        print(f"cached_count: {summary['cached_count']}")
        print(f"closed_count: {payload['closed_count']}")
        print(f"seed_guardrail_checked: {str(args.seed_guardrail).lower()}")
        print(f"seed_guardrail_rejections: {payload['seed_guardrail_rejections']}")
        print(f"laurent_orthogonal_count: {payload['laurent_orthogonal_count']}")
        print(f"coarse_6_4_count: {payload['coarse_6_4_count']}")
        print(f"class_counts: {summary['class_counts']}")
        print(f"stage_average_seconds: {summary['stage_average_seconds']}")
        if not args.stream:
            for result in results:
                print(_format_result_line(result))

    if args.check:
        if not results or any(
            result.generated_algebra_dimension <= 0
            and result.route_label != "candidate_timeout"
            for result in results
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
