"""Sampled Bloch-rule diagnostics for Path-A spatial QCA searches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import gcd, lcm

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import (
    clock_complex_structure,
    split_projectors_3_2,
)
from clifford_3plus2_d5.qca.floquet_alpha import floquet_alpha_operator
from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    _compatible_pair_orientation_sign_choices,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_twist_operator,
    pair_orientation_j_operator,
)
from clifford_3plus2_d5.qca.gates import is_real_matrix
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleBlochTerm,
    RuleLayerInput,
    RuleToVerdictResult,
    _complementary_rank_6_4_pairs,
    _lower_rank_idempotents_inside_pairs,
    center_basis_of_algebra,
    generated_algebra_basis,
    rule_to_verdict,
    solve_central_idempotents,
)
from clifford_3plus2_d5.qca.spatial_1d import (
    SpatialHoppingTerm,
    SpatialLocalQCALayer1D,
    local_qca_laurent_orthogonal,
    local_qca_symbol_unitary_on_samples,
    root_of_unity,
    spatial_1d_combined_local_qca_layer,
    spatial_alpha_local_qca_layer,
    spatial_alpha_prototype,
)


@dataclass(frozen=True)
class BlochRuleLayerInput:
    name: str
    period: int
    dimension: int
    terms: tuple[SpatialHoppingTerm, ...]

    @property
    def locality_radius(self) -> int:
        return max((abs(term.shift) for term in self.terms), default=0)


@dataclass(frozen=True)
class BlochSampleVerdict:
    sample: int
    generated_algebra_dimension: int
    center_dimension: int
    center_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    generated_transported_j_count: int


@dataclass(frozen=True)
class BlochSeedGuardrail:
    raw_seed_witnesses: tuple[str, ...]
    algebraic_seed_witnesses: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.raw_seed_witnesses and not self.algebraic_seed_witnesses


@dataclass(frozen=True)
class BlochRuleVerdict:
    rule_name: str
    period: int
    layer_count: int
    qca_locality_radius: int
    all_layers_finite_radius: bool
    all_layers_laurent_orthogonal: bool
    all_symbols_unitary_on_samples: bool
    coefficient_matrices_real: bool
    coefficient_algebra_dimension: int
    seed_guardrail_passed: bool
    raw_seed_witnesses: tuple[str, ...]
    algebraic_seed_witnesses: tuple[str, ...]
    sample_count: int
    algebra_dimensions_by_sample: tuple[int, ...]
    center_dimensions_by_sample: tuple[int, ...]
    central_idempotent_rank_profiles_by_sample: tuple[tuple[int, ...], ...]
    coarse_6_4_band_split_at_all_samples: bool
    lower_rank_center_at_any_sample: bool
    compatible_j_section_count: int
    transported_j_section_count: int
    rule_generated_transported_j_section_count: int
    alpha_winding: int | None
    eta_winding: int | None
    winding_gcd: int | None
    winding_lcm: int | None
    winding_proxy_4_3: bool
    global_transport_pm_candidate: bool
    strict_bridge_candidate: bool
    route_label: str
    samples: tuple[BlochSampleVerdict, ...]
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class BlochPathACandidate:
    name: str
    layers: tuple[BlochRuleLayerInput, ...]
    compatible_j_signs: tuple[tuple[int, ...], ...]
    transported_j_signs: tuple[tuple[int, ...], ...]
    alpha_winding: int | None = None
    eta_winding: int | None = None


@dataclass(frozen=True)
class BlochPathASearchSummary:
    candidate_count: int
    seed_guardrail_rejections: int
    unseeded_candidate_count: int
    stable_6_4_band_candidates: int
    topological_pm_candidates: int
    rule_generated_j_section_candidates: int
    strict_bridge_candidates: int
    route_label: str
    candidates: tuple[BlochRuleVerdict, ...]
    load_bearing_qca_bridge: bool = False


def bloch_layer_from_spatial_layer(layer: SpatialLocalQCALayer1D) -> BlochRuleLayerInput:
    return BlochRuleLayerInput(
        name=layer.name,
        period=layer.period,
        dimension=layer.dimension,
        terms=layer.terms,
    )


def bloch_symbol_at_root(layer: BlochRuleLayerInput, sample: int) -> sp.Matrix:
    zeta = root_of_unity(layer.period, sample)
    symbol = sp.zeros(layer.dimension)
    for term in layer.terms:
        symbol += term.matrix * zeta**term.shift
    return symbol.applyfunc(sp.simplify)


def bloch_product_symbol_at_root(
    layers: tuple[BlochRuleLayerInput, ...],
    sample: int,
) -> sp.Matrix:
    if not layers:
        return sp.zeros(0)
    result = identity(layers[0].dimension)
    for layer in layers:
        result = bloch_symbol_at_root(layer, sample) * result
    return result.applyfunc(sp.simplify)


def bloch_layer_laurent_orthogonal(layer: BlochRuleLayerInput) -> bool:
    return local_qca_laurent_orthogonal(
        SpatialLocalQCALayer1D(
            name=layer.name,
            period=layer.period,
            dimension=layer.dimension,
            terms=layer.terms,
        )
    )


def bloch_layer_symbol_unitary_on_samples(layer: BlochRuleLayerInput) -> bool:
    return local_qca_symbol_unitary_on_samples(
        SpatialLocalQCALayer1D(
            name=layer.name,
            period=layer.period,
            dimension=layer.dimension,
            terms=layer.terms,
        )
    )


def _matrix_in_span(matrix: sp.Matrix, basis: tuple[sp.Matrix, ...]) -> bool:
    return matrix_span_rank((*basis, matrix)) == matrix_span_rank(basis)


def _coefficient_matrices(layers: tuple[BlochRuleLayerInput, ...]) -> tuple[sp.Matrix, ...]:
    return tuple(term.matrix for layer in layers for term in layer.terms)


def _is_obvious_seed_projector(matrix: sp.Matrix) -> bool:
    if matrix == sp.zeros(matrix.rows) or matrix == identity(matrix.rows):
        return False
    if matrix * matrix != matrix:
        return False
    if matrix != matrix.T:
        return False
    if any(value not in (0, 1) for value in matrix):
        return False
    rank = matrix.rank()
    return 0 < rank < matrix.rows


def bloch_seed_guardrail(layers: tuple[BlochRuleLayerInput, ...]) -> tuple[BlochSeedGuardrail, int]:
    coefficients = _coefficient_matrices(layers)
    dimension = layers[0].dimension if layers else 10
    p_alpha, p_eta = split_projectors_3_2()
    raw_witnesses: list[str] = []
    for layer in layers:
        for term in layer.terms:
            if term.matrix == p_alpha:
                raw_witnesses.append(f"{layer.name}:shift_{term.shift}:P_alpha")
            elif term.matrix == p_eta:
                raw_witnesses.append(f"{layer.name}:shift_{term.shift}:P_eta")
            elif _is_obvious_seed_projector(term.matrix):
                raw_witnesses.append(
                    f"{layer.name}:shift_{term.shift}:projector_rank_{term.matrix.rank()}"
                )

    coefficient_algebra = generated_algebra_basis(coefficients, dimension=dimension)
    algebraic_witnesses = []
    if _matrix_in_span(p_alpha, coefficient_algebra):
        algebraic_witnesses.append("coefficient_algebra:P_alpha")
    if _matrix_in_span(p_eta, coefficient_algebra):
        algebraic_witnesses.append("coefficient_algebra:P_eta")
    return (
        BlochSeedGuardrail(
            raw_seed_witnesses=tuple(raw_witnesses),
            algebraic_seed_witnesses=tuple(algebraic_witnesses),
        ),
        len(coefficient_algebra),
    )


def _transported_signs(
    *,
    compatible_signs: tuple[tuple[int, ...], ...],
    base_signs: tuple[int, ...],
    alpha_modes: tuple[int, ...],
    eta_modes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    transported = []
    for signs in compatible_signs:
        alpha_flips = {signs[mode] * base_signs[mode] for mode in alpha_modes}
        eta_flips = {signs[mode] * base_signs[mode] for mode in eta_modes}
        if len(alpha_flips) == 1 and alpha_flips == eta_flips:
            transported.append(signs)
    return tuple(transported)


def _sample_verdict(
    layers: tuple[BlochRuleLayerInput, ...],
    *,
    sample: int,
    transported_j_signs: tuple[tuple[int, ...], ...],
    max_center_dimension: int,
) -> BlochSampleVerdict:
    dimension = layers[0].dimension
    symbols = tuple(bloch_symbol_at_root(layer, sample) for layer in layers)
    algebra = generated_algebra_basis(symbols, dimension=dimension)
    center = center_basis_of_algebra(algebra, dimension=dimension)
    center_solved, idempotents = solve_central_idempotents(
        center,
        max_center_dimension=max_center_dimension,
        dimension=dimension,
    )
    pairs = _complementary_rank_6_4_pairs(idempotents, dimension=dimension)
    lower_rank = _lower_rank_idempotents_inside_pairs(idempotents, pairs)
    generated_j_count = sum(
        _matrix_in_span(pair_orientation_j_operator(signs), algebra)
        for signs in transported_j_signs
    )
    return BlochSampleVerdict(
        sample=sample,
        generated_algebra_dimension=len(algebra),
        center_dimension=len(center),
        center_solved=center_solved,
        central_idempotent_ranks=(
            tuple(item.rank for item in idempotents) if center_solved else ()
        ),
        complementary_rank_6_4_pairs=len(pairs),
        lower_rank_central_idempotents=len(lower_rank),
        generated_transported_j_count=generated_j_count,
    )


def bloch_rule_to_verdict(
    candidate: BlochPathACandidate,
    *,
    max_center_dimension: int = 8,
) -> BlochRuleVerdict:
    if not candidate.layers:
        raise ValueError("Bloch rule requires at least one layer")
    period = candidate.layers[0].period
    dimension = candidate.layers[0].dimension
    if any(layer.period != period or layer.dimension != dimension for layer in candidate.layers):
        raise ValueError("all Bloch layers must share period and dimension")

    guardrail, coefficient_algebra_dimension = bloch_seed_guardrail(candidate.layers)
    scalar_shift_only = all(len(layer.terms) == 1 for layer in candidate.layers)
    sample_range = (1,) if scalar_shift_only else tuple(range(period))
    samples = (
        ()
        if not guardrail.passed
        else tuple(
            _sample_verdict(
                candidate.layers,
                sample=sample,
                transported_j_signs=candidate.transported_j_signs,
                max_center_dimension=max_center_dimension,
            )
            for sample in sample_range
        )
    )
    all_layers_laurent = all(bloch_layer_laurent_orthogonal(layer) for layer in candidate.layers)
    all_symbols_unitary = all(
        bloch_layer_symbol_unitary_on_samples(layer) for layer in candidate.layers
    )
    all_finite_radius = all(layer.locality_radius < period for layer in candidate.layers)
    coefficient_real = all(is_real_matrix(matrix) for matrix in _coefficient_matrices(candidate.layers))
    stable_coarse_split = bool(samples) and all(
        sample.center_solved
        and sample.central_idempotent_ranks == (0, 4, 6, 10)
        and sample.complementary_rank_6_4_pairs > 0
        for sample in samples
    )
    lower_rank_any = any(sample.lower_rank_central_idempotents for sample in samples)
    generated_j_section_count = min(
        (sample.generated_transported_j_count for sample in samples),
        default=0,
    )
    winding_gcd = (
        gcd(abs(candidate.alpha_winding), abs(candidate.eta_winding))
        if candidate.alpha_winding is not None and candidate.eta_winding is not None
        else None
    )
    winding_lcm = (
        lcm(abs(candidate.alpha_winding), abs(candidate.eta_winding))
        if candidate.alpha_winding is not None and candidate.eta_winding is not None
        else None
    )
    winding_proxy = candidate.alpha_winding == 4 and candidate.eta_winding == 3 and period == 12
    global_pm = len(candidate.transported_j_signs) == 2 and winding_proxy
    strict_bridge = bool(
        guardrail.passed
        and all_finite_radius
        and all_layers_laurent
        and all_symbols_unitary
        and coefficient_real
        and stable_coarse_split
        and not lower_rank_any
        and global_pm
        and generated_j_section_count == 2
    )

    if not guardrail.passed:
        route_label = "bloch_path_a_seeded_guardrail_rejected"
    elif not all_layers_laurent or not all_symbols_unitary:
        route_label = "bloch_path_a_not_unitary"
    elif not stable_coarse_split:
        route_label = "bloch_path_a_no_stable_6_4_band_split"
    elif lower_rank_any:
        route_label = "bloch_path_a_lower_rank_falsified"
    elif not global_pm:
        route_label = "bloch_path_a_no_global_pm_transport"
    elif generated_j_section_count != 2:
        route_label = "bloch_path_a_j_not_rule_generated"
    else:
        route_label = "bloch_path_a_bridge_candidate"

    return BlochRuleVerdict(
        rule_name=candidate.name,
        period=period,
        layer_count=len(candidate.layers),
        qca_locality_radius=max(layer.locality_radius for layer in candidate.layers),
        all_layers_finite_radius=all_finite_radius,
        all_layers_laurent_orthogonal=all_layers_laurent,
        all_symbols_unitary_on_samples=all_symbols_unitary,
        coefficient_matrices_real=coefficient_real,
        coefficient_algebra_dimension=coefficient_algebra_dimension,
        seed_guardrail_passed=guardrail.passed,
        raw_seed_witnesses=guardrail.raw_seed_witnesses,
        algebraic_seed_witnesses=guardrail.algebraic_seed_witnesses,
        sample_count=len(samples),
        algebra_dimensions_by_sample=tuple(
            sample.generated_algebra_dimension for sample in samples
        ),
        center_dimensions_by_sample=tuple(sample.center_dimension for sample in samples),
        central_idempotent_rank_profiles_by_sample=tuple(
            sample.central_idempotent_ranks for sample in samples
        ),
        coarse_6_4_band_split_at_all_samples=stable_coarse_split,
        lower_rank_center_at_any_sample=lower_rank_any,
        compatible_j_section_count=len(candidate.compatible_j_signs),
        transported_j_section_count=len(candidate.transported_j_signs),
        rule_generated_transported_j_section_count=generated_j_section_count,
        alpha_winding=candidate.alpha_winding,
        eta_winding=candidate.eta_winding,
        winding_gcd=winding_gcd,
        winding_lcm=winding_lcm,
        winding_proxy_4_3=winding_proxy,
        global_transport_pm_candidate=global_pm,
        strict_bridge_candidate=strict_bridge,
        route_label=route_label,
        samples=samples,
        load_bearing_qca_bridge=strict_bridge,
    )


def _full_shift_layer(
    *,
    name: str,
    matrix: sp.Matrix,
    shift: int,
    period: int,
    dimension: int = 10,
) -> BlochRuleLayerInput:
    return BlochRuleLayerInput(
        name=name,
        period=period,
        dimension=dimension,
        terms=(SpatialHoppingTerm(shift=shift, matrix=matrix),),
    )


def _mode_edge_matrix(
    edges: tuple[tuple[int, int], ...],
    *,
    dimension: int = 10,
) -> sp.Matrix:
    mode_count = dimension // 2
    matrix = sp.zeros(dimension)
    for source, target in edges:
        matrix[target, source] = 1
        matrix[target + mode_count, source + mode_count] = 1
    return matrix


def bloch_path_a_projector_free_combined_layer() -> RuleLayerInput:
    """Return a projector-free Route-1/Route-2 Bloch candidate.

    The hopping is a monomial mode-pair cycle with edge shifts
    ``(4,4,4,3,3)``.  Its raw coefficients are partial permutations, not
    ``P_alpha/P_eta`` projectors.  The on-site update is the Route-1
    noncommuting ``U2 * U1`` layer.
    """

    rule = spatial_alpha_prototype()
    onsite = floquet_alpha_noncommuting_candidates(pattern_index=0)[0]
    u1 = floquet_alpha_operator(onsite.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(onsite)
    ulocal = sp.simplify(u2 * u1)
    cycle = (1, 2, 3, 4, 0)
    source_shifts = (4, 4, 4, 3, 3)
    terms = []
    for shift in sorted(set(source_shifts)):
        edges = tuple(
            (source, cycle[source])
            for source, source_shift in enumerate(source_shifts)
            if source_shift == shift
        )
        terms.append(
            RuleBlochTerm(
                shift=shift,
                matrix=sp.simplify(ulocal * _mode_edge_matrix(edges, dimension=rule.dimension)),
            )
        )
    representative = sum((term.matrix for term in terms), sp.zeros(rule.dimension))
    return RuleLayerInput(
        name="path_a_projector_free_cycle_combined",
        matrix=representative,
        locality_radius=max(source_shifts),
        bloch_terms=tuple(terms),
    )


def bloch_path_a_projector_free_rule_to_verdict(
    *,
    max_generated_algebra_dimension: int | None = 16,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (bloch_path_a_projector_free_combined_layer(),),
        rule_name="path_a_projector_free_cycle_combined",
        bloch_period=spatial_alpha_prototype().period,
        max_generated_algebra_dimension=max_generated_algebra_dimension,
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def bloch_path_a_candidates() -> tuple[BlochPathACandidate, ...]:
    rule = spatial_alpha_prototype()
    onsite = floquet_alpha_noncommuting_candidates(pattern_index=0)[0]
    u1 = floquet_alpha_operator(onsite.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(onsite)
    ulocal = sp.simplify(u2 * u1)
    compatible = _compatible_pair_orientation_sign_choices(onsite)
    transported = _transported_signs(
        compatible_signs=compatible,
        base_signs=onsite.orientation_signs,
        alpha_modes=onsite.pattern.alpha_modes,
        eta_modes=onsite.pattern.eta_modes,
    )
    seeded_spatial = bloch_layer_from_spatial_layer(spatial_alpha_local_qca_layer(rule))
    combined = bloch_layer_from_spatial_layer(spatial_1d_combined_local_qca_layer(rule))
    return (
        BlochPathACandidate(
            name="path_a_seeded_projector_shift",
            layers=(seeded_spatial,),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
            alpha_winding=4,
            eta_winding=3,
        ),
        BlochPathACandidate(
            name="path_a_combined_route1_route2",
            layers=(combined,),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
            alpha_winding=4,
            eta_winding=3,
        ),
        BlochPathACandidate(
            name="path_a_unseeded_uniform_identity_shift",
            layers=(
                _full_shift_layer(
                    name="uniform_identity_shift",
                    matrix=identity(rule.dimension),
                    shift=1,
                    period=rule.period,
                    dimension=rule.dimension,
                ),
            ),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
        ),
        BlochPathACandidate(
            name="path_a_shifted_u2_u1_layers",
            layers=(
                _full_shift_layer(
                    name="shifted_u1",
                    matrix=u1,
                    shift=1,
                    period=rule.period,
                    dimension=rule.dimension,
                ),
                _full_shift_layer(
                    name="shifted_u2",
                    matrix=u2,
                    shift=3,
                    period=rule.period,
                    dimension=rule.dimension,
                ),
            ),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
            alpha_winding=4,
            eta_winding=3,
        ),
        BlochPathACandidate(
            name="path_a_unseeded_clock_shift",
            layers=(
                _full_shift_layer(
                    name="uniform_clock_shift",
                    matrix=clock_complex_structure(),
                    shift=1,
                    period=rule.period,
                    dimension=rule.dimension,
                ),
            ),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
        ),
        BlochPathACandidate(
            name="path_a_shifted_ulocal",
            layers=(
                _full_shift_layer(
                    name="shifted_ulocal",
                    matrix=ulocal,
                    shift=4,
                    period=rule.period,
                    dimension=rule.dimension,
                ),
            ),
            compatible_j_signs=compatible,
            transported_j_signs=transported,
            alpha_winding=4,
            eta_winding=3,
        ),
    )


def _bloch_rule_to_verdict_for_pool(candidate: BlochPathACandidate) -> BlochRuleVerdict:
    return bloch_rule_to_verdict(candidate)


def bloch_path_a_search_summary(*, jobs: int = 1) -> BlochPathASearchSummary:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    candidates = bloch_path_a_candidates()
    if jobs == 1:
        verdicts = tuple(bloch_rule_to_verdict(candidate) for candidate in candidates)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            verdicts = tuple(executor.map(_bloch_rule_to_verdict_for_pool, candidates))
    unseeded = tuple(verdict for verdict in verdicts if verdict.seed_guardrail_passed)
    strict_count = sum(verdict.strict_bridge_candidate for verdict in verdicts)
    if strict_count:
        route_label = "bloch_path_a_bridge_candidate_found"
    elif any(verdict.global_transport_pm_candidate for verdict in verdicts):
        route_label = "bloch_path_a_seeded_shape_only"
    else:
        route_label = "bloch_path_a_no_unseeded_bridge"
    return BlochPathASearchSummary(
        candidate_count=len(verdicts),
        seed_guardrail_rejections=sum(not verdict.seed_guardrail_passed for verdict in verdicts),
        unseeded_candidate_count=len(unseeded),
        stable_6_4_band_candidates=sum(
            verdict.coarse_6_4_band_split_at_all_samples for verdict in verdicts
        ),
        topological_pm_candidates=sum(verdict.global_transport_pm_candidate for verdict in verdicts),
        rule_generated_j_section_candidates=sum(
            verdict.rule_generated_transported_j_section_count == 2 for verdict in verdicts
        ),
        strict_bridge_candidates=strict_count,
        route_label=route_label,
        candidates=verdicts,
        load_bearing_qca_bridge=bool(strict_count),
    )
