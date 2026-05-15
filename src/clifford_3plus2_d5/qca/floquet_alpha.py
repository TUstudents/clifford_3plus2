"""Floquet-alpha physical primitive family."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import commutator, is_zero_matrix
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    RuleToVerdictResult,
    generated_algebra_basis,
    rule_to_verdict,
)


ALPHA_PHASE = sp.Rational(2, 3) * sp.pi
ETA_PHASE = sp.Rational(1, 2) * sp.pi
FLOQUET_ALPHA_EXACT_WORKING_FIELD = "QQ(zeta_12); real entries in QQ(sqrt(3))"
FLOQUET_ALPHA_SCALED_RELATION = "K_alpha=(2U+I)P_alpha, K_alpha^2=-3P_alpha"


@dataclass(frozen=True)
class FloquetAlphaCandidate:
    pattern_index: int
    alpha_modes: tuple[int, ...]
    eta_modes: tuple[int, ...]

    @property
    def name(self) -> str:
        alpha = "_".join(str(mode + 1) for mode in self.alpha_modes)
        return f"floquet_alpha_pattern_{self.pattern_index}_alpha_{alpha}"


@dataclass(frozen=True)
class FloquetAlphaPolarizationCertificate:
    candidate_name: str
    exact_working_field: str
    alpha_projector_rank: int
    eta_projector_rank: int
    spectral_projectors_are_idempotent: bool
    spectral_projectors_are_complementary: bool
    scaled_alpha_relation: str
    scaled_alpha_square_relation: bool
    scaled_alpha_orthogonality_relation: bool
    scaled_alpha_commutes_with_projectors: bool
    eta_j_square_relation: bool
    eta_j_orthogonality_relation: bool
    scaled_polarization_certified: bool
    normalized_j_requires_sqrt3: bool
    canonical_j_generated_by_floquet: bool
    canonical_j_squared_minus_identity: bool
    canonical_j_orthogonal: bool
    canonical_j_commutes_with_projectors: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    generated_j_moduli_dimension: int | None
    alpha_sector_centralizer_dimension: int
    eta_sector_centralizer_dimension: int
    compatible_centralizer_dimension: int
    compatible_j_moduli_dimension: int | None
    locality_radius_bound: int
    local_compatible_operator_dimension: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    strict_compatible_j_forced: bool
    compatible_j_solved: bool
    alpha_plus_polarization_passed: bool
    pass_strict_rule_to_bridge: bool
    verdict: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaSecondLayerCertificate:
    candidate_name: str
    second_layer_name: str
    u_v_commute: bool
    second_layer_real_orthogonal: bool
    alpha_cycle_order_certified: bool
    eta_swap_order_certified: bool
    generated_algebra_dimension: int
    center_dimension: int
    compatible_centralizer_dimension: int
    rule_center_solved: bool
    rule_verdict: str
    explicit_lower_rank_projector_ranks: tuple[int, ...]
    no_locking_guardrail_passed: bool
    compatible_centralizer_collapsed: bool
    pass_strict_rule_to_bridge: bool
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaTimeReversalCertificate:
    candidate_name: str
    time_reversal_name: str
    time_reversal_origin: str
    k_real_orthogonal: bool
    k_involution: bool
    k_conjugates_floquet_to_inverse: bool
    k_anticommutes_with_canonical_j: bool
    k_in_generated_algebra: bool
    compatible_j_moduli_dimension_before_k: int | None
    k_fixed_compatible_j_moduli_dimension: int | None
    k_fixed_generated_complex_structure_count: int
    k_fixed_local_compatible_complex_structure_count: int
    k_fixed_local_matches_canonical_orbit_count: int
    k_reduces_full_moduli: bool
    k_reduces_to_global_pm: bool
    strict_bridge_candidates: int
    verdict: str
    load_bearing_qca_bridge: bool = False


def _simplify_matrix(matrix: sp.Matrix) -> sp.Matrix:
    return matrix.applyfunc(sp.simplify)


def pair_rotation(mode: int, phase: sp.Expr, *, dimension: int = 10) -> sp.Matrix:
    if not (0 <= mode < dimension // 2):
        raise ValueError("mode index out of range")

    matrix = identity(dimension)
    x_index = mode
    y_index = mode + dimension // 2
    cos_phase = sp.simplify(sp.cos(phase))
    sin_phase = sp.simplify(sp.sin(phase))
    matrix[x_index, x_index] = cos_phase
    matrix[x_index, y_index] = -sin_phase
    matrix[y_index, x_index] = sin_phase
    matrix[y_index, y_index] = cos_phase
    return matrix


def floquet_alpha_operator(candidate: FloquetAlphaCandidate) -> sp.Matrix:
    operator = identity(10)
    for mode in candidate.alpha_modes:
        operator = pair_rotation(mode, ALPHA_PHASE) * operator
    for mode in candidate.eta_modes:
        operator = pair_rotation(mode, ETA_PHASE) * operator
    return sp.simplify(operator)


def mode_pair_permutation_operator(
    mode_mapping: dict[int, int],
    *,
    dimension: int = 10,
) -> sp.Matrix:
    if set(mode_mapping) != set(mode_mapping.values()):
        raise ValueError("mode mapping must be a permutation of its support")
    if not all(0 <= mode < dimension // 2 for mode in mode_mapping):
        raise ValueError("mode index out of range")

    matrix = identity(dimension)
    half_dimension = dimension // 2
    for source in mode_mapping:
        matrix[source, source] = 0
        matrix[source + half_dimension, source + half_dimension] = 0
    for source, target in mode_mapping.items():
        matrix[target, source] = 1
        matrix[target + half_dimension, source + half_dimension] = 1
    return matrix


def floquet_alpha_cycle_swap_operator(candidate: FloquetAlphaCandidate) -> sp.Matrix:
    """Cycle the three alpha mode-pairs and swap the two eta mode-pairs."""

    mode_mapping: dict[int, int] = {}
    for source, target in zip(
        candidate.alpha_modes,
        (*candidate.alpha_modes[1:], candidate.alpha_modes[0]),
        strict=True,
    ):
        mode_mapping[source] = target
    for source, target in zip(candidate.eta_modes, reversed(candidate.eta_modes), strict=True):
        mode_mapping[source] = target
    return mode_pair_permutation_operator(mode_mapping)


def floquet_alpha_cycle_swap_layer(candidate: FloquetAlphaCandidate) -> RuleLayerInput:
    return RuleLayerInput(
        name=f"{candidate.name}_cycle_swap_lock",
        matrix=floquet_alpha_cycle_swap_operator(candidate),
        support=(0,),
        locality_radius=0,
    )


def floquet_alpha_spectral_projectors(
    candidate: FloquetAlphaCandidate,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Return projectors onto the order-3 and order-4 resonance sectors.

    The real minimal polynomials are:

    f_alpha(x) = x^2 + x + 1
    f_eta(x) = x^2 + 1

    Bezout gives ``-x f_alpha + (x + 1) f_eta = 1``, so
    ``P_alpha = (U + I)(U^2 + I)``.
    """

    operator = floquet_alpha_operator(candidate)
    one = identity(10)
    alpha_projector = _simplify_matrix((operator + one) * (operator * operator + one))
    eta_projector = _simplify_matrix(one - alpha_projector)
    return alpha_projector, eta_projector


def _complex_sector_centralizer_dimension(projector: sp.Matrix) -> int:
    rank = projector.rank()
    if rank % 2:
        raise ValueError("real spectral projector rank must be even")
    complex_rank = rank // 2
    return 2 * complex_rank * complex_rank


def floquet_alpha_sector_centralizer_dimensions(
    candidate: FloquetAlphaCandidate,
) -> tuple[int, int]:
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
    return (
        _complex_sector_centralizer_dimension(alpha_projector),
        _complex_sector_centralizer_dimension(eta_projector),
    )


def floquet_alpha_compatible_centralizer_dimension(
    candidate: FloquetAlphaCandidate,
) -> int:
    return sum(floquet_alpha_sector_centralizer_dimensions(candidate))


def floquet_alpha_canonical_j(candidate: FloquetAlphaCandidate) -> sp.Matrix:
    """Return the normalized complex structure over ``QQ(sqrt(3))``."""

    operator = floquet_alpha_operator(candidate)
    _, eta_projector = floquet_alpha_spectral_projectors(candidate)
    alpha_j = floquet_alpha_scaled_alpha_operator(candidate) / sp.sqrt(3)
    eta_j = operator * eta_projector
    return _simplify_matrix(alpha_j + eta_j)


def floquet_alpha_scaled_alpha_operator(candidate: FloquetAlphaCandidate) -> sp.Matrix:
    """Return ``K_alpha=(2U+I)P_alpha`` without dividing by ``sqrt(3)``."""

    operator = floquet_alpha_operator(candidate)
    alpha_projector, _ = floquet_alpha_spectral_projectors(candidate)
    return _simplify_matrix((2 * operator + identity(10)) * alpha_projector)


def floquet_alpha_layer(candidate: FloquetAlphaCandidate) -> RuleLayerInput:
    return RuleLayerInput(
        name=candidate.name,
        matrix=floquet_alpha_operator(candidate),
        support=(0,),
        locality_radius=0,
    )


def floquet_alpha_time_reversal_operator(*, dimension: int = 10) -> sp.Matrix:
    """Return the declared real pair-conjugation involution K.

    In the fixed x_1,...,x_5,y_1,...,y_5 basis this is complex conjugation for
    the standard pair complex structure: x coordinates are fixed and y
    coordinates change sign.
    """

    half_dimension = dimension // 2
    return sp.diag(*([1] * half_dimension), *([-1] * half_dimension))


def floquet_alpha_candidates() -> tuple[FloquetAlphaCandidate, ...]:
    candidates = []
    all_modes = tuple(range(5))
    for pattern_index, alpha_modes in enumerate(combinations(all_modes, 3)):
        alpha_tuple = tuple(alpha_modes)
        eta_tuple = tuple(mode for mode in all_modes if mode not in alpha_tuple)
        candidates.append(
            FloquetAlphaCandidate(
                pattern_index=pattern_index,
                alpha_modes=alpha_tuple,
                eta_modes=eta_tuple,
            )
        )
    return tuple(candidates)


def floquet_alpha_rule_to_verdict(
    candidate: FloquetAlphaCandidate,
    *,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (floquet_alpha_layer(candidate),),
        rule_name=candidate.name,
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def floquet_alpha_cycle_swap_rule_to_verdict(
    candidate: FloquetAlphaCandidate,
    *,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (floquet_alpha_layer(candidate), floquet_alpha_cycle_swap_layer(candidate)),
        rule_name=f"{candidate.name}_with_cycle_swap_lock",
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def _rank_if_idempotent(matrix: sp.Matrix) -> int | None:
    simplified = _simplify_matrix(matrix)
    if _simplify_matrix(simplified * simplified - simplified) != sp.zeros(10):
        return None
    return simplified.rank()


def floquet_alpha_second_layer_certificate(
    candidate: FloquetAlphaCandidate,
) -> FloquetAlphaSecondLayerCertificate:
    u_layer = floquet_alpha_layer(candidate)
    v_layer = floquet_alpha_cycle_swap_layer(candidate)
    u_operator = u_layer.matrix
    v_operator = v_layer.matrix
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
    rule_result = floquet_alpha_cycle_swap_rule_to_verdict(candidate)
    alpha_compatible_centralizer_dimension = floquet_alpha_compatible_centralizer_dimension(
        candidate
    )
    one = identity(10)

    alpha_fixed_projector = _simplify_matrix(
        (one + v_operator + v_operator**2) * alpha_projector / 3
    )
    eta_symmetric_projector = _simplify_matrix((one + v_operator) * eta_projector / 2)
    eta_antisymmetric_projector = _simplify_matrix((one - v_operator) * eta_projector / 2)
    lower_rank_projector_ranks = tuple(
        rank
        for rank in (
            _rank_if_idempotent(alpha_fixed_projector),
            _rank_if_idempotent(eta_symmetric_projector),
            _rank_if_idempotent(eta_antisymmetric_projector),
        )
        if rank is not None and rank < 4
    )

    return FloquetAlphaSecondLayerCertificate(
        candidate_name=candidate.name,
        second_layer_name=v_layer.name,
        u_v_commute=is_zero_matrix(commutator(u_operator, v_operator)),
        second_layer_real_orthogonal=v_operator.T * v_operator == one,
        alpha_cycle_order_certified=(
            _simplify_matrix((v_operator**3 - one) * alpha_projector) == sp.zeros(10)
        ),
        eta_swap_order_certified=(
            _simplify_matrix((v_operator**2 - one) * eta_projector) == sp.zeros(10)
        ),
        generated_algebra_dimension=rule_result.generated_algebra_dimension,
        center_dimension=rule_result.center_dimension,
        compatible_centralizer_dimension=rule_result.compatible_centralizer_dimension,
        rule_center_solved=rule_result.center_solved,
        rule_verdict=rule_result.verdict,
        explicit_lower_rank_projector_ranks=lower_rank_projector_ranks,
        no_locking_guardrail_passed=not lower_rank_projector_ranks,
        compatible_centralizer_collapsed=(
            rule_result.compatible_centralizer_dimension
            < alpha_compatible_centralizer_dimension
        ),
        pass_strict_rule_to_bridge=rule_result.pass_rule_to_bridge,
    )


def _matrix_in_span(matrix: sp.Matrix, basis: tuple[sp.Matrix, ...]) -> bool:
    return matrix_span_rank((*basis, matrix)) == matrix_span_rank(basis)


def _k_fixed_compatible_j_moduli_dimension(candidate: FloquetAlphaCandidate) -> int:
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
    alpha_complex_rank = alpha_projector.rank() // 2
    eta_complex_rank = eta_projector.rank() // 2
    return (alpha_complex_rank**2 // 4) + (eta_complex_rank**2 // 4)


def floquet_alpha_time_reversal_certificate(
    candidate: FloquetAlphaCandidate,
) -> FloquetAlphaTimeReversalCertificate:
    verdict = floquet_alpha_rule_to_verdict(candidate)
    operator = floquet_alpha_operator(candidate)
    canonical_j = floquet_alpha_canonical_j(candidate)
    k_operator = floquet_alpha_time_reversal_operator()
    one = identity(10)
    zero = sp.zeros(10)
    k_fixed_generated = tuple(
        item
        for item in verdict.generated_complex_structures
        if _simplify_matrix(k_operator * item.matrix * k_operator + item.matrix) == zero
    )
    k_fixed_local = tuple(
        item
        for item in verdict.local_compatible_complex_structures
        if _simplify_matrix(k_operator * item.matrix * k_operator + item.matrix) == zero
    )
    canonical_orbit_matches = sum(
        _simplify_matrix(item.matrix - canonical_j) == zero
        or _simplify_matrix(item.matrix + canonical_j) == zero
        for item in k_fixed_local
    )
    generated_basis = generated_algebra_basis((operator,))
    k_in_generated = _matrix_in_span(k_operator, generated_basis)
    k_fixed_moduli_dimension = _k_fixed_compatible_j_moduli_dimension(candidate)
    k_reduces_to_global_pm = (
        k_fixed_moduli_dimension == 0
        and len(k_fixed_local) == 2
        and canonical_orbit_matches == 2
    )

    if k_in_generated and k_reduces_to_global_pm:
        route_verdict = "time_reversal_forces_global_pm"
    elif k_reduces_to_global_pm:
        route_verdict = "declared_time_reversal_reduces_to_global_pm"
    elif k_fixed_moduli_dimension < (verdict.compatible_j_moduli_dimension or 0):
        route_verdict = "declared_time_reversal_reduces_moduli_not_unique"
    else:
        route_verdict = "declared_time_reversal_no_reduction"

    return FloquetAlphaTimeReversalCertificate(
        candidate_name=candidate.name,
        time_reversal_name="declared_pair_conjugation",
        time_reversal_origin="declared_not_rule_generated",
        k_real_orthogonal=_simplify_matrix(k_operator.T * k_operator - one) == zero,
        k_involution=_simplify_matrix(k_operator * k_operator - one) == zero,
        k_conjugates_floquet_to_inverse=(
            _simplify_matrix(k_operator * operator * k_operator - operator.T) == zero
        ),
        k_anticommutes_with_canonical_j=(
            _simplify_matrix(k_operator * canonical_j * k_operator + canonical_j) == zero
        ),
        k_in_generated_algebra=k_in_generated,
        compatible_j_moduli_dimension_before_k=verdict.compatible_j_moduli_dimension,
        k_fixed_compatible_j_moduli_dimension=k_fixed_moduli_dimension,
        k_fixed_generated_complex_structure_count=len(k_fixed_generated),
        k_fixed_local_compatible_complex_structure_count=len(k_fixed_local),
        k_fixed_local_matches_canonical_orbit_count=canonical_orbit_matches,
        k_reduces_full_moduli=(
            verdict.compatible_j_moduli_dimension is not None
            and k_fixed_moduli_dimension < verdict.compatible_j_moduli_dimension
        ),
        k_reduces_to_global_pm=k_reduces_to_global_pm,
        strict_bridge_candidates=0,
        verdict=route_verdict,
    )


def floquet_alpha_polarization_certificate(
    candidate: FloquetAlphaCandidate,
) -> FloquetAlphaPolarizationCertificate:
    verdict = floquet_alpha_rule_to_verdict(candidate)
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
    alpha_sector_dimension, eta_sector_dimension = (
        floquet_alpha_sector_centralizer_dimensions(candidate)
    )
    scaled_alpha = floquet_alpha_scaled_alpha_operator(candidate)
    eta_j = _simplify_matrix(floquet_alpha_operator(candidate) * eta_projector)
    canonical_j = floquet_alpha_canonical_j(candidate)
    one = identity(10)
    zero = sp.zeros(10)
    spectral_projectors_are_idempotent = (
        _simplify_matrix(alpha_projector * alpha_projector - alpha_projector) == zero
        and _simplify_matrix(eta_projector * eta_projector - eta_projector) == zero
    )
    spectral_projectors_are_complementary = (
        _simplify_matrix(alpha_projector + eta_projector - one) == zero
        and _simplify_matrix(alpha_projector * eta_projector) == zero
        and _simplify_matrix(eta_projector * alpha_projector) == zero
    )
    scaled_alpha_square = (
        _simplify_matrix(scaled_alpha * scaled_alpha + 3 * alpha_projector) == zero
    )
    scaled_alpha_orthogonal = (
        _simplify_matrix(scaled_alpha.T * scaled_alpha - 3 * alpha_projector) == zero
    )
    scaled_alpha_commutes = is_zero_matrix(
        commutator(scaled_alpha, alpha_projector)
    ) and is_zero_matrix(commutator(scaled_alpha, eta_projector))
    eta_j_square = _simplify_matrix(eta_j * eta_j + eta_projector) == zero
    eta_j_orthogonal = _simplify_matrix(eta_j.T * eta_j - eta_projector) == zero
    scaled_polarization_certified = (
        scaled_alpha_square
        and scaled_alpha_orthogonal
        and scaled_alpha_commutes
        and eta_j_square
        and eta_j_orthogonal
    )
    canonical_j_squared = _simplify_matrix(canonical_j * canonical_j + one) == zero
    canonical_j_orthogonal = _simplify_matrix(canonical_j.T * canonical_j - one) == zero
    canonical_j_commutes = is_zero_matrix(commutator(canonical_j, alpha_projector)) and is_zero_matrix(
        commutator(canonical_j, eta_projector)
    )
    alpha_plus_polarization_passed = (
        spectral_projectors_are_idempotent
        and spectral_projectors_are_complementary
        and alpha_projector.rank() == 6
        and eta_projector.rank() == 4
        and scaled_polarization_certified
        and verdict.complementary_rank_6_4_pairs > 0
        and not verdict.lower_rank_central_idempotents
    )
    return FloquetAlphaPolarizationCertificate(
        candidate_name=candidate.name,
        exact_working_field=FLOQUET_ALPHA_EXACT_WORKING_FIELD,
        alpha_projector_rank=alpha_projector.rank(),
        eta_projector_rank=eta_projector.rank(),
        spectral_projectors_are_idempotent=spectral_projectors_are_idempotent,
        spectral_projectors_are_complementary=spectral_projectors_are_complementary,
        scaled_alpha_relation=FLOQUET_ALPHA_SCALED_RELATION,
        scaled_alpha_square_relation=scaled_alpha_square,
        scaled_alpha_orthogonality_relation=scaled_alpha_orthogonal,
        scaled_alpha_commutes_with_projectors=scaled_alpha_commutes,
        eta_j_square_relation=eta_j_square,
        eta_j_orthogonality_relation=eta_j_orthogonal,
        scaled_polarization_certified=scaled_polarization_certified,
        normalized_j_requires_sqrt3=True,
        canonical_j_generated_by_floquet=True,
        canonical_j_squared_minus_identity=canonical_j_squared,
        canonical_j_orthogonal=canonical_j_orthogonal,
        canonical_j_commutes_with_projectors=canonical_j_commutes,
        central_idempotent_ranks=tuple(item.rank for item in verdict.central_idempotents),
        complementary_rank_6_4_pairs=verdict.complementary_rank_6_4_pairs,
        lower_rank_central_idempotents=len(verdict.lower_rank_central_idempotents),
        generated_j_moduli_dimension=verdict.generated_j_moduli_dimension,
        alpha_sector_centralizer_dimension=alpha_sector_dimension,
        eta_sector_centralizer_dimension=eta_sector_dimension,
        compatible_centralizer_dimension=verdict.compatible_centralizer_dimension,
        compatible_j_moduli_dimension=verdict.compatible_j_moduli_dimension,
        locality_radius_bound=verdict.locality_radius_bound,
        local_compatible_operator_dimension=verdict.local_compatible_operator_dimension,
        local_compatible_j_solved=verdict.local_compatible_j_solved,
        local_compatible_j_moduli_dimension=verdict.local_compatible_j_moduli_dimension,
        local_compatible_complex_structure_count=(
            len(verdict.local_compatible_complex_structures)
        ),
        strict_compatible_j_forced=verdict.forced_j_found,
        compatible_j_solved=verdict.compatible_j_solved,
        alpha_plus_polarization_passed=alpha_plus_polarization_passed,
        pass_strict_rule_to_bridge=verdict.pass_rule_to_bridge,
        verdict=(
            "polarization_j_produced_not_strictly_unique"
            if alpha_plus_polarization_passed and not verdict.pass_rule_to_bridge
            else verdict.verdict
        ),
    )


def search_floquet_alpha(
    candidates: Iterable[FloquetAlphaCandidate] | None = None,
) -> tuple[RuleToVerdictResult, ...]:
    candidates = candidates if candidates is not None else floquet_alpha_candidates()
    return tuple(floquet_alpha_rule_to_verdict(candidate) for candidate in candidates)
