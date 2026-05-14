"""Floquet-alpha physical primitive family."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, is_zero_matrix
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.rule_verdict import RuleLayerInput, RuleToVerdictResult, rule_to_verdict


ALPHA_PHASE = sp.Rational(2, 3) * sp.pi
ETA_PHASE = sp.Rational(1, 2) * sp.pi
FLOQUET_ALPHA_EXACT_WORKING_FIELD = "QQ(zeta_12); real entries in QQ(sqrt(3))"
FLOQUET_ALPHA_SCALED_RELATION = "K_alpha=(2U+I)P_alpha, K_alpha^2=-3P_alpha"
FLOQUET_ALPHA_ALPHA_SECTOR_CENTRALIZER_DIMENSION = 18
FLOQUET_ALPHA_ETA_SECTOR_CENTRALIZER_DIMENSION = 8
FLOQUET_ALPHA_COMPATIBLE_CENTRALIZER_DIMENSION = 26
FLOQUET_ALPHA_COMPATIBLE_J_MODULI_DIMENSION = 9


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
    compatible_j_moduli_dimension: int
    strict_compatible_j_forced: bool
    compatible_j_solved: bool
    alpha_plus_polarization_passed: bool
    pass_strict_rule_to_bridge: bool
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


def floquet_alpha_polarization_certificate(
    candidate: FloquetAlphaCandidate,
) -> FloquetAlphaPolarizationCertificate:
    verdict = floquet_alpha_rule_to_verdict(candidate)
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
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
        alpha_sector_centralizer_dimension=FLOQUET_ALPHA_ALPHA_SECTOR_CENTRALIZER_DIMENSION,
        eta_sector_centralizer_dimension=FLOQUET_ALPHA_ETA_SECTOR_CENTRALIZER_DIMENSION,
        compatible_centralizer_dimension=verdict.compatible_centralizer_dimension,
        compatible_j_moduli_dimension=(
            verdict.compatible_j_moduli_dimension
            if verdict.compatible_j_moduli_dimension is not None
            else FLOQUET_ALPHA_COMPATIBLE_J_MODULI_DIMENSION
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
