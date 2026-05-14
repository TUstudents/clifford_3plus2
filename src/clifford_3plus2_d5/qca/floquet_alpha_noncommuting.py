"""Block-preserving noncommuting Floquet-alpha twist family."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import matrix_span_rank
from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.qca.floquet_alpha import (
    FloquetAlphaCandidate,
    floquet_alpha_candidates,
    floquet_alpha_canonical_j,
    floquet_alpha_layer,
    floquet_alpha_operator,
    floquet_alpha_spectral_projectors,
)
from clifford_3plus2_d5.qca.rule_verdict import (
    RuleLayerInput,
    RuleToVerdictResult,
    center_basis_of_algebra,
    generated_algebra_basis,
    rule_to_verdict,
)


RouteLabel = str


@dataclass(frozen=True)
class FloquetAlphaNoncommutingCandidate:
    pattern: FloquetAlphaCandidate
    orientation_signs: tuple[int, ...]

    @property
    def name(self) -> str:
        signs = "_".join("p" if sign > 0 else "m" for sign in self.orientation_signs)
        return f"{self.pattern.name}_noncommuting_signed_twist_{signs}"


@dataclass(frozen=True)
class FloquetAlphaNoncommutingCertificate:
    candidate_name: str
    base_candidate_name: str
    twist_layer_name: str
    orientation_signs: tuple[int, ...]
    mode_mapping: tuple[tuple[int, int], ...]
    u1_u2_commute: bool
    u2_real_orthogonal: bool
    u2_preserves_alpha_projector: bool
    u2_preserves_eta_projector: bool
    alpha_orientation_nonconstant: bool
    eta_orientation_nonconstant: bool
    generated_algebra_dimension: int
    center_dimension: int
    center_solved: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    compatible_centralizer_dimension: int
    compatible_j_solved: bool
    compatible_j_moduli_dimension: int | None
    compatible_complex_structure_count: int
    generated_j_solved: bool
    generated_j_moduli_dimension: int | None
    generated_complex_structure_count: int
    local_compatible_operator_dimension: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    forced_j_found: bool
    pass_strict_rule_to_bridge: bool
    rule_verdict: str
    route_label: RouteLabel
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class FloquetAlphaNoncommutingJDiagnostic:
    index: int
    expression: tuple[tuple[str, sp.Expr], ...]
    pair_orientation_signs: tuple[int, ...]
    in_generated_algebra: bool
    in_rule_local_center: bool
    equals_spectral_polarization_j: bool
    equals_negative_spectral_polarization_j: bool
    commutes_with_u1: bool
    commutes_with_u2: bool
    squares_to_minus_identity: bool
    orthogonal: bool
    matrix: sp.Matrix


@dataclass(frozen=True)
class FloquetAlphaNoncommutingJGapCertificate:
    candidate_name: str
    compatible_j_count: int
    generated_algebra_dimension: int
    center_dimension: int
    compatible_centralizer_dimension: int
    compatible_j_solved: bool
    compatible_j_moduli_dimension: int | None
    generated_j_solved: bool
    generated_complex_structure_count: int
    local_compatible_j_solved: bool
    local_compatible_j_moduli_dimension: int | None
    local_compatible_complex_structure_count: int
    compatible_j_in_generated_algebra_count: int
    compatible_j_in_rule_local_center_count: int
    spectral_polarization_j_matched_count: int
    forced_j_found: bool
    reason_for_forced_j_failure: str
    compatible_j_diagnostics: tuple[FloquetAlphaNoncommutingJDiagnostic, ...]
    load_bearing_qca_bridge: bool = False


def _mode_count(dimension: int) -> int:
    if dimension % 2:
        raise ValueError("real carrier dimension must be even")
    return dimension // 2


def _validate_signs(signs: tuple[int, ...], *, dimension: int = 10) -> None:
    if len(signs) != _mode_count(dimension):
        raise ValueError("orientation sign count must match the number of mode-pairs")
    if any(sign not in {-1, 1} for sign in signs):
        raise ValueError("orientation signs must be +/-1")


def floquet_alpha_noncommuting_mode_mapping(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> dict[int, int]:
    """Return the block-preserving cycle/swap support of the signed twist."""

    alpha_modes = candidate.pattern.alpha_modes
    eta_modes = candidate.pattern.eta_modes
    mode_mapping: dict[int, int] = {}
    for source, target in zip(alpha_modes, (*alpha_modes[1:], alpha_modes[0]), strict=True):
        mode_mapping[source] = target
    for source, target in zip(eta_modes, reversed(eta_modes), strict=True):
        mode_mapping[source] = target
    return mode_mapping


def signed_orientation_twist_operator(
    mode_mapping: dict[int, int],
    orientation_signs: tuple[int, ...],
    *,
    dimension: int = 10,
) -> sp.Matrix:
    """Return a signed pair permutation preserving a hidden orientation.

    A mode edge with equal endpoint signs is complex-linear with respect to the
    standard pair orientation.  A sign-changing edge is complex-antilinear:
    ``(x, y) -> (x, -y)``.  The resulting matrix is still real orthogonal and
    block-preserving when the mode permutation preserves the alpha/eta sectors.
    """

    _validate_signs(orientation_signs, dimension=dimension)
    mode_count = _mode_count(dimension)
    if set(mode_mapping) != set(range(mode_count)):
        raise ValueError("mode mapping must be a full mode-pair permutation")
    if set(mode_mapping.values()) != set(range(mode_count)):
        raise ValueError("mode mapping must be a full mode-pair permutation")

    matrix = sp.zeros(dimension)
    for source, target in mode_mapping.items():
        matrix[target, source] = 1
        y_sign = 1 if orientation_signs[source] == orientation_signs[target] else -1
        matrix[target + mode_count, source + mode_count] = y_sign
    return matrix


def floquet_alpha_noncommuting_twist_operator(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> sp.Matrix:
    return signed_orientation_twist_operator(
        floquet_alpha_noncommuting_mode_mapping(candidate),
        candidate.orientation_signs,
    )


def floquet_alpha_noncommuting_twist_layer(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> RuleLayerInput:
    return RuleLayerInput(
        name=f"{candidate.name}_layer",
        matrix=floquet_alpha_noncommuting_twist_operator(candidate),
        support=(0,),
        locality_radius=0,
    )


def floquet_alpha_noncommuting_candidates(
    *,
    pattern_index: int | None = None,
) -> tuple[FloquetAlphaNoncommutingCandidate, ...]:
    """Return the first exact block-preserving noncommuting twist per pattern."""

    candidates: list[FloquetAlphaNoncommutingCandidate] = []
    for pattern in floquet_alpha_candidates():
        if pattern_index is not None and pattern.pattern_index != pattern_index:
            continue
        signs = [1] * 5
        signs[pattern.alpha_modes[-1]] = -1
        signs[pattern.eta_modes[-1]] = -1
        candidates.append(
            FloquetAlphaNoncommutingCandidate(
                pattern=pattern,
                orientation_signs=tuple(signs),
            )
        )
    return tuple(candidates)


def floquet_alpha_noncommuting_rule_to_verdict(
    candidate: FloquetAlphaNoncommutingCandidate,
    *,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (
            floquet_alpha_layer(candidate.pattern),
            floquet_alpha_noncommuting_twist_layer(candidate),
        ),
        rule_name=candidate.name,
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def _orientation_nonconstant(
    signs: tuple[int, ...],
    modes: tuple[int, ...],
) -> bool:
    return len({signs[mode] for mode in modes}) > 1


def _matrix_in_span(matrix: sp.Matrix, basis: tuple[sp.Matrix, ...]) -> bool:
    return matrix_span_rank((*basis, matrix)) == matrix_span_rank(basis)


def _pair_orientation_signs(matrix: sp.Matrix, *, dimension: int = 10) -> tuple[int, ...]:
    mode_count = _mode_count(dimension)
    signs: list[int] = []
    for mode in range(mode_count):
        top_right = sp.simplify(matrix[mode, mode + mode_count])
        bottom_left = sp.simplify(matrix[mode + mode_count, mode])
        if top_right == -1 and bottom_left == 1:
            signs.append(1)
        elif top_right == 1 and bottom_left == -1:
            signs.append(-1)
        else:
            return ()
    return tuple(signs)


def _route_label(
    result: RuleToVerdictResult,
    *,
    preserves_alpha: bool,
    preserves_eta: bool,
    commutes: bool,
) -> RouteLabel:
    if result.pass_rule_to_bridge:
        return "bridge_candidate"
    if commutes:
        return "commuting_not_route_1"
    if not preserves_alpha or not preserves_eta:
        return "coarse_split_not_preserved"
    if not result.center_solved:
        return "center_not_solved"
    if not result.complementary_rank_6_4_pairs:
        return "coarse_center_missing"
    if result.lower_rank_central_idempotents:
        return "lower_rank_center_generated"
    if (
        result.compatible_j_solved
        and result.compatible_j_moduli_dimension == 0
        and result.compatible_complex_structures
    ):
        return "coarse_center_preserved_compatible_j_not_rule_generated"
    return "coarse_center_preserved_j_not_forced"


def _j_gap_failure_reason(
    result: RuleToVerdictResult,
    diagnostics: tuple[FloquetAlphaNoncommutingJDiagnostic, ...],
) -> str:
    if result.forced_j_found:
        return "forced_j_found"
    if not result.compatible_j_solved:
        return "compatible_j_not_solved"
    if result.compatible_j_moduli_dimension != 0:
        return "compatible_j_not_zero_dimensional"
    if not diagnostics:
        return "no_compatible_j_candidates"
    if not any(item.in_generated_algebra for item in diagnostics) and not any(
        item.in_rule_local_center for item in diagnostics
    ):
        return "compatible_j_finite_but_not_generated_or_rule_local"
    if not result.generated_j_solved:
        return "generated_j_not_solved"
    if not result.local_compatible_j_solved:
        return "rule_local_j_not_solved"
    return "compatible_j_not_matched_by_forced_pair"


def floquet_alpha_noncommuting_certificate(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> FloquetAlphaNoncommutingCertificate:
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate.pattern)
    result = floquet_alpha_noncommuting_rule_to_verdict(candidate)
    one = identity(10)
    zero = sp.zeros(10)
    commutes = is_zero_matrix(commutator(u1, u2))
    preserves_alpha = sp.simplify(u2 * alpha_projector * u2.T - alpha_projector) == zero
    preserves_eta = sp.simplify(u2 * eta_projector * u2.T - eta_projector) == zero

    return FloquetAlphaNoncommutingCertificate(
        candidate_name=candidate.name,
        base_candidate_name=candidate.pattern.name,
        twist_layer_name=floquet_alpha_noncommuting_twist_layer(candidate).name,
        orientation_signs=candidate.orientation_signs,
        mode_mapping=tuple(sorted(floquet_alpha_noncommuting_mode_mapping(candidate).items())),
        u1_u2_commute=commutes,
        u2_real_orthogonal=u2.T * u2 == one,
        u2_preserves_alpha_projector=preserves_alpha,
        u2_preserves_eta_projector=preserves_eta,
        alpha_orientation_nonconstant=_orientation_nonconstant(
            candidate.orientation_signs,
            candidate.pattern.alpha_modes,
        ),
        eta_orientation_nonconstant=_orientation_nonconstant(
            candidate.orientation_signs,
            candidate.pattern.eta_modes,
        ),
        generated_algebra_dimension=result.generated_algebra_dimension,
        center_dimension=result.center_dimension,
        center_solved=result.center_solved,
        central_idempotent_ranks=tuple(item.rank for item in result.central_idempotents),
        complementary_rank_6_4_pairs=result.complementary_rank_6_4_pairs,
        lower_rank_central_idempotents=len(result.lower_rank_central_idempotents),
        compatible_centralizer_dimension=result.compatible_centralizer_dimension,
        compatible_j_solved=result.compatible_j_solved,
        compatible_j_moduli_dimension=result.compatible_j_moduli_dimension,
        compatible_complex_structure_count=len(result.compatible_complex_structures),
        generated_j_solved=result.generated_j_solved,
        generated_j_moduli_dimension=result.generated_j_moduli_dimension,
        generated_complex_structure_count=len(result.generated_complex_structures),
        local_compatible_operator_dimension=result.local_compatible_operator_dimension,
        local_compatible_j_solved=result.local_compatible_j_solved,
        local_compatible_j_moduli_dimension=result.local_compatible_j_moduli_dimension,
        local_compatible_complex_structure_count=len(result.local_compatible_complex_structures),
        forced_j_found=result.forced_j_found,
        pass_strict_rule_to_bridge=result.pass_rule_to_bridge,
        rule_verdict=result.verdict,
        route_label=_route_label(
            result,
            preserves_alpha=preserves_alpha,
            preserves_eta=preserves_eta,
            commutes=commutes,
        ),
    )


def floquet_alpha_noncommuting_j_gap_certificate(
    candidate: FloquetAlphaNoncommutingCandidate,
) -> FloquetAlphaNoncommutingJGapCertificate:
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    result = floquet_alpha_noncommuting_rule_to_verdict(candidate)
    generated_basis = generated_algebra_basis((u1, u2))
    local_center_basis = center_basis_of_algebra(generated_basis)
    spectral_j = floquet_alpha_canonical_j(candidate.pattern)
    zero = sp.zeros(10)
    one = identity(10)

    diagnostics: list[FloquetAlphaNoncommutingJDiagnostic] = []
    for index, compatible_j in enumerate(result.compatible_complex_structures):
        matrix = compatible_j.matrix
        diagnostics.append(
            FloquetAlphaNoncommutingJDiagnostic(
                index=index,
                expression=compatible_j.expression,
                pair_orientation_signs=_pair_orientation_signs(matrix),
                in_generated_algebra=_matrix_in_span(matrix, generated_basis),
                in_rule_local_center=_matrix_in_span(matrix, local_center_basis),
                equals_spectral_polarization_j=sp.simplify(matrix - spectral_j) == zero,
                equals_negative_spectral_polarization_j=(
                    sp.simplify(matrix + spectral_j) == zero
                ),
                commutes_with_u1=is_zero_matrix(commutator(matrix, u1)),
                commutes_with_u2=is_zero_matrix(commutator(matrix, u2)),
                squares_to_minus_identity=sp.simplify(matrix * matrix + one) == zero,
                orthogonal=sp.simplify(matrix.T * matrix - one) == zero,
                matrix=matrix,
            )
        )

    diagnostic_tuple = tuple(diagnostics)
    return FloquetAlphaNoncommutingJGapCertificate(
        candidate_name=candidate.name,
        compatible_j_count=len(result.compatible_complex_structures),
        generated_algebra_dimension=result.generated_algebra_dimension,
        center_dimension=result.center_dimension,
        compatible_centralizer_dimension=result.compatible_centralizer_dimension,
        compatible_j_solved=result.compatible_j_solved,
        compatible_j_moduli_dimension=result.compatible_j_moduli_dimension,
        generated_j_solved=result.generated_j_solved,
        generated_complex_structure_count=len(result.generated_complex_structures),
        local_compatible_j_solved=result.local_compatible_j_solved,
        local_compatible_j_moduli_dimension=result.local_compatible_j_moduli_dimension,
        local_compatible_complex_structure_count=len(result.local_compatible_complex_structures),
        compatible_j_in_generated_algebra_count=sum(
            item.in_generated_algebra for item in diagnostic_tuple
        ),
        compatible_j_in_rule_local_center_count=sum(
            item.in_rule_local_center for item in diagnostic_tuple
        ),
        spectral_polarization_j_matched_count=sum(
            item.equals_spectral_polarization_j
            or item.equals_negative_spectral_polarization_j
            for item in diagnostic_tuple
        ),
        forced_j_found=result.forced_j_found,
        reason_for_forced_j_failure=_j_gap_failure_reason(result, diagnostic_tuple),
        compatible_j_diagnostics=diagnostic_tuple,
    )
