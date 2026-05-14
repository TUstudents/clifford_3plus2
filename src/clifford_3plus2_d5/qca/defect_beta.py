"""Defect-beta monodromy primitive family."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.qca.floquet_alpha import (
    ALPHA_PHASE,
    ETA_PHASE,
    FLOQUET_ALPHA_ALPHA_SECTOR_CENTRALIZER_DIMENSION,
    FLOQUET_ALPHA_COMPATIBLE_CENTRALIZER_DIMENSION,
    FLOQUET_ALPHA_COMPATIBLE_J_MODULI_DIMENSION,
    FLOQUET_ALPHA_EXACT_WORKING_FIELD,
    FLOQUET_ALPHA_ETA_SECTOR_CENTRALIZER_DIMENSION,
    pair_rotation,
)
from clifford_3plus2_d5.qca.rule_verdict import RuleLayerInput, RuleToVerdictResult, rule_to_verdict

DEFECT_BETA_EXACT_WORKING_FIELD = FLOQUET_ALPHA_EXACT_WORKING_FIELD
DEFECT_BETA_SCALED_RELATION = "K_omega=(2M+I)P_omega, K_omega^2=-3P_omega"
DEFECT_BETA_OMEGA_SECTOR_CENTRALIZER_DIMENSION = (
    FLOQUET_ALPHA_ALPHA_SECTOR_CENTRALIZER_DIMENSION
)
DEFECT_BETA_I_SECTOR_CENTRALIZER_DIMENSION = FLOQUET_ALPHA_ETA_SECTOR_CENTRALIZER_DIMENSION
DEFECT_BETA_COMPATIBLE_CENTRALIZER_DIMENSION = FLOQUET_ALPHA_COMPATIBLE_CENTRALIZER_DIMENSION
DEFECT_BETA_COMPATIBLE_J_MODULI_DIMENSION = FLOQUET_ALPHA_COMPATIBLE_J_MODULI_DIMENSION


@dataclass(frozen=True)
class DefectBetaCandidate:
    pattern_index: int
    omega_modes: tuple[int, ...]
    i_modes: tuple[int, ...]

    @property
    def name(self) -> str:
        omega = "_".join(str(mode + 1) for mode in self.omega_modes)
        return f"defect_beta_pattern_{self.pattern_index}_omega_{omega}"


@dataclass(frozen=True)
class DefectBetaCertificate:
    candidate_name: str
    exact_working_field: str
    transition_count: int
    monodromy_computed_from_transitions: bool
    entry_exit_transitions_distinct: bool
    transition_determinants: tuple[int, ...]
    clutching_reflection_determinant: int
    clutching_identity_passed: bool
    omega_projector_rank: int
    i_projector_rank: int
    spectral_projectors_are_idempotent: bool
    spectral_projectors_are_complementary: bool
    scaled_omega_relation: str
    scaled_omega_square_relation: bool
    scaled_omega_orthogonality_relation: bool
    scaled_omega_commutes_with_projectors: bool
    i_j_square_relation: bool
    i_j_orthogonality_relation: bool
    scaled_monodromy_certified: bool
    normalized_j_requires_sqrt3: bool
    canonical_j_generated_by_monodromy: bool
    canonical_j_squared_minus_identity: bool
    canonical_j_orthogonal: bool
    canonical_j_commutes_with_projectors: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
    generated_j_moduli_dimension: int | None
    omega_sector_centralizer_dimension: int
    i_sector_centralizer_dimension: int
    compatible_centralizer_dimension: int
    compatible_j_moduli_dimension: int
    strict_compatible_j_forced: bool
    compatible_j_solved: bool
    beta_monodromy_passed: bool
    pass_strict_rule_to_bridge: bool
    verdict: str
    load_bearing_qca_bridge: bool = False


def _simplify_matrix(matrix: sp.Matrix) -> sp.Matrix:
    return matrix.applyfunc(sp.simplify)


def defect_beta_candidates() -> tuple[DefectBetaCandidate, ...]:
    candidates = []
    all_modes = tuple(range(5))
    for pattern_index, omega_modes in enumerate(combinations(all_modes, 3)):
        omega_tuple = tuple(omega_modes)
        i_tuple = tuple(mode for mode in all_modes if mode not in omega_tuple)
        candidates.append(
            DefectBetaCandidate(
                pattern_index=pattern_index,
                omega_modes=omega_tuple,
                i_modes=i_tuple,
            )
        )
    return tuple(candidates)


def defect_beta_clutching_reflection() -> sp.Matrix:
    """Return the orientation-reversing wall chart change C.

    The reflection flips the clock-x half of the real carrier. It satisfies
    C^2 = I and conjugates every pair rotation to its inverse, so entry and
    exit can live in different wall cosets while their product is a monodromy.
    """

    reflection = identity(10)
    for index in range(5):
        reflection[index, index] = -1
    return reflection


def defect_beta_monodromy_core(candidate: DefectBetaCandidate) -> sp.Matrix:
    monodromy = identity(10)
    for mode in candidate.omega_modes:
        monodromy = pair_rotation(mode, ALPHA_PHASE) * monodromy
    for mode in candidate.i_modes:
        monodromy = pair_rotation(mode, ETA_PHASE) * monodromy
    return _simplify_matrix(monodromy)


def defect_beta_transition_functions(
    candidate: DefectBetaCandidate,
) -> tuple[RuleLayerInput, ...]:
    """Return wall-cycle transitions whose product is the defect monodromy."""

    monodromy_core = defect_beta_monodromy_core(candidate)
    clutching = defect_beta_clutching_reflection()
    entry = clutching
    exit = _simplify_matrix(monodromy_core * clutching)
    return (
        RuleLayerInput(
            name=f"{candidate.name}_wall_entry",
            matrix=entry,
            support=(0, 1),
            locality_radius=1,
        ),
        RuleLayerInput(
            name=f"{candidate.name}_wall_exit",
            matrix=exit,
            support=(1, 0),
            locality_radius=1,
        ),
    )


def defect_beta_monodromy_operator(candidate: DefectBetaCandidate) -> sp.Matrix:
    monodromy = identity(10)
    for transition in defect_beta_transition_functions(candidate):
        monodromy = transition.matrix * monodromy
    return _simplify_matrix(monodromy)


def defect_beta_monodromy_layer(candidate: DefectBetaCandidate) -> RuleLayerInput:
    return RuleLayerInput(
        name=f"{candidate.name}_round_trip_monodromy",
        matrix=defect_beta_monodromy_operator(candidate),
        support=(0,),
        locality_radius=0,
    )


def defect_beta_spectral_projectors(
    candidate: DefectBetaCandidate,
) -> tuple[sp.Matrix, sp.Matrix]:
    monodromy = defect_beta_monodromy_operator(candidate)
    one = identity(10)
    omega_projector = _simplify_matrix((monodromy + one) * (monodromy * monodromy + one))
    i_projector = _simplify_matrix(one - omega_projector)
    return omega_projector, i_projector


def defect_beta_canonical_j(candidate: DefectBetaCandidate) -> sp.Matrix:
    """Return the normalized monodromy complex structure over ``QQ(sqrt(3))``."""

    monodromy = defect_beta_monodromy_operator(candidate)
    _, i_projector = defect_beta_spectral_projectors(candidate)
    omega_j = defect_beta_scaled_omega_operator(candidate) / sp.sqrt(3)
    i_j = monodromy * i_projector
    return _simplify_matrix(omega_j + i_j)


def defect_beta_scaled_omega_operator(candidate: DefectBetaCandidate) -> sp.Matrix:
    """Return ``K_omega=(2M+I)P_omega`` without dividing by ``sqrt(3)``."""

    monodromy = defect_beta_monodromy_operator(candidate)
    omega_projector, _ = defect_beta_spectral_projectors(candidate)
    return _simplify_matrix((2 * monodromy + identity(10)) * omega_projector)


def defect_beta_rule_to_verdict(
    candidate: DefectBetaCandidate,
    *,
    max_center_solve_dimension: int = 8,
    max_j_solve_dimension: int = 8,
) -> RuleToVerdictResult:
    return rule_to_verdict(
        (defect_beta_monodromy_layer(candidate),),
        rule_name=candidate.name,
        max_center_solve_dimension=max_center_solve_dimension,
        max_j_solve_dimension=max_j_solve_dimension,
    )


def defect_beta_certificate(candidate: DefectBetaCandidate) -> DefectBetaCertificate:
    verdict = defect_beta_rule_to_verdict(candidate)
    transitions = defect_beta_transition_functions(candidate)
    monodromy_core = defect_beta_monodromy_core(candidate)
    clutching = defect_beta_clutching_reflection()
    computed_monodromy = identity(10)
    for transition in transitions:
        computed_monodromy = transition.matrix * computed_monodromy
    clutching_identity_passed = (
        _simplify_matrix(clutching * clutching - identity(10)) == sp.zeros(10)
        and _simplify_matrix(computed_monodromy - monodromy_core) == sp.zeros(10)
    )
    omega_projector, i_projector = defect_beta_spectral_projectors(candidate)
    scaled_omega = defect_beta_scaled_omega_operator(candidate)
    i_j = _simplify_matrix(defect_beta_monodromy_operator(candidate) * i_projector)
    canonical_j = defect_beta_canonical_j(candidate)
    one = identity(10)
    zero = sp.zeros(10)
    spectral_projectors_are_idempotent = (
        _simplify_matrix(omega_projector * omega_projector - omega_projector) == zero
        and _simplify_matrix(i_projector * i_projector - i_projector) == zero
    )
    spectral_projectors_are_complementary = (
        _simplify_matrix(omega_projector + i_projector - one) == zero
        and _simplify_matrix(omega_projector * i_projector) == zero
        and _simplify_matrix(i_projector * omega_projector) == zero
    )
    scaled_omega_square = (
        _simplify_matrix(scaled_omega * scaled_omega + 3 * omega_projector) == zero
    )
    scaled_omega_orthogonal = (
        _simplify_matrix(scaled_omega.T * scaled_omega - 3 * omega_projector) == zero
    )
    scaled_omega_commutes = is_zero_matrix(
        commutator(scaled_omega, omega_projector)
    ) and is_zero_matrix(commutator(scaled_omega, i_projector))
    i_j_square = _simplify_matrix(i_j * i_j + i_projector) == zero
    i_j_orthogonal = _simplify_matrix(i_j.T * i_j - i_projector) == zero
    scaled_monodromy_certified = (
        scaled_omega_square
        and scaled_omega_orthogonal
        and scaled_omega_commutes
        and i_j_square
        and i_j_orthogonal
    )
    canonical_j_squared = _simplify_matrix(canonical_j * canonical_j + one) == zero
    canonical_j_orthogonal = _simplify_matrix(canonical_j.T * canonical_j - one) == zero
    canonical_j_commutes = is_zero_matrix(commutator(canonical_j, omega_projector)) and is_zero_matrix(
        commutator(canonical_j, i_projector)
    )
    beta_monodromy_passed = (
        len(transitions) == 2
        and transitions[0].matrix != transitions[1].matrix
        and all(transition.matrix.det() == -1 for transition in transitions)
        and clutching.det() == -1
        and clutching_identity_passed
        and spectral_projectors_are_idempotent
        and spectral_projectors_are_complementary
        and omega_projector.rank() == 6
        and i_projector.rank() == 4
        and scaled_monodromy_certified
        and verdict.complementary_rank_6_4_pairs > 0
        and not verdict.lower_rank_central_idempotents
    )
    return DefectBetaCertificate(
        candidate_name=candidate.name,
        exact_working_field=DEFECT_BETA_EXACT_WORKING_FIELD,
        transition_count=len(transitions),
        monodromy_computed_from_transitions=True,
        entry_exit_transitions_distinct=transitions[0].matrix != transitions[1].matrix,
        transition_determinants=tuple(int(transition.matrix.det()) for transition in transitions),
        clutching_reflection_determinant=int(clutching.det()),
        clutching_identity_passed=clutching_identity_passed,
        omega_projector_rank=omega_projector.rank(),
        i_projector_rank=i_projector.rank(),
        spectral_projectors_are_idempotent=spectral_projectors_are_idempotent,
        spectral_projectors_are_complementary=spectral_projectors_are_complementary,
        scaled_omega_relation=DEFECT_BETA_SCALED_RELATION,
        scaled_omega_square_relation=scaled_omega_square,
        scaled_omega_orthogonality_relation=scaled_omega_orthogonal,
        scaled_omega_commutes_with_projectors=scaled_omega_commutes,
        i_j_square_relation=i_j_square,
        i_j_orthogonality_relation=i_j_orthogonal,
        scaled_monodromy_certified=scaled_monodromy_certified,
        normalized_j_requires_sqrt3=True,
        canonical_j_generated_by_monodromy=True,
        canonical_j_squared_minus_identity=canonical_j_squared,
        canonical_j_orthogonal=canonical_j_orthogonal,
        canonical_j_commutes_with_projectors=canonical_j_commutes,
        central_idempotent_ranks=tuple(item.rank for item in verdict.central_idempotents),
        complementary_rank_6_4_pairs=verdict.complementary_rank_6_4_pairs,
        lower_rank_central_idempotents=len(verdict.lower_rank_central_idempotents),
        generated_j_moduli_dimension=verdict.generated_j_moduli_dimension,
        omega_sector_centralizer_dimension=DEFECT_BETA_OMEGA_SECTOR_CENTRALIZER_DIMENSION,
        i_sector_centralizer_dimension=DEFECT_BETA_I_SECTOR_CENTRALIZER_DIMENSION,
        compatible_centralizer_dimension=verdict.compatible_centralizer_dimension,
        compatible_j_moduli_dimension=(
            verdict.compatible_j_moduli_dimension
            if verdict.compatible_j_moduli_dimension is not None
            else DEFECT_BETA_COMPATIBLE_J_MODULI_DIMENSION
        ),
        strict_compatible_j_forced=verdict.forced_j_found,
        compatible_j_solved=verdict.compatible_j_solved,
        beta_monodromy_passed=beta_monodromy_passed,
        pass_strict_rule_to_bridge=verdict.pass_rule_to_bridge,
        verdict=(
            "monodromy_j_produced_not_strictly_unique"
            if beta_monodromy_passed and not verdict.pass_rule_to_bridge
            else verdict.verdict
        ),
    )
