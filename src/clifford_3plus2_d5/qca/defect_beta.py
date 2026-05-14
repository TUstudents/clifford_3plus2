"""Defect-beta monodromy primitive family."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import commutator, identity, is_zero_matrix
from clifford_3plus2_d5.qca.floquet_alpha import ALPHA_PHASE, ETA_PHASE, pair_rotation
from clifford_3plus2_d5.qca.rule_verdict import RuleLayerInput, RuleToVerdictResult, rule_to_verdict


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
    transition_count: int
    monodromy_computed_from_transitions: bool
    omega_projector_rank: int
    i_projector_rank: int
    spectral_projectors_are_idempotent: bool
    spectral_projectors_are_complementary: bool
    canonical_j_generated_by_monodromy: bool
    canonical_j_squared_minus_identity: bool
    canonical_j_orthogonal: bool
    canonical_j_commutes_with_projectors: bool
    central_idempotent_ranks: tuple[int, ...]
    complementary_rank_6_4_pairs: int
    lower_rank_central_idempotents: int
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


def defect_beta_transition_functions(
    candidate: DefectBetaCandidate,
) -> tuple[RuleLayerInput, ...]:
    """Return wall-cycle transitions whose product is the defect monodromy."""

    half_turn = identity(10)
    for mode in candidate.omega_modes:
        half_turn = pair_rotation(mode, ALPHA_PHASE / 2) * half_turn
    for mode in candidate.i_modes:
        half_turn = pair_rotation(mode, ETA_PHASE / 2) * half_turn
    half_turn = _simplify_matrix(half_turn)
    return (
        RuleLayerInput(
            name=f"{candidate.name}_wall_entry",
            matrix=half_turn,
            support=(0, 1),
            locality_radius=1,
        ),
        RuleLayerInput(
            name=f"{candidate.name}_wall_exit",
            matrix=half_turn,
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
    monodromy = defect_beta_monodromy_operator(candidate)
    one = identity(10)
    omega_projector, i_projector = defect_beta_spectral_projectors(candidate)
    omega_j = (sp.Rational(2) / sp.sqrt(3)) * (
        monodromy + sp.Rational(1, 2) * one
    ) * omega_projector
    i_j = monodromy * i_projector
    return _simplify_matrix(omega_j + i_j)


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
    omega_projector, i_projector = defect_beta_spectral_projectors(candidate)
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
    canonical_j_squared = _simplify_matrix(canonical_j * canonical_j + one) == zero
    canonical_j_orthogonal = _simplify_matrix(canonical_j.T * canonical_j - one) == zero
    canonical_j_commutes = is_zero_matrix(commutator(canonical_j, omega_projector)) and is_zero_matrix(
        commutator(canonical_j, i_projector)
    )
    beta_monodromy_passed = (
        len(defect_beta_transition_functions(candidate)) == 2
        and spectral_projectors_are_idempotent
        and spectral_projectors_are_complementary
        and omega_projector.rank() == 6
        and i_projector.rank() == 4
        and canonical_j_squared
        and canonical_j_orthogonal
        and canonical_j_commutes
        and verdict.complementary_rank_6_4_pairs > 0
        and not verdict.lower_rank_central_idempotents
    )
    return DefectBetaCertificate(
        candidate_name=candidate.name,
        transition_count=len(defect_beta_transition_functions(candidate)),
        monodromy_computed_from_transitions=True,
        omega_projector_rank=omega_projector.rank(),
        i_projector_rank=i_projector.rank(),
        spectral_projectors_are_idempotent=spectral_projectors_are_idempotent,
        spectral_projectors_are_complementary=spectral_projectors_are_complementary,
        canonical_j_generated_by_monodromy=True,
        canonical_j_squared_minus_identity=canonical_j_squared,
        canonical_j_orthogonal=canonical_j_orthogonal,
        canonical_j_commutes_with_projectors=canonical_j_commutes,
        central_idempotent_ranks=tuple(item.rank for item in verdict.central_idempotents),
        complementary_rank_6_4_pairs=verdict.complementary_rank_6_4_pairs,
        lower_rank_central_idempotents=len(verdict.lower_rank_central_idempotents),
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
