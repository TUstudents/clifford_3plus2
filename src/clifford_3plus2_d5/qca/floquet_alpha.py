"""Floquet-alpha physical primitive family."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.rule_verdict import RuleLayerInput, RuleToVerdictResult, rule_to_verdict


ALPHA_PHASE = sp.Rational(2, 3) * sp.pi
ETA_PHASE = sp.Rational(1, 2) * sp.pi


@dataclass(frozen=True)
class FloquetAlphaCandidate:
    pattern_index: int
    alpha_modes: tuple[int, ...]
    eta_modes: tuple[int, ...]

    @property
    def name(self) -> str:
        alpha = "_".join(str(mode + 1) for mode in self.alpha_modes)
        return f"floquet_alpha_pattern_{self.pattern_index}_alpha_{alpha}"


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


def search_floquet_alpha(
    candidates: Iterable[FloquetAlphaCandidate] | None = None,
) -> tuple[RuleToVerdictResult, ...]:
    candidates = candidates if candidates is not None else floquet_alpha_candidates()
    return tuple(floquet_alpha_rule_to_verdict(candidate) for candidate in candidates)
