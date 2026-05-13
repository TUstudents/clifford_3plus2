"""Certificates for whether a candidate J is generated and forced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.search.gate_words import (
    PrimitiveGate,
    scan_gate_words,
    standard_period_four_primitives,
)


ForcedJVerdict = Literal["forced_j", "candidate_only", "falsified"]


@dataclass(frozen=True)
class ForcedJCertificate:
    candidate_name: str
    generated_by_gate_word: bool
    gate_word: tuple[str, ...]
    is_real_orthogonal: bool
    determinant: int
    squares_to_minus_identity: bool
    equals_standard_j: bool
    rank_one_pair_rotations_addressable: bool
    qca_forces_j: bool
    verdict: ForcedJVerdict


def is_real_orthogonal(matrix: sp.Matrix, metric: sp.Matrix | None = None) -> bool:
    metric = metric or identity(matrix.rows)
    return matrix.T * metric * matrix == metric


def is_complex_structure(matrix: sp.Matrix, metric: sp.Matrix | None = None) -> bool:
    return (
        is_real_orthogonal(matrix, metric)
        and matrix * matrix == -identity(matrix.rows)
        and matrix.det() == 1
    )


def detect_rank_one_pair_addressability(primitives: tuple[PrimitiveGate, ...]) -> bool:
    return any(gate.independently_addressable for gate in primitives)


def find_gate_word_for_standard_j(
    primitives: tuple[PrimitiveGate, ...], max_depth: int = 4
) -> tuple[tuple[str, ...], sp.Matrix] | None:
    target = standard_real_carrier().complex_structure
    for word, matrix in scan_gate_words(primitives, max_depth):
        if matrix == target:
            return word, matrix
    return None


def j_certificate(
    candidate: sp.Matrix | None = None,
    *,
    candidate_name: str = "standard_clock_j",
    primitives: tuple[PrimitiveGate, ...] | None = None,
    max_depth: int = 4,
    qca_rule_forces_word: bool = False,
) -> ForcedJCertificate:
    carrier = standard_real_carrier()
    candidate = candidate if candidate is not None else carrier.complex_structure
    primitives = primitives if primitives is not None else standard_period_four_primitives()
    word_match = find_gate_word_for_standard_j(primitives, max_depth)
    generated_by_gate_word = word_match is not None and candidate == carrier.complex_structure
    gate_word = word_match[0] if word_match is not None else ()
    rank_one_addressable = detect_rank_one_pair_addressability(primitives)
    determinant = int(candidate.det())
    orthogonal = is_real_orthogonal(candidate, carrier.metric)
    squares = candidate * candidate == -identity(candidate.rows)
    equals_standard = candidate == carrier.complex_structure
    qca_forces_j = (
        generated_by_gate_word
        and qca_rule_forces_word
        and not rank_one_addressable
        and orthogonal
        and squares
        and equals_standard
    )

    if rank_one_addressable:
        verdict: ForcedJVerdict = "falsified"
    elif qca_forces_j:
        verdict = "forced_j"
    else:
        verdict = "candidate_only"

    return ForcedJCertificate(
        candidate_name=candidate_name,
        generated_by_gate_word=generated_by_gate_word,
        gate_word=gate_word,
        is_real_orthogonal=orthogonal,
        determinant=determinant,
        squares_to_minus_identity=squares,
        equals_standard_j=equals_standard,
        rank_one_pair_rotations_addressable=rank_one_addressable,
        qca_forces_j=qca_forces_j,
        verdict=verdict,
    )


def certificate_to_dict(certificate: ForcedJCertificate) -> dict[str, object]:
    return {
        "candidate_name": certificate.candidate_name,
        "generated_by_gate_word": certificate.generated_by_gate_word,
        "gate_word": list(certificate.gate_word),
        "is_real_orthogonal": certificate.is_real_orthogonal,
        "determinant": certificate.determinant,
        "squares_to_minus_identity": certificate.squares_to_minus_identity,
        "equals_standard_j": certificate.equals_standard_j,
        "rank_one_pair_rotations_addressable": (
            certificate.rank_one_pair_rotations_addressable
        ),
        "qca_forces_j": certificate.qca_forces_j,
        "forced_j_verdict": certificate.verdict,
        "forced_j_check_passed": (
            certificate.generated_by_gate_word
            and certificate.is_real_orthogonal
            and certificate.determinant == 1
            and certificate.squares_to_minus_identity
            and certificate.equals_standard_j
        ),
        "load_bearing_qca_bridge": certificate.qca_forces_j,
    }
