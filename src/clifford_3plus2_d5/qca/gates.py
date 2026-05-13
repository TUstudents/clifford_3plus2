"""Exact local gate primitives for finite-depth QCA candidates."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import realify
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.sm.classification import GateClass, classify_real_gate
from clifford_3plus2_d5.sm.embedding import matrix_unit


@dataclass(frozen=True)
class QCALocalGate:
    name: str
    support: tuple[int, ...]
    matrix: sp.Matrix
    locality_radius: int
    source: str = "candidate"


def support_radius(support: tuple[int, ...]) -> int:
    if not support:
        raise ValueError("support must be nonempty")
    return max(support) - min(support)


def support_is_valid(support: tuple[int, ...]) -> bool:
    return bool(support) and all(site >= 0 for site in support)


def gate_has_valid_locality(gate: QCALocalGate) -> bool:
    return (
        support_is_valid(gate.support)
        and gate.locality_radius >= 0
        and gate.locality_radius >= support_radius(gate.support)
    )


def is_real_matrix(matrix: sp.Matrix) -> bool:
    return all(value.is_real is True for value in matrix)


def is_real_orthogonal(matrix: sp.Matrix) -> bool:
    return (
        matrix.shape == (10, 10)
        and is_real_matrix(matrix)
        and matrix.T * matrix == identity(10)
    )


def internal_classification(gate: QCALocalGate) -> GateClass:
    return classify_real_gate(gate.matrix)


def gate_certificate(gate: QCALocalGate) -> dict[str, object]:
    return {
        "name": gate.name,
        "support": list(gate.support),
        "matrix_shape": list(gate.matrix.shape),
        "is_real": is_real_matrix(gate.matrix),
        "is_orthogonal": is_real_orthogonal(gate.matrix),
        "locality_radius": gate.locality_radius,
        "support_is_valid": support_is_valid(gate.support),
        "locality_check_passed": gate_has_valid_locality(gate),
        "internal_classification": internal_classification(gate),
        "internal_action_safe": internal_classification(gate) == "safe_sm_commutant",
        "source": gate.source,
    }


def global_clock_tick_gate() -> QCALocalGate:
    carrier = standard_real_carrier()
    return QCALocalGate(
        name="global_clock_tick",
        support=(0,),
        matrix=carrier.complex_structure,
        locality_radius=0,
        source="candidate_period_four_clock",
    )


def rank_one_color_shift_gate() -> QCALocalGate:
    return QCALocalGate(
        name="rank_one_color_shift",
        support=(0, 1),
        matrix=realify(sp.diag(1, 0, 0, 0, 0)),
        locality_radius=1,
        source="falsifier",
    )


def rank_one_weak_shift_gate() -> QCALocalGate:
    return QCALocalGate(
        name="rank_one_weak_shift",
        support=(0, 1),
        matrix=realify(sp.diag(0, 0, 0, 1, 0)),
        locality_radius=1,
        source="falsifier",
    )


def block_mixing_shift_gate() -> QCALocalGate:
    return QCALocalGate(
        name="block_mixing_shift",
        support=(0, 1),
        matrix=realify(matrix_unit(0, 3) + matrix_unit(3, 0)),
        locality_radius=1,
        source="falsifier",
    )
