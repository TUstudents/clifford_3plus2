"""Classify real internal gates against the SM commutant oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import (
    complex_from_real,
    is_scalar_block,
    realify,
    safe_commutant_closure_examples_pass,
    sm_commutant_basis_matches_expected,
)
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.obstruction_r10.sm.embedding import complex_projector_2, complex_projector_3, matrix_unit


GateClass = Literal[
    "safe_sm_commutant",
    "block_mixing_fail",
    "color_breaking_fail",
    "weak_breaking_fail",
    "antilinear_fail",
    "unknown_fail",
]


@dataclass(frozen=True)
class NamedRealGate:
    name: str
    matrix: sp.Matrix
    expected_class: GateClass | None = None


@dataclass(frozen=True)
class ClassifiedRealGate:
    name: str
    classification: GateClass
    expected_class: GateClass | None
    matches_expected: bool


@dataclass(frozen=True)
class GateClassificationCertificate:
    gates: tuple[ClassifiedRealGate, ...]
    gate_classification_check_passed: bool
    qca_geometric_gate_algebra_safe: bool
    commutant_basis_matches_expected: bool
    safe_algebra_closure_passed: bool
    unsafe_gate_witnesses: tuple[str, ...]
    load_bearing_qca_bridge: bool


def classify_complex_gate(matrix: sp.Matrix) -> GateClass:
    if matrix.shape != (5, 5):
        return "unknown_fail"

    upper_right = matrix[:3, 3:5]
    lower_left = matrix[3:5, :3]
    if upper_right != sp.zeros(3, 2) or lower_left != sp.zeros(2, 3):
        return "block_mixing_fail"
    if not is_scalar_block(matrix[:3, :3]):
        return "color_breaking_fail"
    if not is_scalar_block(matrix[3:5, 3:5]):
        return "weak_breaking_fail"
    return "safe_sm_commutant"


def classify_real_gate(matrix: sp.Matrix) -> GateClass:
    if matrix.shape != (10, 10):
        return "unknown_fail"

    complex_matrix = complex_from_real(matrix)
    if complex_matrix is None:
        return "antilinear_fail"
    return classify_complex_gate(complex_matrix)


def canonical_gate_examples() -> tuple[NamedRealGate, ...]:
    p3 = complex_projector_3()
    p2 = complex_projector_2()
    block_mixer = matrix_unit(0, 3) + matrix_unit(3, 0)
    color_axis_projector = sp.diag(1, 0, 0, 0, 0)
    weak_axis_projector = sp.diag(0, 0, 0, 1, 0)
    conjugation = identity(5).row_join(sp.zeros(5)).col_join(
        sp.zeros(5).row_join(-identity(5))
    )

    return (
        NamedRealGate("P_3", realify(p3), "safe_sm_commutant"),
        NamedRealGate("P_2", realify(p2), "safe_sm_commutant"),
        NamedRealGate("J_P_3", realify(sp.I * p3), "safe_sm_commutant"),
        NamedRealGate("J_P_2", realify(sp.I * p2), "safe_sm_commutant"),
        NamedRealGate("block_mixer", realify(block_mixer), "block_mixing_fail"),
        NamedRealGate(
            "rank_one_color_projector",
            realify(color_axis_projector),
            "color_breaking_fail",
        ),
        NamedRealGate(
            "rank_one_weak_projector",
            realify(weak_axis_projector),
            "weak_breaking_fail",
        ),
        NamedRealGate("real_conjugation", conjugation, "antilinear_fail"),
    )


def gate_classification_certificate(
    gates: tuple[NamedRealGate, ...] | None = None,
) -> GateClassificationCertificate:
    gates = gates if gates is not None else canonical_gate_examples()
    classified = tuple(
        ClassifiedRealGate(
            name=gate.name,
            classification=classify_real_gate(gate.matrix),
            expected_class=gate.expected_class,
            matches_expected=(
                gate.expected_class is None
                or classify_real_gate(gate.matrix) == gate.expected_class
            ),
        )
        for gate in gates
    )
    unsafe_witnesses = tuple(
        gate.name for gate in classified if gate.classification != "safe_sm_commutant"
    )
    basis_matches = sm_commutant_basis_matches_expected()
    safe_algebra_closure = safe_commutant_closure_examples_pass()
    check_passed = (
        all(gate.matches_expected for gate in classified)
        and basis_matches
        and safe_algebra_closure
    )
    qca_safe = not unsafe_witnesses and check_passed

    return GateClassificationCertificate(
        gates=classified,
        gate_classification_check_passed=check_passed,
        qca_geometric_gate_algebra_safe=qca_safe,
        commutant_basis_matches_expected=basis_matches,
        safe_algebra_closure_passed=safe_algebra_closure,
        unsafe_gate_witnesses=unsafe_witnesses,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: GateClassificationCertificate) -> dict[str, object]:
    return {
        "gates": [
            {
                "name": gate.name,
                "classification": gate.classification,
                "expected_class": gate.expected_class,
                "matches_expected": gate.matches_expected,
            }
            for gate in certificate.gates
        ],
        "gate_classification_check_passed": (
            certificate.gate_classification_check_passed
        ),
        "qca_geometric_gate_algebra_safe": certificate.qca_geometric_gate_algebra_safe,
        "commutant_basis_matches_expected": (
            certificate.commutant_basis_matches_expected
        ),
        "safe_algebra_closure_passed": certificate.safe_algebra_closure_passed,
        "unsafe_gate_witnesses": list(certificate.unsafe_gate_witnesses),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }
