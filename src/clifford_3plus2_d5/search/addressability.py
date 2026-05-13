"""Addressability checks for the structural 3+2 split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity, is_zero_matrix, zero
from clifford_3plus2_d5.algebra.projectors import (
    color_axis_projectors,
    projector_pair_check_passed,
    projector_pair_identities,
    projectors_commute_with_j,
    weak_axis_projectors,
)
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier


AddressabilityClass = Literal[
    "safe_block_projector",
    "rank_one_color_projector",
    "rank_one_weak_projector",
    "off_block_mixer",
    "non_scalar_block_control",
    "dimension_mismatch",
]
StructuralSplitVerdict = Literal["structural_split", "candidate_only", "falsified"]


@dataclass(frozen=True)
class AddressableOperator:
    name: str
    matrix: sp.Matrix


@dataclass(frozen=True)
class ClassifiedAddressableOperator:
    name: str
    classification: AddressabilityClass


@dataclass(frozen=True)
class StructuralSplitCertificate:
    projector_identities_passed: bool
    projectors_commute_with_j: bool
    projector_3_rank: int
    projector_2_rank: int
    qca_structural_origin: str
    qca_supplies_structural_3plus2_split: bool
    rank_one_color_projectors_addressable: bool
    rank_one_weak_projectors_addressable: bool
    off_block_controls_addressable: bool
    non_scalar_block_controls_addressable: bool
    addressability_algebra_safe: bool
    addressable_operators: tuple[ClassifiedAddressableOperator, ...]
    structural_split_verdict: StructuralSplitVerdict
    load_bearing_qca_bridge: bool


def standard_block_projectors() -> tuple[AddressableOperator, ...]:
    carrier = standard_real_carrier()
    return (
        AddressableOperator("P_3", carrier.projector_3),
        AddressableOperator("P_2", carrier.projector_2),
    )


def rank_one_color_projector_controls() -> tuple[AddressableOperator, ...]:
    return tuple(
        AddressableOperator(f"color_axis_projector_{index + 1}", projector)
        for index, projector in enumerate(color_axis_projectors())
    )


def rank_one_weak_projector_controls() -> tuple[AddressableOperator, ...]:
    return tuple(
        AddressableOperator(f"weak_axis_projector_{index + 1}", projector)
        for index, projector in enumerate(weak_axis_projectors())
    )


def off_block_mixer_control() -> AddressableOperator:
    mixer = zero(10)
    mixer[0, 3] = 1
    mixer[3, 0] = 1
    mixer[5, 8] = 1
    mixer[8, 5] = 1
    return AddressableOperator("off_block_mixer", mixer)


def classify_addressable_operator(matrix: sp.Matrix) -> AddressabilityClass:
    carrier = standard_real_carrier()
    p3 = carrier.projector_3
    p2 = carrier.projector_2
    safe_projectors = (zero(10), identity(10), p3, p2)

    if matrix.shape != (carrier.dimension, carrier.dimension):
        return "dimension_mismatch"
    if any(matrix == projector for projector in safe_projectors):
        return "safe_block_projector"
    if any(matrix == projector for projector in color_axis_projectors()):
        return "rank_one_color_projector"
    if any(matrix == projector for projector in weak_axis_projectors()):
        return "rank_one_weak_projector"
    if not (is_zero_matrix(p3 * matrix * p2) and is_zero_matrix(p2 * matrix * p3)):
        return "off_block_mixer"
    return "non_scalar_block_control"


def classify_addressable_operators(
    operators: tuple[AddressableOperator, ...],
) -> tuple[ClassifiedAddressableOperator, ...]:
    return tuple(
        ClassifiedAddressableOperator(
            name=operator.name,
            classification=classify_addressable_operator(operator.matrix),
        )
        for operator in operators
    )


def structural_split_certificate(
    *,
    operators: tuple[AddressableOperator, ...] | None = None,
    qca_rule_forces_split: bool = False,
    qca_structural_origin: str = "candidate_only",
) -> StructuralSplitCertificate:
    identities = projector_pair_identities()
    projector_identities_passed = projector_pair_check_passed(identities)
    commute_with_j = projectors_commute_with_j(identities)
    operators = operators if operators is not None else standard_block_projectors()
    classified = classify_addressable_operators(operators)
    classifications = tuple(operator.classification for operator in classified)

    rank_one_color = "rank_one_color_projector" in classifications
    rank_one_weak = "rank_one_weak_projector" in classifications
    off_block = "off_block_mixer" in classifications
    non_scalar_block = "non_scalar_block_control" in classifications
    dimension_mismatch = "dimension_mismatch" in classifications
    addressability_safe = not any(
        classification != "safe_block_projector" for classification in classifications
    )
    qca_origin_is_structural = qca_structural_origin != "candidate_only"
    qca_supplies_split = (
        qca_rule_forces_split
        and qca_origin_is_structural
        and projector_identities_passed
        and commute_with_j
        and addressability_safe
    )

    if not addressability_safe:
        verdict: StructuralSplitVerdict = "falsified"
    elif qca_supplies_split:
        verdict = "structural_split"
    else:
        verdict = "candidate_only"

    return StructuralSplitCertificate(
        projector_identities_passed=projector_identities_passed,
        projectors_commute_with_j=commute_with_j,
        projector_3_rank=identities.projector_3_rank,
        projector_2_rank=identities.projector_2_rank,
        qca_structural_origin=qca_structural_origin,
        qca_supplies_structural_3plus2_split=qca_supplies_split,
        rank_one_color_projectors_addressable=rank_one_color,
        rank_one_weak_projectors_addressable=rank_one_weak,
        off_block_controls_addressable=off_block,
        non_scalar_block_controls_addressable=non_scalar_block or dimension_mismatch,
        addressability_algebra_safe=addressability_safe,
        addressable_operators=classified,
        structural_split_verdict=verdict,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: StructuralSplitCertificate) -> dict[str, object]:
    return {
        "projector_identities_passed": certificate.projector_identities_passed,
        "projectors_commute_with_j": certificate.projectors_commute_with_j,
        "projector_3_rank": certificate.projector_3_rank,
        "projector_2_rank": certificate.projector_2_rank,
        "qca_structural_origin": certificate.qca_structural_origin,
        "qca_supplies_structural_3plus2_split": (
            certificate.qca_supplies_structural_3plus2_split
        ),
        "rank_one_color_projectors_addressable": (
            certificate.rank_one_color_projectors_addressable
        ),
        "rank_one_weak_projectors_addressable": (
            certificate.rank_one_weak_projectors_addressable
        ),
        "off_block_controls_addressable": certificate.off_block_controls_addressable,
        "non_scalar_block_controls_addressable": (
            certificate.non_scalar_block_controls_addressable
        ),
        "addressability_algebra_safe": certificate.addressability_algebra_safe,
        "addressable_operators": [
            {
                "name": operator.name,
                "classification": operator.classification,
            }
            for operator in certificate.addressable_operators
        ],
        "structural_split_verdict": certificate.structural_split_verdict,
        "structural_split_check_passed": (
            certificate.projector_identities_passed
            and certificate.projectors_commute_with_j
            and certificate.addressability_algebra_safe
        ),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }
