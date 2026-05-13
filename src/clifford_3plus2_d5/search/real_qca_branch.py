"""Real-QCA-first branch checks for Phase 8A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.search.addressability import (
    AddressableOperator,
    certificate_to_dict as structural_split_certificate_to_dict,
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
    structural_split_certificate,
)
from clifford_3plus2_d5.search.forced_j import (
    certificate_to_dict as forced_j_certificate_to_dict,
)
from clifford_3plus2_d5.search.forced_j import j_certificate
from clifford_3plus2_d5.search.gate_words import (
    PrimitiveGate,
    rank_one_pair_rotation,
)
from clifford_3plus2_d5.search.normalizer import (
    certificate_to_dict as normalizer_certificate_to_dict,
)
from clifford_3plus2_d5.search.normalizer import normalizer_certificate


RealQCAPrimitiveClass = Literal[
    "global_clock_rotation",
    "whole_block_rotation",
    "structural_graph_local",
    "translation_invariant_layer",
    "rank_one_pair_rotation",
]
RealQCABranchVerdict = Literal["forced_candidate", "candidate_only", "falsified"]


@dataclass(frozen=True)
class RealQCAPrimitive:
    name: str
    matrix: sp.Matrix
    primitive_class: RealQCAPrimitiveClass
    finite_depth: bool = True
    translation_invariant: bool = True
    word_eligible: bool = True
    independently_addressable_pair: bool = False
    source: str = "candidate"


@dataclass(frozen=True)
class RealQCABranchCertificate:
    branch_name: str
    max_depth: int
    primitive_count: int
    addressable_operator_count: int
    candidate_word_found: bool
    candidate_word: tuple[str, ...]
    word_depth: int
    finite_depth: bool
    translation_invariant: bool
    generates_j: bool
    generates_split: bool
    j_forced_by_rule_space: bool
    split_forced_by_rule_space: bool
    forbidden_rank_one_controls_present: bool
    forbidden_off_block_controls_present: bool
    addressability_algebra_safe: bool
    normalizer_verdict: str
    forced_j_certificate: dict[str, object]
    structural_split_certificate: dict[str, object]
    normalizer_certificate: dict[str, object]
    real_qca_branch_verdict: RealQCABranchVerdict
    load_bearing_qca_bridge: bool


def global_clock_rotation_primitive() -> RealQCAPrimitive:
    carrier = standard_real_carrier()
    return RealQCAPrimitive(
        name="global_clock_tick",
        matrix=carrier.complex_structure,
        primitive_class="global_clock_rotation",
        source="candidate_period_four_clock",
    )


def standard_real_qca_primitives() -> tuple[RealQCAPrimitive, ...]:
    return (global_clock_rotation_primitive(),)


def rank_one_pair_rotation_primitives() -> tuple[RealQCAPrimitive, ...]:
    return tuple(
        RealQCAPrimitive(
            name=f"rank_one_pair_rotation_{index + 1}",
            matrix=rank_one_pair_rotation(index),
            primitive_class="rank_one_pair_rotation",
            independently_addressable_pair=True,
            source="falsifier",
        )
        for index in range(5)
    )


def primitive_gate_projection(
    primitives: tuple[RealQCAPrimitive, ...],
) -> tuple[PrimitiveGate, ...]:
    return tuple(
        PrimitiveGate(
            name=primitive.name,
            matrix=primitive.matrix,
            independently_addressable=primitive.independently_addressable_pair,
        )
        for primitive in primitives
        if primitive.word_eligible
    )


def _primitive_by_name(
    primitives: tuple[RealQCAPrimitive, ...],
) -> dict[str, RealQCAPrimitive]:
    return {primitive.name: primitive for primitive in primitives}


def _word_primitives(
    primitives: tuple[RealQCAPrimitive, ...],
    word: tuple[str, ...],
) -> tuple[RealQCAPrimitive, ...]:
    by_name = _primitive_by_name(primitives)
    return tuple(by_name[name] for name in word if name in by_name)


def real_qca_branch_certificate(
    *,
    primitives: tuple[RealQCAPrimitive, ...] | None = None,
    addressable_operators: tuple[AddressableOperator, ...] | None = None,
    max_depth: int = 4,
    rule_data_source: str = "candidate_only",
    qca_rule_forces_j: bool = False,
    qca_rule_forces_split: bool = False,
) -> RealQCABranchCertificate:
    primitives = primitives if primitives is not None else standard_real_qca_primitives()
    addressable_operators = (
        addressable_operators
        if addressable_operators is not None
        else standard_block_projectors()
    )

    forced_j = j_certificate(
        primitives=primitive_gate_projection(primitives),
        max_depth=max_depth,
        qca_rule_forces_word=qca_rule_forces_j,
    )
    split = structural_split_certificate(
        operators=addressable_operators,
        qca_rule_forces_split=qca_rule_forces_split,
        qca_structural_origin=rule_data_source,
    )
    normalizer = normalizer_certificate(
        rule_data_operators=addressable_operators,
        addressable_operators=addressable_operators,
        rule_data_source=rule_data_source,
        qca_rule_forces_j=qca_rule_forces_j,
        qca_rule_forces_split=qca_rule_forces_split,
    )

    word_primitives = _word_primitives(primitives, forced_j.gate_word)
    word_depth = len(forced_j.gate_word)
    finite_depth = forced_j.generated_by_gate_word and all(
        primitive.finite_depth for primitive in word_primitives
    )
    translation_invariant = forced_j.generated_by_gate_word and all(
        primitive.translation_invariant for primitive in word_primitives
    )
    generates_j = (
        forced_j.generated_by_gate_word
        and forced_j.is_real_orthogonal
        and forced_j.determinant == 1
        and forced_j.squares_to_minus_identity
        and forced_j.equals_standard_j
    )
    generates_split = split.projector_identities_passed and split.projectors_commute_with_j
    forbidden_rank_one = (
        forced_j.rank_one_pair_rotations_addressable
        or split.rank_one_color_projectors_addressable
        or split.rank_one_weak_projectors_addressable
    )
    forbidden_off_block = split.off_block_controls_addressable
    j_forced = forced_j.qca_forces_j and normalizer.j_unique_or_forced
    split_forced = (
        split.qca_supplies_structural_3plus2_split
        and normalizer.split_unique_or_forced
    )

    if (
        forced_j.verdict == "falsified"
        or split.structural_split_verdict == "falsified"
        or normalizer.forcedness_verdict == "falsified"
        or forbidden_rank_one
        or forbidden_off_block
    ):
        verdict: RealQCABranchVerdict = "falsified"
    elif j_forced and split_forced:
        verdict = "forced_candidate"
    else:
        verdict = "candidate_only"

    return RealQCABranchCertificate(
        branch_name="phase_8a_stronger_real_qca_first",
        max_depth=max_depth,
        primitive_count=len(primitives),
        addressable_operator_count=len(addressable_operators),
        candidate_word_found=forced_j.generated_by_gate_word,
        candidate_word=forced_j.gate_word,
        word_depth=word_depth,
        finite_depth=finite_depth,
        translation_invariant=translation_invariant,
        generates_j=generates_j,
        generates_split=generates_split,
        j_forced_by_rule_space=j_forced,
        split_forced_by_rule_space=split_forced,
        forbidden_rank_one_controls_present=forbidden_rank_one,
        forbidden_off_block_controls_present=forbidden_off_block,
        addressability_algebra_safe=normalizer.addressability_algebra_safe,
        normalizer_verdict=normalizer.forcedness_verdict,
        forced_j_certificate=forced_j_certificate_to_dict(forced_j),
        structural_split_certificate=structural_split_certificate_to_dict(split),
        normalizer_certificate=normalizer_certificate_to_dict(normalizer),
        real_qca_branch_verdict=verdict,
        load_bearing_qca_bridge=False,
    )


def certificate_to_dict(certificate: RealQCABranchCertificate) -> dict[str, object]:
    return {
        "branch_name": certificate.branch_name,
        "max_depth": certificate.max_depth,
        "primitive_count": certificate.primitive_count,
        "addressable_operator_count": certificate.addressable_operator_count,
        "candidate_word_found": certificate.candidate_word_found,
        "candidate_word": list(certificate.candidate_word),
        "word_depth": certificate.word_depth,
        "finite_depth": certificate.finite_depth,
        "translation_invariant": certificate.translation_invariant,
        "generates_j": certificate.generates_j,
        "generates_split": certificate.generates_split,
        "j_forced_by_rule_space": certificate.j_forced_by_rule_space,
        "split_forced_by_rule_space": certificate.split_forced_by_rule_space,
        "forbidden_rank_one_controls_present": (
            certificate.forbidden_rank_one_controls_present
        ),
        "forbidden_off_block_controls_present": (
            certificate.forbidden_off_block_controls_present
        ),
        "addressability_algebra_safe": certificate.addressability_algebra_safe,
        "normalizer_verdict": certificate.normalizer_verdict,
        "forced_j_certificate": certificate.forced_j_certificate,
        "structural_split_certificate": certificate.structural_split_certificate,
        "normalizer_certificate": certificate.normalizer_certificate,
        "real_qca_branch_verdict": certificate.real_qca_branch_verdict,
        "real_qca_branch_check_passed": (
            certificate.candidate_word_found
            and certificate.finite_depth
            and certificate.translation_invariant
            and certificate.generates_j
            and certificate.generates_split
            and not certificate.forbidden_rank_one_controls_present
            and not certificate.forbidden_off_block_controls_present
            and certificate.addressability_algebra_safe
        ),
        "load_bearing_qca_bridge": certificate.load_bearing_qca_bridge,
    }


def controls_with_rank_one_color() -> tuple[AddressableOperator, ...]:
    return standard_block_projectors() + rank_one_color_projector_controls()


def controls_with_rank_one_weak() -> tuple[AddressableOperator, ...]:
    return standard_block_projectors() + rank_one_weak_projector_controls()


def controls_with_off_block() -> tuple[AddressableOperator, ...]:
    return standard_block_projectors() + (off_block_mixer_control(),)
