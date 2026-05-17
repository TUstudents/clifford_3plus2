"""Exact primitive families for the first rule-space exploration sprint."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.explore.rule_space import (
    ExplorationPrimitive,
    PrimitiveFamily,
    PrimitiveSet,
    RuleSpace,
)
from clifford_3plus2_d5.obstruction_r10.search.addressability import (
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
)
from clifford_3plus2_d5.obstruction_r10.search.gate_words import rank_one_pair_rotation


def identity_primitive() -> ExplorationPrimitive:
    return ExplorationPrimitive(
        name="identity",
        matrix=identity(10),
        primitive_class="identity",
    )


def global_clock_primitive() -> ExplorationPrimitive:
    carrier = standard_real_carrier()
    return ExplorationPrimitive(
        name="global_clock_tick",
        matrix=carrier.complex_structure,
        primitive_class="global_clock",
        source="candidate_period_four_clock",
    )


def minus_identity_primitive() -> ExplorationPrimitive:
    return ExplorationPrimitive(
        name="minus_identity",
        matrix=-identity(10),
        primitive_class="whole_block",
    )


def block_reflection_primitive() -> ExplorationPrimitive:
    carrier = standard_real_carrier()
    return ExplorationPrimitive(
        name="block_reflection_3_minus_2",
        matrix=carrier.projector_3 - carrier.projector_2,
        primitive_class="whole_block",
    )


def mode_swap_primitive(left: int, right: int) -> ExplorationPrimitive:
    if left == right:
        raise ValueError("mode swap needs distinct indices")
    if not (0 <= left < 5 and 0 <= right < 5):
        raise ValueError("mode swap indices out of range")

    permutation = list(range(5))
    permutation[left], permutation[right] = permutation[right], permutation[left]
    mode_permutation = sp.zeros(5)
    for row, column in enumerate(permutation):
        mode_permutation[row, column] = 1
    matrix = sp.kronecker_product(identity(2), mode_permutation)
    return ExplorationPrimitive(
        name=f"mode_swap_{left + 1}_{right + 1}",
        matrix=matrix,
        primitive_class="mode_permutation",
    )


def rank_one_pair_primitive(index: int) -> ExplorationPrimitive:
    return ExplorationPrimitive(
        name=f"rank_one_pair_rotation_{index + 1}",
        matrix=rank_one_pair_rotation(index),
        primitive_class="rank_one_pair",
        independently_addressable_pair=True,
        source="falsifier",
    )


def default_primitive_families() -> tuple[PrimitiveFamily, ...]:
    return (
        PrimitiveFamily("identity", (identity_primitive(),)),
        PrimitiveFamily("global_clock", (global_clock_primitive(),)),
        PrimitiveFamily("whole_block", (minus_identity_primitive(), block_reflection_primitive())),
        PrimitiveFamily("mode_permutation", (mode_swap_primitive(0, 1), mode_swap_primitive(3, 4))),
        PrimitiveFamily("rank_one_pair_falsifier", (rank_one_pair_primitive(0),)),
    )


def default_e1_rule_space() -> RuleSpace:
    block_controls = standard_block_projectors()
    clock = global_clock_primitive()
    identity_gate = identity_primitive()
    minus_identity = minus_identity_primitive()
    block_reflection = block_reflection_primitive()

    primitive_sets = (
        PrimitiveSet("identity_only", (identity_gate,), block_controls),
        PrimitiveSet("clock_sanity", (clock,), block_controls),
        PrimitiveSet("clock_with_minus_identity", (clock, minus_identity), block_controls),
        PrimitiveSet("clock_with_block_reflection", (clock, block_reflection), block_controls),
        PrimitiveSet("clock_with_color_swap", (clock, mode_swap_primitive(0, 1)), block_controls),
        PrimitiveSet("clock_with_weak_swap", (clock, mode_swap_primitive(3, 4)), block_controls),
        PrimitiveSet(
            "rank_one_pair_falsifier",
            (clock, rank_one_pair_primitive(0)),
            block_controls,
        ),
        PrimitiveSet(
            "rank_one_color_control_falsifier",
            (clock,),
            block_controls + rank_one_color_projector_controls(),
        ),
        PrimitiveSet(
            "rank_one_weak_control_falsifier",
            (clock,),
            block_controls + rank_one_weak_projector_controls(),
        ),
        PrimitiveSet(
            "off_block_control_falsifier",
            (clock,),
            block_controls + (off_block_mixer_control(),),
        ),
    )
    return RuleSpace(
        name="e1_bounded_real_rule_space",
        primitive_families=default_primitive_families(),
        primitive_sets=primitive_sets,
    )
