"""Profile factories for the complex-linear split lab."""

from __future__ import annotations

from clifford_3plus2_d5.lepton.complex_carrier import (
    complex_c3_lepton_family_projectors,
    complex_c5_3plus2_projectors,
)
from clifford_3plus2_d5.lepton.complex_predicates import (
    c3_split_idempotent_policy,
    c5_split_idempotent_policy,
)
from clifford_3plus2_d5.lepton.complex_verdict import ComplexSplitProfile


def complex_c3_split_profile() -> ComplexSplitProfile:
    return ComplexSplitProfile(
        name="complex_c3_lepton_family_split",
        dimension=3,
        target_projectors=complex_c3_lepton_family_projectors(),
        max_algebra_dimension=16,
        max_center_dimension=4,
        expected_block_dimensions=(1, 4),
        expected_block_commutativity=(True, False),
        idempotent_policy=c3_split_idempotent_policy,
    )


def complex_c5_split_profile() -> ComplexSplitProfile:
    return ComplexSplitProfile(
        name="complex_c5_3plus2_split",
        dimension=5,
        target_projectors=complex_c5_3plus2_projectors(),
        max_algebra_dimension=32,
        max_center_dimension=6,
        expected_block_dimensions=(9, 4),
        expected_block_commutativity=(False, False),
        idempotent_policy=c5_split_idempotent_policy,
    )


def complex_c5_discovered_split_profile() -> ComplexSplitProfile:
    return ComplexSplitProfile(
        name="complex_c5_discovered_3plus2_split",
        dimension=5,
        target_projectors=complex_c5_3plus2_projectors(),
        max_algebra_dimension=32,
        max_center_dimension=6,
        expected_block_dimensions=(9, 4),
        expected_block_commutativity=(False, False),
        target_ranks=(3, 2),
        allow_discovered_split=True,
    )
