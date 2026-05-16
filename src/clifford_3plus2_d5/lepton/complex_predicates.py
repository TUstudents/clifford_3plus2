"""Predicate helpers for complex-linear split verdicts."""

from __future__ import annotations

from collections.abc import Sequence

from clifford_3plus2_d5.lepton.complex_verdict import (
    ComplexCentralIdempotent,
    ComplexSplitProfile,
    split_idempotent_policy,
)


def c3_split_idempotent_policy(
    profile: ComplexSplitProfile,
    idempotents: Sequence[ComplexCentralIdempotent],
) -> str:
    """Allow the rank-1 singlet target, but reject rank-1 refinements."""

    return split_idempotent_policy(profile, idempotents)


def c5_split_idempotent_policy(
    profile: ComplexSplitProfile,
    idempotents: Sequence[ComplexCentralIdempotent],
) -> str:
    return split_idempotent_policy(profile, idempotents)
