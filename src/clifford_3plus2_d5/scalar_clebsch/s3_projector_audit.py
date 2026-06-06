"""Finite S3 projector audit for the down-sector count problem.

The goal is to separate three statements that are easy to conflate:

* the regular S3 shell has dimension 6;
* a rank-2 standard copy exists, but selecting one copy is a defect-polarization
  choice rather than a central S3 projector;
* rank 5 exists as the complement of a one-dimensional irrep, but S3 alone does
  not choose whether to exclude the trivial or sign line.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import sympy as sp


S3Element = tuple[int, int, int]


def s3_elements() -> tuple[S3Element, ...]:
    """Return S3 as permutations of ``(0,1,2)``."""

    return tuple(permutations((0, 1, 2)))


def compose(left: S3Element, right: S3Element) -> S3Element:
    """Return composition ``left o right``."""

    return tuple(left[index] for index in right)


def permutation_sign(element: S3Element) -> int:
    """Return the parity sign of an S3 element."""

    inversions = 0
    for first, second in combinations(range(3), 2):
        if element[first] > element[second]:
            inversions += 1
    return -1 if inversions % 2 else 1


def fixed_point_count(element: S3Element) -> int:
    """Return the number of fixed points of a permutation."""

    return sum(1 for index, image in enumerate(element) if index == image)


def standard_character(element: S3Element) -> int:
    """Return the character of the two-dimensional standard irrep of S3."""

    return fixed_point_count(element) - 1


def left_regular_matrix(element: S3Element) -> sp.Matrix:
    """Return the left-regular permutation matrix for an S3 element."""

    elements = s3_elements()
    positions = {group_element: index for index, group_element in enumerate(elements)}
    matrix = sp.zeros(len(elements), len(elements))
    for column, basis_element in enumerate(elements):
        image = compose(element, basis_element)
        matrix[positions[image], column] = 1
    return matrix


def central_projectors() -> dict[str, sp.Matrix]:
    """Return the three central isotypic projectors of the regular S3 module."""

    elements = s3_elements()
    identity = sp.eye(len(elements))
    trivial = sp.zeros(len(elements), len(elements))
    sign = sp.zeros(len(elements), len(elements))
    standard = sp.zeros(len(elements), len(elements))
    for element in elements:
        regular = left_regular_matrix(element)
        trivial += regular
        sign += permutation_sign(element) * regular
        standard += standard_character(element) * regular

    trivial = sp.Rational(1, 6) * trivial
    sign = sp.Rational(1, 6) * sign
    standard = sp.Rational(2, 6) * standard

    # Keep the exact character projector and the complement check aligned.
    assert sp.simplify(standard - (identity - trivial - sign)) == sp.zeros(6, 6)
    return {
        "trivial": trivial,
        "sign": sign,
        "standard_isotypic": standard,
    }


def projector_rank(projector: sp.Matrix) -> int:
    """Return the exact rank of a projector."""

    return int(projector.rank())


def central_projector_ranks() -> dict[str, int]:
    """Return ranks of central S3 projector sums."""

    projectors = central_projectors()
    identity = sp.eye(6)
    return {
        "0": 0,
        "trivial": projector_rank(projectors["trivial"]),
        "sign": projector_rank(projectors["sign"]),
        "standard_isotypic": projector_rank(projectors["standard_isotypic"]),
        "trivial_plus_sign": projector_rank(projectors["trivial"] + projectors["sign"]),
        "trivial_plus_standard": projector_rank(
            projectors["trivial"] + projectors["standard_isotypic"]
        ),
        "sign_plus_standard": projector_rank(
            projectors["sign"] + projectors["standard_isotypic"]
        ),
        "regular": projector_rank(identity),
    }


def orbit_incidence_vectors() -> sp.Matrix:
    """Return incidence columns for the three values of ``g(0)``."""

    elements = s3_elements()
    columns: list[list[int]] = []
    for image in range(3):
        columns.append([1 if element[0] == image else 0 for element in elements])
    return sp.Matrix(columns).T


def standard_copy_projector() -> sp.Matrix:
    """Return a rank-2 standard-copy projector selected by a port polarization."""

    incidence = orbit_incidence_vectors()
    vector_01 = incidence[:, 0] - incidence[:, 1]
    vector_12 = incidence[:, 1] - incidence[:, 2]
    basis = sp.Matrix.hstack(vector_01, vector_12)
    return (basis * (basis.T * basis).inv() * basis.T).applyfunc(sp.simplify)


def standard_copy_projector_rank() -> int:
    """Return the rank of the defect-selected standard-copy projector."""

    return projector_rank(standard_copy_projector())


def rank_five_projectors() -> dict[str, sp.Matrix]:
    """Return the two central rank-5 projector candidates."""

    projectors = central_projectors()
    identity = sp.eye(6)
    return {
        "regular_minus_trivial": identity - projectors["trivial"],
        "regular_minus_sign": identity - projectors["sign"],
    }


def rank_five_projector_ranks() -> dict[str, int]:
    """Return ranks of the two central rank-5 projector candidates."""

    return {
        name: projector_rank(projector)
        for name, projector in rank_five_projectors().items()
    }


def central_s3_does_not_force_rank_two() -> bool:
    """Return true when central S3 does not supply a rank-2 standard copy."""

    ranks = central_projector_ranks()
    return ranks["standard_isotypic"] == 4 and standard_copy_projector_rank() == 2


def rank_five_is_not_unique() -> bool:
    """Return true when S3 supplies two rank-5 complements."""

    ranks = rank_five_projector_ranks()
    return ranks == {"regular_minus_trivial": 5, "regular_minus_sign": 5}


@dataclass(frozen=True)
class S3ProjectorAuditPayload:
    """Audit payload for S3 projector count availability."""

    final_verdict: str
    central_ranks: dict[str, int]
    standard_copy_rank: int
    rank_five_ranks: dict[str, int]
    central_standard_rank_two_absent: bool
    rank_two_requires_defect_polarization: bool
    rank_five_not_unique: bool
    count_vector_available: bool
    count_vector_forced_by_s3_alone: bool
    interpretation: str


def s3_projector_audit_payload() -> S3ProjectorAuditPayload:
    """Return the S3 projector audit verdict."""

    central_ranks = central_projector_ranks()
    standard_rank = standard_copy_projector_rank()
    rank_five_ranks = rank_five_projector_ranks()
    standard_rank_two_absent = central_s3_does_not_force_rank_two()
    rank_five_ambiguous = rank_five_is_not_unique()
    count_vector_available = (
        central_ranks["regular"] == 6
        and standard_rank == 2
        and set(rank_five_ranks.values()) == {5}
    )
    count_vector_forced = False
    checks_pass = (
        central_ranks["trivial"] == 1
        and central_ranks["sign"] == 1
        and central_ranks["standard_isotypic"] == 4
        and central_ranks["regular"] == 6
        and standard_rank == 2
        and standard_rank_two_absent
        and rank_five_ambiguous
        and count_vector_available
    )

    if checks_pass:
        final_verdict = "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
        interpretation = (
            "The S3 regular shell can host the desired count vector (6,2,5): "
            "6 is the regular shell, 2 is a defect-polarized standard copy, "
            "and 5 is the complement of a one-dimensional irrep. However, S3 "
            "alone does not force the vector: the central rank-2 projector is "
            "trivial plus sign rather than a standard doublet, and rank 5 has "
            "two central complements. The missing theorem is the defect "
            "selection rule that chooses the standard copy and the excluded "
            "one-dimensional line."
        )
    else:
        final_verdict = "S3_PROJECTOR_COUNT_AVAILABILITY_KILL"
        interpretation = (
            "The regular S3 projector audit failed to make the counts 6, 2, "
            "and 5 available with the expected central/noncentral status."
        )

    return S3ProjectorAuditPayload(
        final_verdict=final_verdict,
        central_ranks=central_ranks,
        standard_copy_rank=standard_rank,
        rank_five_ranks=rank_five_ranks,
        central_standard_rank_two_absent=standard_rank_two_absent,
        rank_two_requires_defect_polarization=standard_rank_two_absent and standard_rank == 2,
        rank_five_not_unique=rank_five_ambiguous,
        count_vector_available=count_vector_available,
        count_vector_forced_by_s3_alone=count_vector_forced,
        interpretation=interpretation,
    )
