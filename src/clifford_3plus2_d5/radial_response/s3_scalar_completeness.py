"""R8 S3-shell completeness gate for scalar repair successors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    S3Element,
    compose,
    fixed_point_count,
    permutation_sign,
    s3_elements,
)


IDENTITY: S3Element = (0, 1, 2)
TRIALITY_PLUS: S3Element = (1, 2, 0)
TRIALITY_MINUS: S3Element = (2, 0, 1)


class S3ScalarSector(StrEnum):
    """S3 repair-sector classification."""

    IDENTITY = "IDENTITY"
    SCALAR_HOLOMORPHIC = "SCALAR_HOLOMORPHIC"
    HERMITIAN_Z2 = "HERMITIAN_Z2"


@dataclass(frozen=True)
class S3ScalarElementRow:
    """One row in the R8 S3 scalar-shell census."""

    element: S3Element
    label: str
    sector: S3ScalarSector
    scalar_allowed: bool
    reason: str


@dataclass(frozen=True)
class ScalarS3ShellCompletenessPayload:
    """Payload for the R8 S3 scalar-shell completeness gate."""

    final_verdict: str
    element_count: int
    identity_rejected: bool
    allowed_successors: tuple[str, str]
    allowed_elements: tuple[S3Element, S3Element]
    transposition_count: int
    transpositions_rejected: bool
    transposition_conjugation_swaps_triality: bool
    r7_labels_match: bool
    full_qca_boundary_completeness_derived: bool
    census: tuple[S3ScalarElementRow, ...]
    interpretation: str


def inverse_element(element: S3Element) -> S3Element:
    """Return the inverse permutation."""

    inverse = [0, 0, 0]
    for index, image in enumerate(element):
        inverse[image] = index
    return tuple(inverse)


def conjugate_element(conjugator: S3Element, element: S3Element) -> S3Element:
    """Return ``conjugator * element * conjugator^-1``."""

    return compose(compose(conjugator, element), inverse_element(conjugator))


def is_identity(element: S3Element) -> bool:
    """Return whether an S3 element is the identity."""

    return element == IDENTITY


def is_three_cycle(element: S3Element) -> bool:
    """Return whether an S3 element is a non-identity cyclic/triality element."""

    return element != IDENTITY and permutation_sign(element) == 1 and fixed_point_count(element) == 0


def is_transposition(element: S3Element) -> bool:
    """Return whether an S3 element is a transposition."""

    return permutation_sign(element) == -1 and fixed_point_count(element) == 1


def scalar_successor_label(element: S3Element) -> str:
    """Return the R7 successor label for an allowed cyclic element."""

    if element == TRIALITY_PLUS:
        return "triality_plus"
    if element == TRIALITY_MINUS:
        return "triality_minus"
    raise ValueError("element is not one of the scalar triality successors")


def classify_s3_scalar_element(element: S3Element) -> S3ScalarElementRow:
    """Classify one S3 element for scalar holomorphic one-tick repair."""

    if is_three_cycle(element):
        return S3ScalarElementRow(
            element=element,
            label=scalar_successor_label(element),
            sector=S3ScalarSector.SCALAR_HOLOMORPHIC,
            scalar_allowed=True,
            reason="non-identity A3 triality element",
        )
    if is_identity(element):
        return S3ScalarElementRow(
            element=element,
            label="identity",
            sector=S3ScalarSector.IDENTITY,
            scalar_allowed=False,
            reason="identity is same-state, not repair",
        )
    if is_transposition(element):
        return S3ScalarElementRow(
            element=element,
            label=f"transposition_{element}",
            sector=S3ScalarSector.HERMITIAN_Z2,
            scalar_allowed=False,
            reason="transposition belongs to Hermitian/Z2 repair sector",
        )
    raise ValueError(f"unclassified S3 element: {element}")


def s3_scalar_census() -> tuple[S3ScalarElementRow, ...]:
    """Return the complete S3 scalar-shell repair census."""

    return tuple(classify_s3_scalar_element(element) for element in s3_elements())


def allowed_scalar_s3_successor_elements() -> tuple[S3Element, S3Element]:
    """Return the two allowed scalar repair elements."""

    allowed = tuple(row.element for row in s3_scalar_census() if row.scalar_allowed)
    if allowed != (TRIALITY_PLUS, TRIALITY_MINUS):
        raise AssertionError(f"unexpected scalar successor order: {allowed}")
    return allowed


def allowed_scalar_s3_successor_labels() -> tuple[str, str]:
    """Return labels matching the R7 two-successor certificate."""

    return tuple(scalar_successor_label(element) for element in allowed_scalar_s3_successor_elements())


def transposition_elements() -> tuple[S3Element, ...]:
    """Return the three S3 transpositions."""

    return tuple(element for element in s3_elements() if is_transposition(element))


def transposition_conjugation_swaps_triality() -> bool:
    """Return whether every transposition swaps the two triality elements."""

    return all(
        conjugate_element(transposition, TRIALITY_PLUS) == TRIALITY_MINUS
        and conjugate_element(transposition, TRIALITY_MINUS) == TRIALITY_PLUS
        for transposition in transposition_elements()
    )


def scalar_s3_shell_completeness_pass() -> bool:
    """Return whether the S3 shell leaves exactly the R7 scalar successor pair."""

    census = s3_scalar_census()
    return (
        len(census) == 6
        and len([row for row in census if row.sector == S3ScalarSector.IDENTITY]) == 1
        and allowed_scalar_s3_successor_labels() == ("triality_plus", "triality_minus")
        and len(transposition_elements()) == 3
        and all(
            not row.scalar_allowed
            for row in census
            if row.sector == S3ScalarSector.HERMITIAN_Z2
        )
        and transposition_conjugation_swaps_triality()
    )


def scalar_s3_shell_completeness_payload() -> ScalarS3ShellCompletenessPayload:
    """Return the R8 S3 scalar-shell completeness verdict."""

    census = s3_scalar_census()
    identity_rejected = any(
        row.sector == S3ScalarSector.IDENTITY and not row.scalar_allowed
        for row in census
    )
    transpositions_rejected = all(
        not row.scalar_allowed
        for row in census
        if row.sector == S3ScalarSector.HERMITIAN_Z2
    )
    conjugation_swaps = transposition_conjugation_swaps_triality()
    labels = allowed_scalar_s3_successor_labels()
    checks_pass = (
        scalar_s3_shell_completeness_pass()
        and identity_rejected
        and transpositions_rejected
        and labels == ("triality_plus", "triality_minus")
    )

    if checks_pass:
        final_verdict = "SCALAR_S3_SHELL_COMPLETENESS_PASS"
        interpretation = (
            "Inside the S3 regular shell, the scalar holomorphic one-tick "
            "repair sector contains exactly the two non-identity A3 triality "
            "elements. The identity is same-state, and the three transpositions "
            "belong to the Hermitian/Z2 repair sector. Thus R7's two-successor "
            "basis is complete within S3, but this does not prove that the "
            "full QCA boundary Hilbert space has no extra scalar outputs."
        )
    else:
        final_verdict = "SCALAR_S3_SHELL_COMPLETENESS_KILL"
        interpretation = "The S3 shell does not reduce to exactly the R7 scalar successor pair."

    return ScalarS3ShellCompletenessPayload(
        final_verdict=final_verdict,
        element_count=len(census),
        identity_rejected=identity_rejected,
        allowed_successors=labels,
        allowed_elements=allowed_scalar_s3_successor_elements(),
        transposition_count=len(transposition_elements()),
        transpositions_rejected=transpositions_rejected,
        transposition_conjugation_swaps_triality=conjugation_swaps,
        r7_labels_match=labels == ("triality_plus", "triality_minus"),
        full_qca_boundary_completeness_derived=False,
        census=census,
        interpretation=interpretation,
    )
