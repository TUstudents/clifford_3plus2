"""R9 QCA-to-S3 scalar boundary reduction gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations

from clifford_3plus2_d5.boundary_response.vacuum_framing import (
    induced_residual_permutation,
    selected_exit_stabilizer_permutations,
)
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    S3ScalarSector,
    allowed_scalar_s3_successor_elements,
    allowed_scalar_s3_successor_labels,
    classify_s3_scalar_element,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import S3Element, s3_elements


SELECTED_EXIT = 0


class QCABoundaryReductionVeto(StrEnum):
    """Exact veto labels for QCA scalar-boundary reduction candidates."""

    TETRAHEDRAL_AUTOMORPHISM = "TETRAHEDRAL_AUTOMORPHISM"
    VACUUM_FRAME_PRESERVING = "VACUUM_FRAME_PRESERVING"
    SAME_STATE = "SAME_STATE"
    HERMITIAN_Z2_SECTOR = "HERMITIAN_Z2_SECTOR"


class QCABoundaryReductionVerdict(StrEnum):
    """Verdict for one tetrahedral scalar-boundary candidate."""

    ALLOW = "ALLOW"
    FORBID = "FORBID"


class QCABoundaryFailureClass(StrEnum):
    """Interpretation of a rejected QCA scalar-boundary candidate."""

    ALLOW = "ALLOW"
    NON_AUTOMORPHISM = "NON_AUTOMORPHISM"
    VACUUM_FRAME_BREAKING = "VACUUM_FRAME_BREAKING"
    SAME_STATE = "SAME_STATE"
    HERMITIAN_Z2 = "HERMITIAN_Z2"


@dataclass(frozen=True)
class QCABoundaryReductionRow:
    """One row in the R9 QCA-to-S3 reduction census."""

    candidate: tuple[int, ...]
    verdict: QCABoundaryReductionVerdict
    vetoes: tuple[QCABoundaryReductionVeto, ...]
    failure_class: QCABoundaryFailureClass
    residual_action: S3Element | None
    scalar_label: str | None


@dataclass(frozen=True)
class QCAS3ReductionPayload:
    """Payload for the R9 QCA-to-S3 scalar reduction gate."""

    final_verdict: str
    tetrahedral_automorphism_count: int
    selected_stabilizer_count: int
    selected_moving_rejected_count: int
    induced_s3_count: int
    induced_s3_is_full: bool
    identity_control_rejected: bool
    transposition_controls_rejected: bool
    non_automorphism_controls_rejected: bool
    allowed_successors: tuple[str, str]
    allowed_elements: tuple[S3Element, S3Element]
    r8_successors_match: bool
    microscopic_automorphism_premise_derived: bool
    candidate_rows: tuple[QCABoundaryReductionRow, ...]
    non_automorphism_control_rows: tuple[QCABoundaryReductionRow, ...]
    interpretation: str


def all_tetrahedral_exit_automorphisms() -> tuple[tuple[int, ...], ...]:
    """Return all S4 permutations of the four tetrahedral exits."""

    return tuple(tuple(perm) for perm in permutations(range(4)))


def non_automorphism_controls() -> tuple[tuple[int, ...], ...]:
    """Return finite controls that are not tetrahedral exit automorphisms."""

    return (
        (0, 0, 1, 2),
        (0, 1, 2),
        (0, 1, 2, 4),
    )


def is_tetrahedral_automorphism(candidate: tuple[int, ...]) -> bool:
    """Return whether ``candidate`` is a permutation of the four exits."""

    return len(candidate) == 4 and tuple(sorted(candidate)) == (0, 1, 2, 3)


def preserves_selected_exit(candidate: tuple[int, ...], selected: int = SELECTED_EXIT) -> bool:
    """Return whether ``candidate`` preserves the selected vacuum exit."""

    return is_tetrahedral_automorphism(candidate) and candidate[selected] == selected


def residual_action_for_candidate(
    candidate: tuple[int, ...],
    selected: int = SELECTED_EXIT,
) -> S3Element | None:
    """Return the residual S3 action induced by a selected-preserving candidate."""

    if not preserves_selected_exit(candidate, selected):
        return None
    return induced_residual_permutation(candidate, selected)  # type: ignore[return-value]


def qca_boundary_reduction_row(candidate: tuple[int, ...]) -> QCABoundaryReductionRow:
    """Return one R9 reduction row."""

    if not is_tetrahedral_automorphism(candidate):
        return QCABoundaryReductionRow(
            candidate=candidate,
            verdict=QCABoundaryReductionVerdict.FORBID,
            vetoes=(QCABoundaryReductionVeto.TETRAHEDRAL_AUTOMORPHISM,),
            failure_class=QCABoundaryFailureClass.NON_AUTOMORPHISM,
            residual_action=None,
            scalar_label=None,
        )
    if not preserves_selected_exit(candidate):
        return QCABoundaryReductionRow(
            candidate=candidate,
            verdict=QCABoundaryReductionVerdict.FORBID,
            vetoes=(QCABoundaryReductionVeto.VACUUM_FRAME_PRESERVING,),
            failure_class=QCABoundaryFailureClass.VACUUM_FRAME_BREAKING,
            residual_action=None,
            scalar_label=None,
        )

    residual_action = residual_action_for_candidate(candidate)
    if residual_action is None:
        raise AssertionError("selected-preserving candidate must induce residual action")
    scalar_row = classify_s3_scalar_element(residual_action)
    if scalar_row.scalar_allowed:
        return QCABoundaryReductionRow(
            candidate=candidate,
            verdict=QCABoundaryReductionVerdict.ALLOW,
            vetoes=(),
            failure_class=QCABoundaryFailureClass.ALLOW,
            residual_action=residual_action,
            scalar_label=scalar_row.label,
        )

    if scalar_row.sector == S3ScalarSector.IDENTITY:
        vetoes = (QCABoundaryReductionVeto.SAME_STATE,)
        failure = QCABoundaryFailureClass.SAME_STATE
    else:
        vetoes = (QCABoundaryReductionVeto.HERMITIAN_Z2_SECTOR,)
        failure = QCABoundaryFailureClass.HERMITIAN_Z2
    return QCABoundaryReductionRow(
        candidate=candidate,
        verdict=QCABoundaryReductionVerdict.FORBID,
        vetoes=vetoes,
        failure_class=failure,
        residual_action=residual_action,
        scalar_label=None,
    )


def qca_boundary_reduction_rows() -> tuple[QCABoundaryReductionRow, ...]:
    """Return R9 rows over all tetrahedral automorphism candidates."""

    return tuple(qca_boundary_reduction_row(candidate) for candidate in all_tetrahedral_exit_automorphisms())


def qca_non_automorphism_control_rows() -> tuple[QCABoundaryReductionRow, ...]:
    """Return R9 non-automorphism control rows."""

    return tuple(qca_boundary_reduction_row(candidate) for candidate in non_automorphism_controls())


def selected_exit_stabilizer_count() -> int:
    """Return the selected-exit stabilizer size."""

    return len(selected_exit_stabilizer_permutations(SELECTED_EXIT))


def induced_residual_s3_image() -> tuple[S3Element, ...]:
    """Return the residual actions induced by selected-exit-preserving automorphisms."""

    actions = {
        residual_action_for_candidate(candidate)
        for candidate in all_tetrahedral_exit_automorphisms()
        if preserves_selected_exit(candidate)
    }
    if None in actions:
        raise AssertionError("selected-preserving image unexpectedly contains None")
    return tuple(sorted(action for action in actions if action is not None))


def induced_residual_image_is_full_s3() -> bool:
    """Return whether selected-preserving automorphisms induce all residual S3."""

    return set(induced_residual_s3_image()) == set(s3_elements())


def qca_scalar_boundary_allowed_elements() -> tuple[S3Element, S3Element]:
    """Return allowed scalar holomorphic residual actions from BCC framing."""

    allowed = tuple(
        row.residual_action
        for row in qca_boundary_reduction_rows()
        if row.verdict == QCABoundaryReductionVerdict.ALLOW
    )
    expected = allowed_scalar_s3_successor_elements()
    if allowed != expected:
        raise AssertionError(f"unexpected QCA scalar successor actions: {allowed}")
    return expected


def qca_scalar_boundary_allowed_labels() -> tuple[str, str]:
    """Return allowed scalar holomorphic successor labels from BCC framing."""

    labels = tuple(
        row.scalar_label
        for row in qca_boundary_reduction_rows()
        if row.verdict == QCABoundaryReductionVerdict.ALLOW
    )
    expected = allowed_scalar_s3_successor_labels()
    if labels != expected:
        raise AssertionError(f"unexpected QCA scalar successor labels: {labels}")
    return expected


def selected_exit_moving_controls_rejected() -> bool:
    """Return whether all selected-exit-moving automorphisms are rejected."""

    moving_rows = [
        row
        for row in qca_boundary_reduction_rows()
        if is_tetrahedral_automorphism(row.candidate)
        and not preserves_selected_exit(row.candidate)
    ]
    return len(moving_rows) == 18 and all(
        row.verdict == QCABoundaryReductionVerdict.FORBID
        and row.failure_class == QCABoundaryFailureClass.VACUUM_FRAME_BREAKING
        and QCABoundaryReductionVeto.VACUUM_FRAME_PRESERVING in row.vetoes
        for row in moving_rows
    )


def identity_control_rejected() -> bool:
    """Return whether the selected-preserving residual identity is same-state."""

    rows = [
        row
        for row in qca_boundary_reduction_rows()
        if row.residual_action == (0, 1, 2)
    ]
    return len(rows) == 1 and rows[0].failure_class == QCABoundaryFailureClass.SAME_STATE


def transposition_controls_rejected() -> bool:
    """Return whether selected-preserving residual transpositions are Z2 controls."""

    rows = [
        row
        for row in qca_boundary_reduction_rows()
        if row.failure_class == QCABoundaryFailureClass.HERMITIAN_Z2
    ]
    return len(rows) == 3 and all(
        row.verdict == QCABoundaryReductionVerdict.FORBID
        and QCABoundaryReductionVeto.HERMITIAN_Z2_SECTOR in row.vetoes
        for row in rows
    )


def non_automorphism_controls_rejected() -> bool:
    """Return whether explicit non-automorphism controls are rejected."""

    return all(
        row.verdict == QCABoundaryReductionVerdict.FORBID
        and row.failure_class == QCABoundaryFailureClass.NON_AUTOMORPHISM
        and QCABoundaryReductionVeto.TETRAHEDRAL_AUTOMORPHISM in row.vetoes
        for row in qca_non_automorphism_control_rows()
    )


def qca_s3_reduction_pass() -> bool:
    """Return whether R9 reduces the scalar boundary sector to R8's S3 pair."""

    return (
        len(all_tetrahedral_exit_automorphisms()) == 24
        and selected_exit_stabilizer_count() == 6
        and selected_exit_moving_controls_rejected()
        and induced_residual_image_is_full_s3()
        and qca_scalar_boundary_allowed_labels() == ("triality_plus", "triality_minus")
        and identity_control_rejected()
        and transposition_controls_rejected()
        and non_automorphism_controls_rejected()
    )


def qca_s3_reduction_payload() -> QCAS3ReductionPayload:
    """Return the R9 QCA-to-S3 scalar reduction verdict."""

    rows = qca_boundary_reduction_rows()
    controls = qca_non_automorphism_control_rows()
    induced_full = induced_residual_image_is_full_s3()
    identity_rejected = identity_control_rejected()
    transpositions_rejected = transposition_controls_rejected()
    non_auto_rejected = non_automorphism_controls_rejected()
    allowed_labels = qca_scalar_boundary_allowed_labels()
    allowed_elements = qca_scalar_boundary_allowed_elements()
    r8_matches = (
        allowed_labels == allowed_scalar_s3_successor_labels()
        and allowed_elements == allowed_scalar_s3_successor_elements()
    )
    checks_pass = (
        qca_s3_reduction_pass()
        and induced_full
        and identity_rejected
        and transpositions_rejected
        and non_auto_rejected
        and r8_matches
    )

    if checks_pass:
        final_verdict = "QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS"
        interpretation = (
            "Vacuum-frame-preserving tetrahedral exit automorphisms have "
            "selected-exit stabilizer S3 and induce the full residual S3 shell. "
            "After the scalar holomorphic restriction, only the two non-identity "
            "A3 cycles remain, matching R8. The result depends on the premise "
            "that one-tick scalar boundary repair is represented by such "
            "vacuum-frame-preserving tetrahedral automorphisms."
        )
    else:
        final_verdict = "QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_KILL"
        interpretation = "The BCC/vacuum-framed automorphism census failed to reduce scalar repair to R8's S3 pair."

    return QCAS3ReductionPayload(
        final_verdict=final_verdict,
        tetrahedral_automorphism_count=len(all_tetrahedral_exit_automorphisms()),
        selected_stabilizer_count=selected_exit_stabilizer_count(),
        selected_moving_rejected_count=len(
            [
                row
                for row in rows
                if row.failure_class == QCABoundaryFailureClass.VACUUM_FRAME_BREAKING
            ]
        ),
        induced_s3_count=len(induced_residual_s3_image()),
        induced_s3_is_full=induced_full,
        identity_control_rejected=identity_rejected,
        transposition_controls_rejected=transpositions_rejected,
        non_automorphism_controls_rejected=non_auto_rejected,
        allowed_successors=allowed_labels,
        allowed_elements=allowed_elements,
        r8_successors_match=r8_matches,
        microscopic_automorphism_premise_derived=False,
        candidate_rows=rows,
        non_automorphism_control_rows=controls,
        interpretation=interpretation,
    )
