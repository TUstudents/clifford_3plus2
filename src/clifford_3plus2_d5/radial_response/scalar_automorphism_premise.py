"""R10 scalar automorphism premise gate.

R9 proves that vacuum-frame-preserving tetrahedral exit automorphisms reduce
the scalar boundary sector to the two triality successors.  This gate checks
the narrower bridge premise: a declared one-tick scalar-local boundary repair
map is sufficient to land inside the R9 automorphism census, while nonlocal,
non-scalar, non-deterministic, frame-breaking, and Hermitian/Z2 controls are
vetoed explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clifford_3plus2_d5.radial_response.qca_s3_reduction import (
    QCABoundaryFailureClass,
    QCABoundaryReductionVerdict,
    QCABoundaryReductionVeto,
    qca_boundary_reduction_row,
    qca_s3_reduction_pass,
)
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    IDENTITY,
    TRIALITY_MINUS,
    TRIALITY_PLUS,
    S3Element,
    transposition_elements,
)


SELECTED_EXIT = 0


class ScalarAutomorphismPremiseVeto(StrEnum):
    """Exact veto labels for scalar automorphism premise candidates."""

    ONE_TICK_LOCALITY = "ONE_TICK_LOCALITY"
    SCALAR_LOCALITY = "SCALAR_LOCALITY"
    DETERMINISTIC_EXIT_MAP = "DETERMINISTIC_EXIT_MAP"
    TETRAHEDRAL_AUTOMORPHISM = "TETRAHEDRAL_AUTOMORPHISM"
    VACUUM_FRAME_PRESERVING = "VACUUM_FRAME_PRESERVING"
    SCALAR_HOLOMORPHIC_SECTOR = "SCALAR_HOLOMORPHIC_SECTOR"


class ScalarAutomorphismPremiseVerdict(StrEnum):
    """Verdict for one scalar automorphism premise candidate."""

    ALLOW = "ALLOW"
    FORBID = "FORBID"


class ScalarAutomorphismFailureClass(StrEnum):
    """Interpretation of a forbidden scalar automorphism candidate."""

    ALLOW = "ALLOW"
    NONLOCAL = "NONLOCAL"
    NON_SCALAR_LOCAL = "NON_SCALAR_LOCAL"
    NON_DETERMINISTIC_EXIT_MAP = "NON_DETERMINISTIC_EXIT_MAP"
    NON_AUTOMORPHISM = "NON_AUTOMORPHISM"
    VACUUM_FRAME_BREAKING = "VACUUM_FRAME_BREAKING"
    SAME_STATE = "SAME_STATE"
    HERMITIAN_Z2 = "HERMITIAN_Z2"


@dataclass(frozen=True)
class ScalarBoundaryMapCandidate:
    """Finite candidate for a declared scalar-local one-tick boundary map."""

    label: str
    exit_map: tuple[int, ...] | None
    one_tick_local: bool
    scalar_local: bool
    deterministic_exit_map: bool


@dataclass(frozen=True)
class ScalarAutomorphismPremiseRow:
    """One row in the R10 scalar automorphism premise certificate."""

    candidate: ScalarBoundaryMapCandidate
    verdict: ScalarAutomorphismPremiseVerdict
    vetoes: tuple[ScalarAutomorphismPremiseVeto, ...]
    failure_class: ScalarAutomorphismFailureClass
    residual_action: S3Element | None
    scalar_label: str | None


@dataclass(frozen=True)
class ScalarAutomorphismPremisePayload:
    """Payload for the R10 scalar automorphism premise gate."""

    final_verdict: str
    candidate_count: int
    allowed_maps: tuple[str, ...]
    allowed_successors: tuple[str, str]
    certificate_complete: bool
    forbidden_rows_have_vetoes: bool
    selected_exit_moving_control_rejected: bool
    generic_linear_control_rejected: bool
    non_automorphism_control_rejected: bool
    nonlocal_control_rejected: bool
    nonscalar_control_rejected: bool
    identity_control_rejected: bool
    hermitian_z2_control_rejected: bool
    r9_reduction_applies: bool
    declared_scalar_local_map_sufficient: bool
    full_bb_qca_update_derived: bool
    certificate: tuple[ScalarAutomorphismPremiseRow, ...]
    interpretation: str


def lift_residual_action_to_selected_exit_map(
    residual_action: S3Element,
    selected: int = SELECTED_EXIT,
) -> tuple[int, ...]:
    """Lift a residual S3 action to an S4 map preserving ``selected``."""

    residual_exits = tuple(index for index in range(4) if index != selected)
    lifted = [0, 0, 0, 0]
    lifted[selected] = selected
    for source_coordinate, image_coordinate in enumerate(residual_action):
        source_exit = residual_exits[source_coordinate]
        image_exit = residual_exits[image_coordinate]
        lifted[source_exit] = image_exit
    return tuple(lifted)


def scalar_boundary_map_candidate_basis() -> tuple[ScalarBoundaryMapCandidate, ...]:
    """Return the finite modeled basis for the scalar automorphism premise."""

    return (
        ScalarBoundaryMapCandidate(
            label="triality_plus_automorphism",
            exit_map=lift_residual_action_to_selected_exit_map(TRIALITY_PLUS),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="triality_minus_automorphism",
            exit_map=lift_residual_action_to_selected_exit_map(TRIALITY_MINUS),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="selected_exit_moving_automorphism",
            exit_map=(1, 0, 2, 3),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="generic_linear_exit_mixture",
            exit_map=None,
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=False,
        ),
        ScalarBoundaryMapCandidate(
            label="non_automorphism_exit_map",
            exit_map=(0, 0, 1, 2),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="two_tick_triality_map",
            exit_map=lift_residual_action_to_selected_exit_map(TRIALITY_PLUS),
            one_tick_local=False,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="spin_coupled_triality_map",
            exit_map=lift_residual_action_to_selected_exit_map(TRIALITY_PLUS),
            one_tick_local=True,
            scalar_local=False,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="identity_same_state_map",
            exit_map=lift_residual_action_to_selected_exit_map(IDENTITY),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
        ScalarBoundaryMapCandidate(
            label="hermitian_z2_transposition_map",
            exit_map=lift_residual_action_to_selected_exit_map(transposition_elements()[0]),
            one_tick_local=True,
            scalar_local=True,
            deterministic_exit_map=True,
        ),
    )


def scalar_boundary_map_candidate_by_label(label: str) -> ScalarBoundaryMapCandidate:
    """Return one scalar boundary-map candidate by label."""

    for candidate in scalar_boundary_map_candidate_basis():
        if candidate.label == label:
            return candidate
    raise KeyError(label)


def _mapped_failure_class(failure: QCABoundaryFailureClass) -> ScalarAutomorphismFailureClass:
    """Map an R9 failure class into the R10 interpretation."""

    if failure == QCABoundaryFailureClass.ALLOW:
        return ScalarAutomorphismFailureClass.ALLOW
    if failure == QCABoundaryFailureClass.NON_AUTOMORPHISM:
        return ScalarAutomorphismFailureClass.NON_AUTOMORPHISM
    if failure == QCABoundaryFailureClass.VACUUM_FRAME_BREAKING:
        return ScalarAutomorphismFailureClass.VACUUM_FRAME_BREAKING
    if failure == QCABoundaryFailureClass.SAME_STATE:
        return ScalarAutomorphismFailureClass.SAME_STATE
    if failure == QCABoundaryFailureClass.HERMITIAN_Z2:
        return ScalarAutomorphismFailureClass.HERMITIAN_Z2
    raise ValueError(f"unmapped R9 failure class: {failure}")


def _mapped_vetoes(
    vetoes: tuple[QCABoundaryReductionVeto, ...],
) -> tuple[ScalarAutomorphismPremiseVeto, ...]:
    """Map R9 vetoes into R10 veto labels."""

    mapped: list[ScalarAutomorphismPremiseVeto] = []
    for veto in vetoes:
        if veto == QCABoundaryReductionVeto.TETRAHEDRAL_AUTOMORPHISM:
            mapped.append(ScalarAutomorphismPremiseVeto.TETRAHEDRAL_AUTOMORPHISM)
        elif veto == QCABoundaryReductionVeto.VACUUM_FRAME_PRESERVING:
            mapped.append(ScalarAutomorphismPremiseVeto.VACUUM_FRAME_PRESERVING)
        elif veto in (
            QCABoundaryReductionVeto.SAME_STATE,
            QCABoundaryReductionVeto.HERMITIAN_Z2_SECTOR,
        ):
            mapped.append(ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR)
        else:
            raise ValueError(f"unmapped R9 veto: {veto}")
    return tuple(mapped)


def scalar_automorphism_premise_row(label: str) -> ScalarAutomorphismPremiseRow:
    """Return one R10 premise row by candidate label."""

    candidate = scalar_boundary_map_candidate_by_label(label)
    vetoes: list[ScalarAutomorphismPremiseVeto] = []
    if not candidate.one_tick_local:
        vetoes.append(ScalarAutomorphismPremiseVeto.ONE_TICK_LOCALITY)
    if not candidate.scalar_local:
        vetoes.append(ScalarAutomorphismPremiseVeto.SCALAR_LOCALITY)
    if not candidate.deterministic_exit_map or candidate.exit_map is None:
        vetoes.append(ScalarAutomorphismPremiseVeto.DETERMINISTIC_EXIT_MAP)
        return ScalarAutomorphismPremiseRow(
            candidate=candidate,
            verdict=ScalarAutomorphismPremiseVerdict.FORBID,
            vetoes=tuple(vetoes),
            failure_class=ScalarAutomorphismFailureClass.NON_DETERMINISTIC_EXIT_MAP,
            residual_action=None,
            scalar_label=None,
        )

    qca_row = qca_boundary_reduction_row(candidate.exit_map)
    vetoes.extend(_mapped_vetoes(qca_row.vetoes))
    verdict = (
        ScalarAutomorphismPremiseVerdict.ALLOW
        if not vetoes and qca_row.verdict == QCABoundaryReductionVerdict.ALLOW
        else ScalarAutomorphismPremiseVerdict.FORBID
    )
    if verdict == ScalarAutomorphismPremiseVerdict.ALLOW:
        failure_class = ScalarAutomorphismFailureClass.ALLOW
    elif not candidate.one_tick_local:
        failure_class = ScalarAutomorphismFailureClass.NONLOCAL
    elif not candidate.scalar_local:
        failure_class = ScalarAutomorphismFailureClass.NON_SCALAR_LOCAL
    else:
        failure_class = _mapped_failure_class(qca_row.failure_class)

    return ScalarAutomorphismPremiseRow(
        candidate=candidate,
        verdict=verdict,
        vetoes=tuple(vetoes),
        failure_class=failure_class,
        residual_action=qca_row.residual_action,
        scalar_label=qca_row.scalar_label,
    )


def scalar_automorphism_premise_certificate() -> tuple[ScalarAutomorphismPremiseRow, ...]:
    """Return the complete R10 scalar automorphism premise certificate."""

    return tuple(
        scalar_automorphism_premise_row(candidate.label)
        for candidate in scalar_boundary_map_candidate_basis()
    )


def scalar_automorphism_certificate_is_complete() -> bool:
    """Return whether every candidate has exactly one premise row."""

    labels = tuple(candidate.label for candidate in scalar_boundary_map_candidate_basis())
    row_labels = tuple(row.candidate.label for row in scalar_automorphism_premise_certificate())
    return row_labels == labels and len(row_labels) == len(set(row_labels))


def scalar_automorphism_forbidden_rows_have_vetoes() -> bool:
    """Return whether every forbidden candidate records at least one veto."""

    return all(
        row.verdict == ScalarAutomorphismPremiseVerdict.ALLOW or len(row.vetoes) > 0
        for row in scalar_automorphism_premise_certificate()
    )


def scalar_automorphism_allowed_maps() -> tuple[str, ...]:
    """Return allowed scalar boundary-map labels."""

    return tuple(
        row.candidate.label
        for row in scalar_automorphism_premise_certificate()
        if row.verdict == ScalarAutomorphismPremiseVerdict.ALLOW
    )


def scalar_automorphism_allowed_successors() -> tuple[str, str]:
    """Return allowed scalar successor labels from the accepted maps."""

    labels = tuple(
        row.scalar_label
        for row in scalar_automorphism_premise_certificate()
        if row.verdict == ScalarAutomorphismPremiseVerdict.ALLOW
    )
    if labels != ("triality_plus", "triality_minus"):
        raise AssertionError(f"unexpected scalar automorphism successor labels: {labels}")
    return labels


def scalar_automorphism_control_rejected(
    label: str,
    failure_class: ScalarAutomorphismFailureClass,
    veto: ScalarAutomorphismPremiseVeto,
) -> bool:
    """Return whether one named control is rejected with the expected veto."""

    row = scalar_automorphism_premise_row(label)
    return (
        row.verdict == ScalarAutomorphismPremiseVerdict.FORBID
        and row.failure_class == failure_class
        and veto in row.vetoes
    )


def scalar_automorphism_premise_pass() -> bool:
    """Return whether R10 bridges declared scalar-local maps to R9."""

    return (
        scalar_automorphism_certificate_is_complete()
        and scalar_automorphism_forbidden_rows_have_vetoes()
        and scalar_automorphism_allowed_maps()
        == ("triality_plus_automorphism", "triality_minus_automorphism")
        and scalar_automorphism_allowed_successors() == ("triality_plus", "triality_minus")
        and scalar_automorphism_control_rejected(
            "selected_exit_moving_automorphism",
            ScalarAutomorphismFailureClass.VACUUM_FRAME_BREAKING,
            ScalarAutomorphismPremiseVeto.VACUUM_FRAME_PRESERVING,
        )
        and scalar_automorphism_control_rejected(
            "generic_linear_exit_mixture",
            ScalarAutomorphismFailureClass.NON_DETERMINISTIC_EXIT_MAP,
            ScalarAutomorphismPremiseVeto.DETERMINISTIC_EXIT_MAP,
        )
        and scalar_automorphism_control_rejected(
            "non_automorphism_exit_map",
            ScalarAutomorphismFailureClass.NON_AUTOMORPHISM,
            ScalarAutomorphismPremiseVeto.TETRAHEDRAL_AUTOMORPHISM,
        )
        and scalar_automorphism_control_rejected(
            "two_tick_triality_map",
            ScalarAutomorphismFailureClass.NONLOCAL,
            ScalarAutomorphismPremiseVeto.ONE_TICK_LOCALITY,
        )
        and scalar_automorphism_control_rejected(
            "spin_coupled_triality_map",
            ScalarAutomorphismFailureClass.NON_SCALAR_LOCAL,
            ScalarAutomorphismPremiseVeto.SCALAR_LOCALITY,
        )
        and scalar_automorphism_control_rejected(
            "identity_same_state_map",
            ScalarAutomorphismFailureClass.SAME_STATE,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        )
        and scalar_automorphism_control_rejected(
            "hermitian_z2_transposition_map",
            ScalarAutomorphismFailureClass.HERMITIAN_Z2,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        )
        and qca_s3_reduction_pass()
    )


def scalar_automorphism_premise_payload() -> ScalarAutomorphismPremisePayload:
    """Return the R10 scalar automorphism premise verdict."""

    certificate = scalar_automorphism_premise_certificate()
    complete = scalar_automorphism_certificate_is_complete()
    vetoed = scalar_automorphism_forbidden_rows_have_vetoes()
    checks_pass = scalar_automorphism_premise_pass()

    if checks_pass:
        final_verdict = "SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS"
        interpretation = (
            "A declared one-tick scalar-local boundary repair map with a "
            "deterministic tetrahedral exit action lands exactly in the R9 "
            "vacuum-frame-preserving automorphism census. The only accepted "
            "maps induce the two scalar holomorphic triality successors. "
            "Generic linear mixtures, non-automorphisms, selected-exit-moving "
            "maps, nonlocal maps, non-scalar maps, same-state identity, and "
            "Hermitian/Z2 transposition controls are rejected. This derives "
            "the R9 premise from the declared scalar-local map class, but not "
            "from the full BB/QCA microscopic update."
        )
    else:
        final_verdict = "SCALAR_AUTOMORPHISM_PREMISE_KILL"
        interpretation = "The declared scalar-local map class did not reduce to the R9 automorphism premise."

    return ScalarAutomorphismPremisePayload(
        final_verdict=final_verdict,
        candidate_count=len(certificate),
        allowed_maps=scalar_automorphism_allowed_maps(),
        allowed_successors=scalar_automorphism_allowed_successors(),
        certificate_complete=complete,
        forbidden_rows_have_vetoes=vetoed,
        selected_exit_moving_control_rejected=scalar_automorphism_control_rejected(
            "selected_exit_moving_automorphism",
            ScalarAutomorphismFailureClass.VACUUM_FRAME_BREAKING,
            ScalarAutomorphismPremiseVeto.VACUUM_FRAME_PRESERVING,
        ),
        generic_linear_control_rejected=scalar_automorphism_control_rejected(
            "generic_linear_exit_mixture",
            ScalarAutomorphismFailureClass.NON_DETERMINISTIC_EXIT_MAP,
            ScalarAutomorphismPremiseVeto.DETERMINISTIC_EXIT_MAP,
        ),
        non_automorphism_control_rejected=scalar_automorphism_control_rejected(
            "non_automorphism_exit_map",
            ScalarAutomorphismFailureClass.NON_AUTOMORPHISM,
            ScalarAutomorphismPremiseVeto.TETRAHEDRAL_AUTOMORPHISM,
        ),
        nonlocal_control_rejected=scalar_automorphism_control_rejected(
            "two_tick_triality_map",
            ScalarAutomorphismFailureClass.NONLOCAL,
            ScalarAutomorphismPremiseVeto.ONE_TICK_LOCALITY,
        ),
        nonscalar_control_rejected=scalar_automorphism_control_rejected(
            "spin_coupled_triality_map",
            ScalarAutomorphismFailureClass.NON_SCALAR_LOCAL,
            ScalarAutomorphismPremiseVeto.SCALAR_LOCALITY,
        ),
        identity_control_rejected=scalar_automorphism_control_rejected(
            "identity_same_state_map",
            ScalarAutomorphismFailureClass.SAME_STATE,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        ),
        hermitian_z2_control_rejected=scalar_automorphism_control_rejected(
            "hermitian_z2_transposition_map",
            ScalarAutomorphismFailureClass.HERMITIAN_Z2,
            ScalarAutomorphismPremiseVeto.SCALAR_HOLOMORPHIC_SECTOR,
        ),
        r9_reduction_applies=qca_s3_reduction_pass(),
        declared_scalar_local_map_sufficient=checks_pass,
        full_bb_qca_update_derived=False,
        certificate=certificate,
        interpretation=interpretation,
    )
