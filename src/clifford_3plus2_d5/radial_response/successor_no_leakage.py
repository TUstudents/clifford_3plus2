"""R7 finite successor/no-leakage certificate for scalar repair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScalarSuccessorVeto(StrEnum):
    """Exact veto labels for scalar repair successor candidates."""

    HEIGHT_LOWERING = "HEIGHT_LOWERING"
    ONE_TICK_REPAIR = "ONE_TICK_REPAIR"
    SCALAR_SECTOR = "SCALAR_SECTOR"
    Z2_CONJUGATE_PAIR = "Z2_CONJUGATE_PAIR"
    BOUNDARY_REPAIR_SECTOR = "BOUNDARY_REPAIR_SECTOR"


class ScalarSuccessorVerdict(StrEnum):
    """Certificate verdict for one scalar repair candidate."""

    ALLOW = "ALLOW"
    FORBID = "FORBID"


class ScalarFailureClass(StrEnum):
    """Interpretation of a forbidden scalar successor candidate."""

    ALLOW = "ALLOW"
    SAME_STATE = "SAME_STATE"
    WRONG_HEIGHT = "WRONG_HEIGHT"
    EXTERNAL_LEAKAGE = "EXTERNAL_LEAKAGE"
    ASYMMETRIC_SECTOR = "ASYMMETRIC_SECTOR"
    THIRD_SUCCESSOR = "THIRD_SUCCESSOR"
    NONLOCAL_REPAIR = "NONLOCAL_REPAIR"


@dataclass(frozen=True)
class ScalarSuccessorCandidate:
    """Finite candidate output state for the scalar repair certificate."""

    label: str
    height_lowering: bool
    one_tick_repair: bool
    scalar_sector: bool
    z2_conjugate_pair: bool
    boundary_repair_sector: bool
    failure_class: ScalarFailureClass


@dataclass(frozen=True)
class ScalarSuccessorCertificateRow:
    """One row in the finite scalar successor certificate."""

    source: str
    target: str
    verdict: ScalarSuccessorVerdict
    vetoes: tuple[ScalarSuccessorVeto, ...]
    failure_class: ScalarFailureClass


@dataclass(frozen=True)
class ScalarSuccessorCertificatePayload:
    """Payload for the R7 scalar successor/no-leakage gate."""

    final_verdict: str
    source: str
    basis_size: int
    row_count: int
    allowed_successors: tuple[str, ...]
    certificate_complete: bool
    forbidden_rows_have_vetoes: bool
    exactly_two_successors: bool
    z2_pair_successors: bool
    leakage_controls_rejected: bool
    third_successor_control_rejected: bool
    microscopic_basis_completeness_derived: bool
    certificate: tuple[ScalarSuccessorCertificateRow, ...]
    interpretation: str


SCALAR_REPAIR_SOURCE = "scalar_repair_seed"


def scalar_successor_candidate_basis() -> tuple[ScalarSuccessorCandidate, ...]:
    """Return the finite modeled output basis for one scalar repair tick."""

    return (
        ScalarSuccessorCandidate(
            label="triality_plus",
            height_lowering=True,
            one_tick_repair=True,
            scalar_sector=True,
            z2_conjugate_pair=True,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.ALLOW,
        ),
        ScalarSuccessorCandidate(
            label="triality_minus",
            height_lowering=True,
            one_tick_repair=True,
            scalar_sector=True,
            z2_conjugate_pair=True,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.ALLOW,
        ),
        ScalarSuccessorCandidate(
            label="same_state",
            height_lowering=False,
            one_tick_repair=False,
            scalar_sector=True,
            z2_conjugate_pair=False,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.SAME_STATE,
        ),
        ScalarSuccessorCandidate(
            label="wrong_height",
            height_lowering=False,
            one_tick_repair=True,
            scalar_sector=True,
            z2_conjugate_pair=True,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.WRONG_HEIGHT,
        ),
        ScalarSuccessorCandidate(
            label="two_tick_repair",
            height_lowering=True,
            one_tick_repair=False,
            scalar_sector=True,
            z2_conjugate_pair=True,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.NONLOCAL_REPAIR,
        ),
        ScalarSuccessorCandidate(
            label="external_leakage",
            height_lowering=True,
            one_tick_repair=True,
            scalar_sector=True,
            z2_conjugate_pair=True,
            boundary_repair_sector=False,
            failure_class=ScalarFailureClass.EXTERNAL_LEAKAGE,
        ),
        ScalarSuccessorCandidate(
            label="asymmetric_sector",
            height_lowering=True,
            one_tick_repair=True,
            scalar_sector=False,
            z2_conjugate_pair=True,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.ASYMMETRIC_SECTOR,
        ),
        ScalarSuccessorCandidate(
            label="third_repair_successor",
            height_lowering=True,
            one_tick_repair=True,
            scalar_sector=True,
            z2_conjugate_pair=False,
            boundary_repair_sector=True,
            failure_class=ScalarFailureClass.THIRD_SUCCESSOR,
        ),
    )


def scalar_candidate_by_label(label: str) -> ScalarSuccessorCandidate:
    """Return one scalar successor candidate by label."""

    for candidate in scalar_successor_candidate_basis():
        if candidate.label == label:
            return candidate
    raise KeyError(label)


def scalar_successor_vetoes(target_label: str) -> tuple[ScalarSuccessorVeto, ...]:
    """Return all exact vetoes for a scalar repair candidate."""

    candidate = scalar_candidate_by_label(target_label)
    vetoes: list[ScalarSuccessorVeto] = []
    if not candidate.height_lowering:
        vetoes.append(ScalarSuccessorVeto.HEIGHT_LOWERING)
    if not candidate.one_tick_repair:
        vetoes.append(ScalarSuccessorVeto.ONE_TICK_REPAIR)
    if not candidate.scalar_sector:
        vetoes.append(ScalarSuccessorVeto.SCALAR_SECTOR)
    if not candidate.z2_conjugate_pair:
        vetoes.append(ScalarSuccessorVeto.Z2_CONJUGATE_PAIR)
    if not candidate.boundary_repair_sector:
        vetoes.append(ScalarSuccessorVeto.BOUNDARY_REPAIR_SECTOR)
    return tuple(vetoes)


def scalar_successor_certificate_row(target_label: str) -> ScalarSuccessorCertificateRow:
    """Return one scalar successor certificate row."""

    candidate = scalar_candidate_by_label(target_label)
    vetoes = scalar_successor_vetoes(target_label)
    verdict = ScalarSuccessorVerdict.ALLOW if not vetoes else ScalarSuccessorVerdict.FORBID
    failure_class = ScalarFailureClass.ALLOW if verdict == ScalarSuccessorVerdict.ALLOW else candidate.failure_class
    return ScalarSuccessorCertificateRow(
        source=SCALAR_REPAIR_SOURCE,
        target=target_label,
        verdict=verdict,
        vetoes=vetoes,
        failure_class=failure_class,
    )


def scalar_successor_certificate() -> tuple[ScalarSuccessorCertificateRow, ...]:
    """Return the complete finite scalar successor certificate."""

    return tuple(
        scalar_successor_certificate_row(candidate.label)
        for candidate in scalar_successor_candidate_basis()
    )


def scalar_successor_certificate_is_complete() -> bool:
    """Return whether every candidate has exactly one certificate row."""

    labels = tuple(candidate.label for candidate in scalar_successor_candidate_basis())
    row_targets = tuple(row.target for row in scalar_successor_certificate())
    return row_targets == labels and len(row_targets) == len(set(row_targets))


def scalar_forbidden_rows_have_vetoes() -> bool:
    """Return whether every forbidden candidate records at least one veto."""

    return all(
        row.verdict == ScalarSuccessorVerdict.ALLOW or len(row.vetoes) > 0
        for row in scalar_successor_certificate()
    )


def scalar_allowed_successors() -> tuple[str, ...]:
    """Return allowed scalar repair successor labels."""

    return tuple(
        row.target
        for row in scalar_successor_certificate()
        if row.verdict == ScalarSuccessorVerdict.ALLOW
    )


def scalar_successors_are_z2_pair() -> bool:
    """Return whether the allowed successors are the intended Z2 pair."""

    return scalar_allowed_successors() == ("triality_plus", "triality_minus")


def scalar_leakage_controls_rejected() -> bool:
    """Return whether leakage/asymmetry/nonlocal controls are vetoed."""

    required = {
        "same_state": ScalarSuccessorVeto.HEIGHT_LOWERING,
        "wrong_height": ScalarSuccessorVeto.HEIGHT_LOWERING,
        "two_tick_repair": ScalarSuccessorVeto.ONE_TICK_REPAIR,
        "external_leakage": ScalarSuccessorVeto.BOUNDARY_REPAIR_SECTOR,
        "asymmetric_sector": ScalarSuccessorVeto.SCALAR_SECTOR,
    }
    return all(
        scalar_successor_certificate_row(label).verdict == ScalarSuccessorVerdict.FORBID
        and veto in scalar_successor_certificate_row(label).vetoes
        for label, veto in required.items()
    )


def scalar_third_successor_control_rejected() -> bool:
    """Return whether a third otherwise repair-like successor is vetoed."""

    row = scalar_successor_certificate_row("third_repair_successor")
    return (
        row.verdict == ScalarSuccessorVerdict.FORBID
        and row.failure_class == ScalarFailureClass.THIRD_SUCCESSOR
        and ScalarSuccessorVeto.Z2_CONJUGATE_PAIR in row.vetoes
    )


def scalar_successor_no_leakage_certificate_pass() -> bool:
    """Return whether R7 proves the R5 successor/no-leakage condition."""

    return (
        scalar_successor_certificate_is_complete()
        and scalar_forbidden_rows_have_vetoes()
        and len(scalar_allowed_successors()) == 2
        and scalar_successors_are_z2_pair()
        and scalar_leakage_controls_rejected()
        and scalar_third_successor_control_rejected()
    )


def scalar_successor_certificate_payload() -> ScalarSuccessorCertificatePayload:
    """Return the R7 scalar successor/no-leakage verdict."""

    complete = scalar_successor_certificate_is_complete()
    vetoed = scalar_forbidden_rows_have_vetoes()
    exactly_two = len(scalar_allowed_successors()) == 2
    z2_pair = scalar_successors_are_z2_pair()
    leakage_rejected = scalar_leakage_controls_rejected()
    third_rejected = scalar_third_successor_control_rejected()
    checks_pass = (
        complete
        and vetoed
        and exactly_two
        and z2_pair
        and leakage_rejected
        and third_rejected
    )

    if checks_pass:
        final_verdict = "SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS"
        interpretation = (
            "Within the finite modeled scalar candidate basis, the only "
            "allowed one-tick repair outputs are the Z2-conjugate pair "
            "triality_plus/triality_minus. Every forbidden control has an "
            "explicit selection-rule veto. This certifies R5's two-channel "
            "no-leakage hypothesis, but does not prove microscopic basis "
            "completeness for the full QCA boundary Hilbert space."
        )
    elif not complete or not vetoed:
        final_verdict = "SCALAR_SUCCESSOR_CERTIFICATE_INCOMPLETE_KILL"
        interpretation = "The scalar successor certificate is incomplete or has veto-free forbidden rows."
    elif not exactly_two or not z2_pair:
        final_verdict = "SCALAR_TWO_SUCCESSOR_CONDITION_KILL"
        interpretation = "The finite certificate does not leave exactly the Z2-conjugate repair pair."
    else:
        final_verdict = "SCALAR_SUCCESSOR_CONTROL_KILL"
        interpretation = "A leakage, asymmetry, nonlocal, or third-successor control failed."

    return ScalarSuccessorCertificatePayload(
        final_verdict=final_verdict,
        source=SCALAR_REPAIR_SOURCE,
        basis_size=len(scalar_successor_candidate_basis()),
        row_count=len(scalar_successor_certificate()),
        allowed_successors=scalar_allowed_successors(),
        certificate_complete=complete,
        forbidden_rows_have_vetoes=vetoed,
        exactly_two_successors=exactly_two,
        z2_pair_successors=z2_pair,
        leakage_controls_rejected=leakage_rejected,
        third_successor_control_rejected=third_rejected,
        microscopic_basis_completeness_derived=False,
        certificate=scalar_successor_certificate(),
        interpretation=interpretation,
    )
