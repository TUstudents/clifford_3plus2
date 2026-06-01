"""V12 unique-successor enumeration certificate.

V11 proves that unique active successors imply no leakage.  V12 implements the
finite certificate format for that condition.  The certificate is a table over a
finite local boundary basis: each active source/candidate pair is either
``ALLOW`` or ``FORBID`` with at least one exact veto.

This module certifies the currently modeled residual-boundary candidate basis.
It does not derive that this basis is the complete microscopic BCC-QCA local
boundary basis.  That basis-completeness step remains a separate physical gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clifford_3plus2_d5.depth_scar.selection_no_leakage import (
    allowed_sets_are_unique_successors,
    unique_successor_no_leakage_pass,
)


class Veto(StrEnum):
    """Exact selection-rule veto labels."""

    LOCALITY = "LOCALITY"
    HEIGHT = "HEIGHT"
    BCC_PARITY = "BCC_PARITY"
    COLOR = "COLOR"
    WEYL = "WEYL"
    BOUNDARY_SECTOR = "BOUNDARY_SECTOR"
    SUPERSELECTION = "SUPERSELECTION"


class Verdict(StrEnum):
    """Transition certificate verdict."""

    ALLOW = "ALLOW"
    FORBID = "FORBID"


class FailureClass(StrEnum):
    """Interpretation of a forbidden transition."""

    ALLOW = "ALLOW"
    EXTERNAL_LEAKAGE = "EXTERNAL_LEAKAGE"
    SHORTCUT_REPAIR = "SHORTCUT_REPAIR"
    FORBIDDEN_REPAIRED_RANGE = "FORBIDDEN_REPAIRED_RANGE"


@dataclass(frozen=True)
class BoundaryState:
    """Finite local boundary candidate state used by the V12 certificate."""

    label: str
    height: int
    path_position: int
    bcc_parity: int
    color: str
    weyl: str
    boundary_sector: str
    superselection: str


@dataclass(frozen=True)
class TransitionCertificateRow:
    """One row of the finite V12 transition certificate."""

    source: str
    target: str
    verdict: Verdict
    vetoes: tuple[Veto, ...]
    failure_class: FailureClass


ACTIVE_SOURCES = ("a", "b")
TARGET_SUCCESSORS = {"a": "u", "b": "a"}
REPAIRED_RANGE = ("u", "a")


def local_boundary_basis() -> tuple[BoundaryState, ...]:
    """Return the finite candidate basis for the V12 certificate.

    The first three states are the residual ports.  The remaining states are
    exact-veto controls for bulk/spectator leakage and internal-sector mismatch.
    """

    return (
        BoundaryState("u", 0, 0, 0, "singlet", "compatible", "repair", "residual"),
        BoundaryState("a", 1, 1, 1, "singlet", "compatible", "repair", "residual"),
        BoundaryState("b", 2, 2, 0, "singlet", "compatible", "repair", "residual"),
        BoundaryState("bulk_u", 0, 0, 0, "singlet", "compatible", "bulk", "residual"),
        BoundaryState("bulk_a", 1, 1, 1, "singlet", "compatible", "bulk", "residual"),
        BoundaryState("spectator_u", 0, 0, 0, "singlet", "compatible", "spectator", "residual"),
        BoundaryState("wrong_color_u", 0, 0, 0, "triplet", "compatible", "repair", "residual"),
        BoundaryState("wrong_color_a", 1, 1, 1, "triplet", "compatible", "repair", "residual"),
        BoundaryState("wrong_weyl_u", 0, 0, 0, "singlet", "wrong", "repair", "residual"),
        BoundaryState("wrong_weyl_a", 1, 1, 1, "singlet", "wrong", "repair", "residual"),
        BoundaryState("orthogonal_coin_u", 0, 0, 0, "singlet", "compatible", "repair", "orthogonal_coin"),
        BoundaryState("orthogonal_coin_a", 1, 1, 1, "singlet", "compatible", "repair", "orthogonal_coin"),
    )


def state_by_label(label: str) -> BoundaryState:
    """Return a basis state by label."""

    for state in local_boundary_basis():
        if state.label == label:
            return state
    raise KeyError(label)


def boundary_distance(source: BoundaryState, target: BoundaryState) -> int:
    """Return residual path distance between source and target positions."""

    return abs(source.path_position - target.path_position)


def transition_vetoes(source_label: str, target_label: str) -> tuple[Veto, ...]:
    """Return all exact vetoes for a candidate transition."""

    source = state_by_label(source_label)
    target = state_by_label(target_label)
    vetoes: list[Veto] = []

    if boundary_distance(source, target) != 1:
        vetoes.append(Veto.LOCALITY)
    if target.height != source.height - 1:
        vetoes.append(Veto.HEIGHT)
    if target.bcc_parity != 1 - source.bcc_parity:
        vetoes.append(Veto.BCC_PARITY)
    if target.color != source.color:
        vetoes.append(Veto.COLOR)
    if target.weyl != source.weyl:
        vetoes.append(Veto.WEYL)
    if target.boundary_sector != "repair":
        vetoes.append(Veto.BOUNDARY_SECTOR)
    if target.superselection != "residual":
        vetoes.append(Veto.SUPERSELECTION)

    return tuple(vetoes)


def transition_verdict(source_label: str, target_label: str) -> Verdict:
    """Return the transition verdict."""

    return Verdict.ALLOW if not transition_vetoes(source_label, target_label) else Verdict.FORBID


def transition_failure_class(source_label: str, target_label: str) -> FailureClass:
    """Return the leakage/topology class of a transition."""

    if transition_verdict(source_label, target_label) == Verdict.ALLOW:
        return FailureClass.ALLOW
    if source_label == "b" and target_label == "u":
        return FailureClass.SHORTCUT_REPAIR
    if target_label not in REPAIRED_RANGE:
        return FailureClass.EXTERNAL_LEAKAGE
    return FailureClass.FORBIDDEN_REPAIRED_RANGE


def transition_certificate_row(source_label: str, target_label: str) -> TransitionCertificateRow:
    """Return one transition-certificate row."""

    vetoes = transition_vetoes(source_label, target_label)
    verdict = Verdict.ALLOW if not vetoes else Verdict.FORBID
    return TransitionCertificateRow(
        source=source_label,
        target=target_label,
        verdict=verdict,
        vetoes=vetoes,
        failure_class=transition_failure_class(source_label, target_label),
    )


def transition_certificate() -> tuple[TransitionCertificateRow, ...]:
    """Return the full finite V12 certificate table."""

    labels = tuple(state.label for state in local_boundary_basis())
    return tuple(
        transition_certificate_row(source, target)
        for source in ACTIVE_SOURCES
        for target in labels
    )


def certificate_is_complete() -> bool:
    """Return whether every active source/candidate pair has exactly one row."""

    labels = tuple(state.label for state in local_boundary_basis())
    pairs = {(row.source, row.target) for row in transition_certificate()}
    expected = {(source, target) for source in ACTIVE_SOURCES for target in labels}
    return pairs == expected and len(transition_certificate()) == len(expected)


def all_forbidden_rows_have_vetoes() -> bool:
    """Return whether every forbidden transition records at least one veto."""

    return all(
        row.verdict == Verdict.ALLOW or len(row.vetoes) > 0
        for row in transition_certificate()
    )


def allowed_successors_from_certificate() -> dict[str, tuple[str, ...]]:
    """Return allowed successor labels for each active source."""

    allowed: dict[str, list[str]] = {source: [] for source in ACTIVE_SOURCES}
    for row in transition_certificate():
        if row.verdict == Verdict.ALLOW:
            allowed[row.source].append(row.target)
    return {source: tuple(targets) for source, targets in allowed.items()}


def certificate_targets_pass() -> bool:
    """Return whether the certificate proves ``Omega(a)={u}``, ``Omega(b)={a}``."""

    return allowed_successors_from_certificate() == {
        "a": ("u",),
        "b": ("a",),
    }


def shortcut_repair_control_rejected() -> bool:
    """Return whether the direct ``b -> u`` shortcut is explicitly vetoed."""

    row = transition_certificate_row("b", "u")
    return (
        row.verdict == Verdict.FORBID
        and row.failure_class == FailureClass.SHORTCUT_REPAIR
        and Veto.LOCALITY in row.vetoes
        and Veto.BCC_PARITY in row.vetoes
    )


def external_leakage_controls_rejected() -> bool:
    """Return whether non-repair-sector and wrong-internal candidates are vetoed."""

    required = {
        ("a", "bulk_u"): Veto.BOUNDARY_SECTOR,
        ("a", "spectator_u"): Veto.BOUNDARY_SECTOR,
        ("a", "wrong_color_u"): Veto.COLOR,
        ("b", "wrong_color_a"): Veto.COLOR,
        ("a", "wrong_weyl_u"): Veto.WEYL,
        ("b", "wrong_weyl_a"): Veto.WEYL,
        ("a", "orthogonal_coin_u"): Veto.SUPERSELECTION,
        ("b", "orthogonal_coin_a"): Veto.SUPERSELECTION,
    }
    return all(
        transition_certificate_row(source, target).verdict == Verdict.FORBID
        and veto in transition_certificate_row(source, target).vetoes
        and transition_certificate_row(source, target).failure_class == FailureClass.EXTERNAL_LEAKAGE
        for (source, target), veto in required.items()
    )


def certificate_implies_v11_no_leakage() -> bool:
    """Return whether the certificate satisfies V11's unique-successor condition."""

    return (
        certificate_targets_pass()
        and allowed_sets_are_unique_successors(allowed_successors_from_certificate())
        and unique_successor_no_leakage_pass()
    )


def unique_successor_enumeration_certificate_pass() -> bool:
    """Return whether the V12 certificate gate passes."""

    return (
        certificate_is_complete()
        and all_forbidden_rows_have_vetoes()
        and certificate_targets_pass()
        and shortcut_repair_control_rejected()
        and external_leakage_controls_rejected()
        and certificate_implies_v11_no_leakage()
    )


@dataclass(frozen=True)
class SuccessorCertificatePayload:
    """V12 payload for the unique-successor enumeration certificate."""

    final_verdict: str
    basis_size: int
    row_count: int
    allowed_successors: dict[str, tuple[str, ...]]
    certificate_complete: bool
    forbidden_rows_have_vetoes: bool
    shortcut_repair_control_rejected: bool
    external_leakage_controls_rejected: bool
    certificate_implies_v11_no_leakage: bool
    microscopic_basis_completeness_derived: bool
    certificate: tuple[TransitionCertificateRow, ...]
    interpretation: str


def successor_certificate_payload() -> SuccessorCertificatePayload:
    """Return the V12 successor-certificate verdict."""

    complete = certificate_is_complete()
    vetoed = all_forbidden_rows_have_vetoes()
    targets = certificate_targets_pass()
    shortcut = shortcut_repair_control_rejected()
    external = external_leakage_controls_rejected()
    implies_v11 = certificate_implies_v11_no_leakage()

    checks_pass = complete and vetoed and targets and shortcut and external and implies_v11

    if checks_pass:
        final_verdict = "V12_UNIQUE_SUCCESSOR_ENUMERATION_CERTIFICATE_PASS"
        interpretation = (
            "The finite certificate table has exactly one allowed successor "
            "for each active source: Omega(a)={u} and Omega(b)={a}. Every "
            "forbidden candidate has an explicit selection-rule veto, the "
            "shortcut b->u is rejected as topology-changing shortcut repair, "
            "and external leakage controls are rejected. The table satisfies "
            "V11's no-leakage condition, but completeness of the microscopic "
            "basis is not derived."
        )
    elif not complete or not vetoed:
        final_verdict = "SUCCESSOR_CERTIFICATE_INCOMPLETE_KILL"
        interpretation = "The finite certificate is missing rows or forbidden rows without vetoes."
    elif not targets or not implies_v11:
        final_verdict = "UNIQUE_SUCCESSOR_ENUMERATION_KILL"
        interpretation = "The certificate does not prove Omega(a)={u}, Omega(b)={a}."
    else:
        final_verdict = "SUCCESSOR_CERTIFICATE_CONTROL_KILL"
        interpretation = "A shortcut, leakage, or V11 implication control failed."

    return SuccessorCertificatePayload(
        final_verdict=final_verdict,
        basis_size=len(local_boundary_basis()),
        row_count=len(transition_certificate()),
        allowed_successors=allowed_successors_from_certificate(),
        certificate_complete=complete,
        forbidden_rows_have_vetoes=vetoed,
        shortcut_repair_control_rejected=shortcut,
        external_leakage_controls_rejected=external,
        certificate_implies_v11_no_leakage=implies_v11,
        microscopic_basis_completeness_derived=False,
        certificate=transition_certificate(),
        interpretation=interpretation,
    )
