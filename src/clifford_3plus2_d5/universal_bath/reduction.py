"""Reduction taxonomy for universal-bath sectors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ReductionKind(StrEnum):
    """Allowed bath-reduction classes."""

    POSITIVE_JACOBI = "positive_scalar_jacobi"
    INDEFINITE_LOOKAHEAD_JACOBI = "symmetric_indefinite_lookahead_jacobi"
    CMV_OPUC = "unitary_cmv_opuc"


@dataclass(frozen=True)
class SectorReduction:
    """Reduction assignment for one sector or channel family."""

    sector: str
    reduction: ReductionKind
    reason: str


@dataclass(frozen=True)
class ReductionTaxonomyPayload:
    """Verdict payload for the reduction taxonomy."""

    final_verdict: str
    assignments: tuple[SectorReduction, ...]
    positive_sectors: tuple[str, ...]
    indefinite_sectors: tuple[str, ...]
    cmv_sectors: tuple[str, ...]
    interpretation: str


def sector_reductions() -> tuple[SectorReduction, ...]:
    """Return the current write-down of the bath-reduction taxonomy."""

    return (
        SectorReduction(
            "neutrino",
            ReductionKind.POSITIVE_JACOBI,
            "one-sided Hermitian positive Schur/seesaw response",
        ),
        SectorReduction(
            "kinetic_self_energy",
            ReductionKind.POSITIVE_JACOBI,
            "positive one-sided passive response",
        ),
        SectorReduction(
            "down_quark",
            ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI,
            "real symmetric shell with possible signature breakdown",
        ),
        SectorReduction(
            "charged_lepton",
            ReductionKind.CMV_OPUC,
            "unitary chiral boundary scattering with intrinsic frame rotation",
        ),
        SectorReduction(
            "up_quark",
            ReductionKind.CMV_OPUC,
            "holomorphic/nilpotent chiral entry into a unitary boundary process",
        ),
        SectorReduction(
            "ckm_phase",
            ReductionKind.CMV_OPUC,
            "phase should live in disk-valued Verblunsky data, not a real-line residue",
        ),
    )


def reduction_taxonomy_payload() -> ReductionTaxonomyPayload:
    """Return the reduction-taxonomy verdict."""

    assignments = sector_reductions()
    positive = tuple(
        item.sector for item in assignments if item.reduction == ReductionKind.POSITIVE_JACOBI
    )
    indefinite = tuple(
        item.sector
        for item in assignments
        if item.reduction == ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI
    )
    cmv = tuple(item.sector for item in assignments if item.reduction == ReductionKind.CMV_OPUC)
    all_classes_present = bool(positive and indefinite and cmv)

    if all_classes_present:
        final_verdict = "BATH_REDUCTION_TAXONOMY_PASS"
        interpretation = (
            "The sidecar does not force every sector into scalar Jacobi form. "
            "Positive one-sided responses use scalar Jacobi; real symmetric "
            "non-positive shells require indefinite look-ahead Lanczos; chiral "
            "unitary sectors and phases are assigned to CMV/OPUC, where phases "
            "are Verblunsky data on the disk."
        )
    else:
        final_verdict = "BATH_REDUCTION_TAXONOMY_INCOMPLETE"
        interpretation = "At least one reduction class is missing from the taxonomy."

    return ReductionTaxonomyPayload(
        final_verdict=final_verdict,
        assignments=assignments,
        positive_sectors=positive,
        indefinite_sectors=indefinite,
        cmv_sectors=cmv,
        interpretation=interpretation,
    )

