"""Down-sector dark-line framing for the rank-5 bottom coefficient."""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    S3ProjectorAuditPayload,
    s3_projector_audit_payload,
)


@dataclass(frozen=True)
class DownDarkLinePayload:
    """Payload for the down-sector dark-line gate."""

    final_verdict: str
    s3_projectors: S3ProjectorAuditPayload
    rank_five_available: bool
    rank_five_forced: bool
    rank_two_standard_forced: bool
    interpretation: str


def down_dark_line_payload() -> DownDarkLinePayload:
    """Return the down-sector regular-minus-dark-line verdict."""

    s3_projectors = s3_projector_audit_payload()
    rank_five_available = set(s3_projectors.rank_five_ranks.values()) == {5}
    rank_five_forced = not s3_projectors.rank_five_not_unique
    rank_two_standard_forced = not s3_projectors.rank_two_requires_defect_polarization
    checks_pass = (
        s3_projectors.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
        and rank_five_available
        and not rank_five_forced
        and not rank_two_standard_forced
    )

    if checks_pass:
        final_verdict = "DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED"
        interpretation = (
            "The data-improved bottom coefficient can be read as regular S3 "
            "minus one forbidden one-dimensional line, but the regular shell "
            "has two such complements and S3 alone does not choose one. The "
            "middle rank-2 standard copy also needs defect polarization. Thus "
            "the down radial residue is available as a dark-line response, not "
            "derived from S3 alone."
        )
    else:
        final_verdict = "DOWN_DARK_LINE_GATE_KILL"
        interpretation = (
            "The S3 projector audit did not support the rank-5 dark-line "
            "availability with the expected non-forced status."
        )

    return DownDarkLinePayload(
        final_verdict=final_verdict,
        s3_projectors=s3_projectors,
        rank_five_available=rank_five_available,
        rank_five_forced=rank_five_forced,
        rank_two_standard_forced=rank_two_standard_forced,
        interpretation=interpretation,
    )
