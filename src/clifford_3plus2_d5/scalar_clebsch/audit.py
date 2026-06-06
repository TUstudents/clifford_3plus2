"""Aggregate scalar Clebsch audit."""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    DownSubsetAuditPayload,
    down_subset_audit_payload,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    S3ProjectorAuditPayload,
    s3_projector_audit_payload,
)
from clifford_3plus2_d5.scalar_clebsch.taylor_up import (
    TaylorUpAuditPayload,
    taylor_up_audit_payload,
)


@dataclass(frozen=True)
class ScalarClebschAuditPayload:
    """Combined verdict for the scalar Clebsch sidecar."""

    final_verdict: str
    up: TaylorUpAuditPayload
    down: DownSubsetAuditPayload
    s3_projectors: S3ProjectorAuditPayload
    interpretation: str


def scalar_clebsch_audit_payload() -> ScalarClebschAuditPayload:
    """Return the aggregate scalar Clebsch verdict."""

    up = taylor_up_audit_payload()
    down = down_subset_audit_payload()
    s3_projectors = s3_projector_audit_payload()
    checks_pass = (
        up.final_verdict == "NILPOTENT_TAYLOR_UP_CLEBSCH_PASS"
        and down.final_verdict == "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
        and s3_projectors.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
    )

    if checks_pass:
        final_verdict = "SCALAR_CLEBSCH_THEORY_CONDITIONAL_PASS"
        interpretation = (
            "The up-sector scalar Clebsches follow from a length-3 nilpotent "
            "Taylor response with x=1/sqrt(2). The down sector now separates "
            "the S3/projector baseline (6,2,4) from the data-improved "
            "odd-shell candidate (6,2,5). The regular S3 projector audit "
            "shows that (6,2,5) is available but not forced by S3 alone: rank "
            "2 requires defect polarization and rank 5 requires choosing "
            "which one-dimensional line is excluded. This is a scalar "
            "mass-coefficient story, not a CKM current-amplitude story."
        )
    else:
        final_verdict = "SCALAR_CLEBSCH_THEORY_KILL"
        interpretation = (
            "The nilpotent Taylor up-sector gate or the down-sector "
            "baseline/candidate/projector gate failed."
        )

    return ScalarClebschAuditPayload(
        final_verdict=final_verdict,
        up=up,
        down=down,
        s3_projectors=s3_projectors,
        interpretation=interpretation,
    )
