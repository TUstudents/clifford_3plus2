"""Aggregate tests for the scalar Clebsch sidecar."""

from clifford_3plus2_d5.scalar_clebsch import scalar_clebsch_audit_payload


def test_scalar_clebsch_audit_payload_passes() -> None:
    payload = scalar_clebsch_audit_payload()
    assert payload.final_verdict == "SCALAR_CLEBSCH_THEORY_CONDITIONAL_PASS"
    assert payload.up.final_verdict == "NILPOTENT_TAYLOR_UP_CLEBSCH_PASS"
    assert payload.down.final_verdict == "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
    assert payload.s3_projectors.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
