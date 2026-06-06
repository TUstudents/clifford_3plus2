"""Aggregate tests for radial_response."""

from clifford_3plus2_d5.radial_response import radial_response_audit_payload


def test_radial_response_audit_payload_passes() -> None:
    payload = radial_response_audit_payload()
    assert payload.final_verdict == "RADIAL_RESPONSE_THEORY_PASS"
    assert payload.green_function.final_verdict == "MASS_AS_BOUNDARY_RECIRCULATION_PASS"
    assert payload.up_stacking.final_verdict == "UP_STACKING_LAW_EXPONENTIAL_FAVORED"
    assert payload.literal_nilpotent_yukawa.final_verdict == "LITERAL_NILPOTENT_YUKAWA_KILL"
    assert payload.down_dark_line.final_verdict == "DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED"
    assert payload.two_channel_isometry.final_verdict == "TWO_CHANNEL_REPAIR_ISOMETRY_PASS"
    assert payload.unitary_defect.final_verdict == "MINIMAL_UNITARY_S3_DEFECT_FORM_PASS"
    assert (
        payload.successor_certificate.final_verdict
        == "SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS"
    )
    assert payload.s3_shell_completeness.final_verdict == "SCALAR_S3_SHELL_COMPLETENESS_PASS"
    assert payload.qca_s3_reduction.final_verdict == "QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS"
    assert (
        payload.scalar_automorphism_premise.final_verdict
        == "SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS"
    )
    assert (
        payload.silver_transfer_inheritance.final_verdict
        == "RADIAL_SILVER_TRANSFER_INHERITANCE_PASS"
    )
    assert (
        payload.pole_residue_rigidity.final_verdict
        == "RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS"
    )
    assert (
        payload.spectral_measure_selection.final_verdict
        == "RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY"
    )
