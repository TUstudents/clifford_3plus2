"""Aggregate audit for radial mass-response theory."""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.radial_response.down_dark_line import (
    DownDarkLinePayload,
    down_dark_line_payload,
)
from clifford_3plus2_d5.radial_response.green_function import (
    GreenFunctionPayload,
    green_function_payload,
)
from clifford_3plus2_d5.radial_response.literal_nilpotent_yukawa import (
    LiteralNilpotentYukawaPayload,
    literal_nilpotent_yukawa_payload,
)
from clifford_3plus2_d5.radial_response.pole_residue_rigidity import (
    PoleResidueRigidityPayload,
    pole_residue_rigidity_payload,
)
from clifford_3plus2_d5.radial_response.qca_s3_reduction import (
    QCAS3ReductionPayload,
    qca_s3_reduction_payload,
)
from clifford_3plus2_d5.radial_response.s3_scalar_completeness import (
    ScalarS3ShellCompletenessPayload,
    scalar_s3_shell_completeness_payload,
)
from clifford_3plus2_d5.radial_response.scalar_automorphism_premise import (
    ScalarAutomorphismPremisePayload,
    scalar_automorphism_premise_payload,
)
from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import (
    SilverTransferInheritancePayload,
    silver_transfer_inheritance_payload,
)
from clifford_3plus2_d5.radial_response.spectral_measure_selection import (
    SpectralMeasureSelectionPayload,
    spectral_measure_selection_payload,
)
from clifford_3plus2_d5.radial_response.successor_no_leakage import (
    ScalarSuccessorCertificatePayload,
    scalar_successor_certificate_payload,
)
from clifford_3plus2_d5.radial_response.two_channel_isometry import (
    TwoChannelIsometryPayload,
    two_channel_isometry_payload,
)
from clifford_3plus2_d5.radial_response.unitary_defect import (
    MinimalUnitaryDefectPayload,
    minimal_unitary_defect_payload,
)
from clifford_3plus2_d5.radial_response.up_stacking import (
    UpStackingPayload,
    up_stacking_payload,
)


@dataclass(frozen=True)
class RadialResponseAuditPayload:
    """Combined verdict for the radial-response sidecar."""

    final_verdict: str
    green_function: GreenFunctionPayload
    up_stacking: UpStackingPayload
    literal_nilpotent_yukawa: LiteralNilpotentYukawaPayload
    down_dark_line: DownDarkLinePayload
    two_channel_isometry: TwoChannelIsometryPayload
    unitary_defect: MinimalUnitaryDefectPayload
    successor_certificate: ScalarSuccessorCertificatePayload
    s3_shell_completeness: ScalarS3ShellCompletenessPayload
    qca_s3_reduction: QCAS3ReductionPayload
    scalar_automorphism_premise: ScalarAutomorphismPremisePayload
    silver_transfer_inheritance: SilverTransferInheritancePayload
    pole_residue_rigidity: PoleResidueRigidityPayload
    spectral_measure_selection: SpectralMeasureSelectionPayload
    interpretation: str


def radial_response_audit_payload() -> RadialResponseAuditPayload:
    """Return the aggregate radial-response verdict."""

    green = green_function_payload()
    up = up_stacking_payload()
    literal = literal_nilpotent_yukawa_payload()
    down = down_dark_line_payload()
    two_channel = two_channel_isometry_payload()
    unitary = minimal_unitary_defect_payload()
    successor = scalar_successor_certificate_payload()
    s3_shell = scalar_s3_shell_completeness_payload()
    qca_reduction = qca_s3_reduction_payload()
    scalar_automorphism = scalar_automorphism_premise_payload()
    silver_transfer = silver_transfer_inheritance_payload()
    pole_residue = pole_residue_rigidity_payload()
    spectral_measure = spectral_measure_selection_payload()
    checks_pass = (
        green.final_verdict == "MASS_AS_BOUNDARY_RECIRCULATION_PASS"
        and up.final_verdict == "UP_STACKING_LAW_EXPONENTIAL_FAVORED"
        and literal.final_verdict == "LITERAL_NILPOTENT_YUKAWA_KILL"
        and down.final_verdict == "DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED"
        and two_channel.final_verdict == "TWO_CHANNEL_REPAIR_ISOMETRY_PASS"
        and unitary.final_verdict == "MINIMAL_UNITARY_S3_DEFECT_FORM_PASS"
        and successor.final_verdict == "SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS"
        and s3_shell.final_verdict == "SCALAR_S3_SHELL_COMPLETENESS_PASS"
        and qca_reduction.final_verdict == "QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS"
        and scalar_automorphism.final_verdict
        == "SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS"
        and silver_transfer.final_verdict == "RADIAL_SILVER_TRANSFER_INHERITANCE_PASS"
        and pole_residue.final_verdict == "RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS"
        and spectral_measure.final_verdict == "RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY"
    )

    if checks_pass:
        final_verdict = "RADIAL_RESPONSE_THEORY_PASS"
        interpretation = (
            "Masses are organized as boundary recirculation residues/pole "
            "shifts. The up-sector factorial relation is a stacking-law "
            "statement, not a derivation of x; literal exp(xN) is killed as a "
            "Yukawa matrix; and the down rank-5 bottom residue is available as "
            "a dark-line complement but not forced by S3 alone. Two-channel "
            "repair isometry derives x=1/sqrt(2) only when the microscopic "
            "successor/no-leakage assumptions hold. The finite R7 certificate "
            "models those assumptions as exactly two Z2-conjugate successors, "
            "and R8 proves this pair is complete inside the S3 regular shell. "
            "R9 reduces the scalar boundary sector to that S3 shell under the "
            "vacuum-frame-preserving tetrahedral automorphism premise. R10 "
            "derives that premise from a declared one-tick scalar-local "
            "boundary-map class and rejects nonlocal, non-scalar, generic "
            "linear-mixture, frame-breaking, same-state, and Hermitian/Z2 "
            "controls. Full microscopic derivation of that declared class from "
            "the BB/QCA update remains open. The minimal unitary S3 defect form "
            "is valid but does not force phase or radial values. R11 makes "
            "radial_response inherit the already-derived silver transfer root "
            "from boundary_response/flavor_a_track: epsilon=sqrt(2)-1, eta="
            "epsilon^2, and r=epsilon^4. Thus the open radial burden is "
            "not transfer-root derivation. R12 proves a no-go for pole/residue "
            "rigidity from the current finite data: admissible triality-head "
            "baths with the same coupling norm can have different self-energies, "
            "poles, and residues. A further dynamical spectral principle is "
            "needed for actual mass values. R13 confirms that the target up "
            "and down textures define positive finite spectral measures and "
            "unique Jacobi baths, but the existing simple forward baths do not "
            "select those measures. This is reconstruction-only, not a mass "
            "derivation."
        )
    else:
        final_verdict = "RADIAL_RESPONSE_THEORY_KILL"
        interpretation = "One or more radial-response gates failed."

    return RadialResponseAuditPayload(
        final_verdict=final_verdict,
        green_function=green,
        up_stacking=up,
        literal_nilpotent_yukawa=literal,
        down_dark_line=down,
        two_channel_isometry=two_channel,
        unitary_defect=unitary,
        successor_certificate=successor,
        s3_shell_completeness=s3_shell,
        qca_s3_reduction=qca_reduction,
        scalar_automorphism_premise=scalar_automorphism,
        silver_transfer_inheritance=silver_transfer,
        pole_residue_rigidity=pole_residue,
        spectral_measure_selection=spectral_measure,
        interpretation=interpretation,
    )
