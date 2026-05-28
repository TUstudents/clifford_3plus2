"""V14 conditional CKM assembly gate.

V11 derives the primitive quark shell and flat-coin phase, V12 derives the
raw transfer-depth hierarchy, and V13 derives the color/BCC Clebsch
prefactors.  V14 only assembles those already-gated ingredients into the
standard three-rotation CKM texture:

    V_CKM = R_23(theta_23) R_13(theta_13, delta_q) R_12(theta_12).

This is conditional on the Q1-Q3 boundary-shell model.  It is not a theorem
that the microscopic QCA forces that quark shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
    quark_boundary_phase_factor,
    quark_boundary_shell_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    quark_clebsch_audit_payload,
    quark_prefactor_summary,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_transfer_hierarchy_audit_payload,
    quark_transition_amplitude,
)


@dataclass(frozen=True)
class CKMSines:
    """Exact CKM sine inputs assembled from Q1-Q3."""

    s12: sp.Expr
    s23: sp.Expr
    s13: sp.Expr


def ckm_sines() -> CKMSines:
    """Return exact CKM sines from the quark boundary-shell gates."""

    prefactors = quark_prefactor_summary()
    return CKMSines(
        s12=sp.simplify(
            prefactors.color_return_factor * prefactors.normalized_cabibbo_leakage
        ),
        s23=sp.simplify(
            prefactors.symmetric_bcc_factor * quark_transition_amplitude(2, 3)
        ),
        s13=sp.simplify(
            prefactors.antisymmetric_bcc_factor * quark_transition_amplitude(1, 3)
        ),
    )


def ckm_phase_angle() -> sp.Expr:
    """Return the positive-branch quark CP phase angle."""

    return quark_boundary_phase_angle()


def _cosine_from_sine(sine: sp.Expr) -> sp.Expr:
    """Return the positive cosine associated with an exact sine."""

    return sp.sqrt(1 - sine**2)


def ckm_rotation_12() -> sp.Matrix:
    """Return the real ``1-2`` CKM rotation."""

    sines = ckm_sines()
    cosine = _cosine_from_sine(sines.s12)
    return sp.Matrix(
        [
            [cosine, sines.s12, 0],
            [-sines.s12, cosine, 0],
            [0, 0, 1],
        ]
    )


def ckm_rotation_23() -> sp.Matrix:
    """Return the real ``2-3`` CKM rotation."""

    sines = ckm_sines()
    cosine = _cosine_from_sine(sines.s23)
    return sp.Matrix(
        [
            [1, 0, 0],
            [0, cosine, sines.s23],
            [0, -sines.s23, cosine],
        ]
    )


def ckm_rotation_13() -> sp.Matrix:
    """Return the complex ``1-3`` CKM rotation with phase ``delta_q``."""

    sines = ckm_sines()
    cosine = _cosine_from_sine(sines.s13)
    phase_plus = quark_boundary_phase_factor()
    phase_minus = sp.conjugate(phase_plus)
    return sp.Matrix(
        [
            [cosine, 0, sines.s13 * phase_minus],
            [0, 1, 0],
            [-sines.s13 * phase_plus, 0, cosine],
        ]
    )


def conditional_ckm_matrix() -> sp.Matrix:
    """Return the conditional CKM matrix ``R23 R13 R12``."""

    return sp.simplify(ckm_rotation_23() * ckm_rotation_13() * ckm_rotation_12())


def ckm_magnitude_matrix() -> sp.Matrix:
    """Return the exact entrywise magnitude matrix of the conditional CKM."""

    matrix = conditional_ckm_matrix()
    return matrix.applyfunc(lambda entry: sp.sqrt(sp.simplify(entry * sp.conjugate(entry))))


def ckm_jarlskog() -> sp.Expr:
    """Return the CKM Jarlskog invariant for the chosen phase convention."""

    matrix = conditional_ckm_matrix()
    invariant = matrix[0, 0] * matrix[1, 1] * sp.conjugate(matrix[0, 1]) * sp.conjugate(
        matrix[1, 0]
    )
    return sp.simplify(sp.im(invariant))


@dataclass(frozen=True)
class CKMConditionalAuditPayload:
    """Verdict payload for the V14 conditional CKM assembly."""

    final_verdict: str
    sines: CKMSines
    phase_angle: sp.Expr
    phase_degrees: float
    magnitude_matrix: sp.Matrix
    jarlskog: sp.Expr
    prerequisites_pass: bool
    conditionally_assembled: bool
    interpretation: str


def ckm_conditional_audit_payload() -> CKMConditionalAuditPayload:
    """Return the V14 conditional CKM assembly verdict."""

    q1 = quark_boundary_shell_audit_payload()
    q2 = quark_transfer_hierarchy_audit_payload()
    q3 = quark_clebsch_audit_payload()
    sines = ckm_sines()
    matrix = conditional_ckm_matrix()
    residual = matrix.conjugate().T * matrix - sp.eye(3)

    prerequisites_pass = (
        q1.final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
        and q2.final_verdict == "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
        and q3.final_verdict == "QUARK_CLEBSCH_Q3_PASS"
    )
    unitary = max(abs(complex(sp.N(entry, 40))) for entry in residual) < 1e-30
    checks_pass = prerequisites_pass and unitary

    if checks_pass:
        final_verdict = "CKM_CONDITIONAL_ASSEMBLY_PASS"
        interpretation = (
            "Q1, Q2, and Q3 assemble into the advertised conditional CKM "
            "texture. The sines are the color/BCC-dressed transfer amplitudes, "
            "the phase is the flat primitive quark coin phase atan(sqrt(5)), "
            "and the resulting three-rotation matrix is unitary. This remains "
            "conditional on the quark primitive boundary-shell model."
        )
    else:
        final_verdict = "CKM_CONDITIONAL_ASSEMBLY_KILL"
        interpretation = (
            "A Q1/Q2/Q3 prerequisite or the assembled CKM unitarity check "
            "failed, so the conditional CKM texture is not accepted."
        )

    return CKMConditionalAuditPayload(
        final_verdict=final_verdict,
        sines=sines,
        phase_angle=ckm_phase_angle(),
        phase_degrees=degrees(float(sp.N(ckm_phase_angle(), 40))),
        magnitude_matrix=ckm_magnitude_matrix(),
        jarlskog=ckm_jarlskog(),
        prerequisites_pass=prerequisites_pass,
        conditionally_assembled=checks_pass,
        interpretation=interpretation,
    )
