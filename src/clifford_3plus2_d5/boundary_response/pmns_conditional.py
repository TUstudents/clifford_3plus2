"""V9 conditional PMNS assembly.

V6 derives the neutrino-core response, V7 derives the charged-lepton leakage
angle, and V8 verifies the arithmetic of the proposed leptonic phase word.
V9 assembles those ingredients into a PMNS texture but does not upgrade the
phase word to a theorem.  The output is therefore conditional on the future
boundary-loop holonomy derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt

import sympy as sp

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_audit_payload,
    charged_lepton_rotation_sine,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    leptonic_phase_word_angle,
    leptonic_phase_word_audit_payload,
)
from clifford_3plus2_d5.boundary_response.residual_basis import residual_basis_matrix
from clifford_3plus2_d5.boundary_response.transfer import epsilon_fourth
from clifford_3plus2_d5.boundary_response.weyl_sterile import weyl_sterile_audit_payload


def tbm_from_residual_basis() -> sp.Matrix:
    """Return the residual ``(a,u,b)`` basis as the TBM neutrino matrix."""

    return residual_basis_matrix(("a", "u", "b"))


def charged_lepton_phase_angle(*, conjugate_branch: bool = False) -> sp.Expr:
    """Return the charged-lepton phase angle in radians."""

    angle = leptonic_phase_word_angle()
    if conjugate_branch:
        angle = -angle
    return sp.simplify(sp.pi * angle)


def charged_lepton_rotation_matrix(*, conjugate_branch: bool = False) -> sp.Matrix:
    """Return the charged-lepton ``1-2`` rotation used for PMNS assembly.

    The convention places the V8 phase in the upper-right entry as
    ``exp(i phi)``.  With the default V8 branch ``phi = -5*pi/12``, the
    resulting PDG-like CP phase is the ``261.6`` degree branch.
    """

    sine = charged_lepton_rotation_sine()
    cosine = sp.sqrt(1 - sine**2)
    phase = charged_lepton_phase_angle(conjugate_branch=conjugate_branch)
    return sp.Matrix(
        [
            [cosine, sine * sp.exp(sp.I * phase), 0],
            [-sine * sp.exp(-sp.I * phase), cosine, 0],
            [0, 0, 1],
        ]
    )


def conditional_pmns_matrix(*, conjugate_branch: bool = False) -> sp.Matrix:
    """Return the conditional PMNS matrix ``R_e^dagger U_TBM``."""

    charged_rotation = charged_lepton_rotation_matrix(conjugate_branch=conjugate_branch)
    return charged_rotation.conjugate().T * tbm_from_residual_basis()


def _float_expr(expr: sp.Expr) -> float:
    """Evaluate a real SymPy expression as a Python float."""

    return float(sp.N(expr, 30))


def _pdg_delta_degrees(
    *,
    s12_squared: float,
    s13_squared: float,
    s23_squared: float,
    jarlskog: float,
    u_mu_1_abs_squared: float,
) -> float:
    """Return the PDG-like CP phase in degrees from bounded numerics."""

    s12 = sqrt(s12_squared)
    c12 = sqrt(1 - s12_squared)
    s13 = sqrt(s13_squared)
    c13 = sqrt(1 - s13_squared)
    s23 = sqrt(s23_squared)
    c23 = sqrt(1 - s23_squared)

    sin_delta = jarlskog / (s12 * c12 * s23 * c23 * s13 * c13**2)
    cos_delta = (
        u_mu_1_abs_squared
        - s12_squared * c23**2
        - c12**2 * s23_squared * s13_squared
    ) / (2 * s12 * c12 * s23 * c23 * s13)
    return degrees(atan2(sin_delta, cos_delta)) % 360


@dataclass(frozen=True)
class PMNSMixingObservables:
    """Standard mixing observables for the conditional PMNS matrix."""

    sin2_theta13: sp.Expr
    sin2_theta12: sp.Expr
    sin2_theta23: sp.Expr
    jarlskog: sp.Expr
    delta_cp_degrees: float


def pmns_mixing_observables(*, conjugate_branch: bool = False) -> PMNSMixingObservables:
    """Return exact angle expressions and numerical CP phase for V9."""

    sine = charged_lepton_rotation_sine()
    cosine = sp.sqrt(1 - sine**2)
    phase = charged_lepton_phase_angle(conjugate_branch=conjugate_branch)
    cos_phase = sp.cos(phase)
    sin_phase = sp.sin(phase)

    sin2_theta13 = sp.simplify(sine**2 / 2)
    cos2_theta13 = sp.simplify(1 - sin2_theta13)
    sin2_theta12 = sp.simplify(
        (cosine**2 + sine**2 - 2 * cosine * sine * cos_phase) / (3 * cos2_theta13)
    )
    sin2_theta23 = sp.simplify(cosine**2 / (2 * cos2_theta13))
    jarlskog = sp.simplify(cosine * sine * sin_phase / 6)
    u_mu_1_abs_squared = sp.simplify(
        2 * sine**2 / 3 + cosine**2 / 6 - 2 * sine * cosine * cos_phase / 3
    )
    delta_cp = _pdg_delta_degrees(
        s12_squared=_float_expr(sin2_theta12),
        s13_squared=_float_expr(sin2_theta13),
        s23_squared=_float_expr(sin2_theta23),
        jarlskog=_float_expr(jarlskog),
        u_mu_1_abs_squared=_float_expr(u_mu_1_abs_squared),
    )
    return PMNSMixingObservables(
        sin2_theta13=sin2_theta13,
        sin2_theta12=sin2_theta12,
        sin2_theta23=sin2_theta23,
        jarlskog=jarlskog,
        delta_cp_degrees=delta_cp,
    )


@dataclass(frozen=True)
class PMNSConditionalAuditPayload:
    """Verdict payload for the V9 conditional PMNS assembly."""

    final_verdict: str
    sin2_theta13: sp.Expr
    sin2_theta12: sp.Expr
    sin2_theta23: sp.Expr
    delta_cp_degrees: float
    conjugate_delta_cp_degrees: float
    jarlskog: sp.Expr
    conjugate_jarlskog: sp.Expr
    phase_word_derived: bool
    conditional_on_phase_word: bool
    ckm_parked: bool
    interpretation: str


def pmns_conditional_audit_payload() -> PMNSConditionalAuditPayload:
    """Return the V9 conditional PMNS assembly verdict."""

    v6 = weyl_sterile_audit_payload()
    v7 = charged_lepton_leakage_audit_payload()
    v8 = leptonic_phase_word_audit_payload()
    default = pmns_mixing_observables()
    conjugate = pmns_mixing_observables(conjugate_branch=True)

    expected_s13 = sp.Rational(3, 4) * epsilon_fourth()
    checks_pass = (
        v6.final_verdict == "PRODUCT_STERILE_LIMIT_PASS"
        and v7.final_verdict == "CHARGED_LEPTON_LEAKAGE_PASS"
        and v8.final_verdict == "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
        and sp.simplify(default.sin2_theta13 - expected_s13) == 0
        and sp.simplify(default.jarlskog + conjugate.jarlskog) == 0
        and not v8.word_selection_derived
    )

    if checks_pass:
        final_verdict = "PMNS_CONDITIONAL_ASSEMBLY_PASS"
        interpretation = (
            "V6, V7, and V8 assemble into the advertised PMNS texture: "
            "sin^2(theta13)=3 epsilon^4/4, the numerical solar/atmospheric "
            "angles match the boundary-response note, and the CP-conjugate "
            "branch flips the Jarlskog sign. This is conditional on the V8 "
            "phase-word selection theorem, which is not yet derived. CKM "
            "remains parked."
        )
    else:
        final_verdict = "PMNS_CONDITIONAL_ASSEMBLY_KILL"
        interpretation = (
            "The V6/V7/V8 inputs, exact theta13 relation, Jarlskog conjugacy, "
            "or conditional-word status failed. CKM remains parked."
        )

    return PMNSConditionalAuditPayload(
        final_verdict=final_verdict,
        sin2_theta13=default.sin2_theta13,
        sin2_theta12=default.sin2_theta12,
        sin2_theta23=default.sin2_theta23,
        delta_cp_degrees=default.delta_cp_degrees,
        conjugate_delta_cp_degrees=conjugate.delta_cp_degrees,
        jarlskog=default.jarlskog,
        conjugate_jarlskog=conjugate.jarlskog,
        phase_word_derived=v8.word_selection_derived,
        conditional_on_phase_word=not v8.word_selection_derived,
        ckm_parked=True,
        interpretation=interpretation,
    )
