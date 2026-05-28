"""V13 quark color/BCC Clebsch gate.

V11 derives the primitive quark shell and V12 derives the raw transfer-depth
hierarchy.  V13 audits only the remaining Q3 prefactors:

    C_F = 4/3,
    symmetric BCC factor = sqrt(2),
    antisymmetric BCC factor = 1/sqrt(2),
    normalized Cabibbo leakage = epsilon^2 / sqrt(1 + epsilon^4).

CKM assembly remains parked for V14.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_transfer_hierarchy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def su3_fundamental_generators(*, normalized: bool = True) -> tuple[sp.Matrix, ...]:
    """Return exact SU(3) fundamental generators.

    By default this returns ``T^A = lambda^A / 2``.  Setting
    ``normalized=False`` returns the raw Gell-Mann matrices as a negative
    control.
    """

    zero = sp.Integer(0)
    one = sp.Integer(1)
    lambda_matrices = (
        sp.Matrix([[zero, one, zero], [one, zero, zero], [zero, zero, zero]]),
        sp.Matrix([[zero, -sp.I, zero], [sp.I, zero, zero], [zero, zero, zero]]),
        sp.Matrix([[one, zero, zero], [zero, -one, zero], [zero, zero, zero]]),
        sp.Matrix([[zero, zero, one], [zero, zero, zero], [one, zero, zero]]),
        sp.Matrix([[zero, zero, -sp.I], [zero, zero, zero], [sp.I, zero, zero]]),
        sp.Matrix([[zero, zero, zero], [zero, zero, one], [zero, one, zero]]),
        sp.Matrix([[zero, zero, zero], [zero, zero, -sp.I], [zero, sp.I, zero]]),
        sp.Matrix(
            [
                [1 / sp.sqrt(3), zero, zero],
                [zero, 1 / sp.sqrt(3), zero],
                [zero, zero, -2 / sp.sqrt(3)],
            ]
        ),
    )
    if normalized:
        return tuple(matrix / 2 for matrix in lambda_matrices)
    return lambda_matrices


def color_return_contraction(
    generators: tuple[sp.Matrix, ...] | None = None,
) -> sp.Matrix:
    """Return ``sum_A T^A T^A`` for the supplied color generators."""

    selected = su3_fundamental_generators() if generators is None else generators
    return sum((generator * generator for generator in selected), sp.zeros(3, 3))


def color_return_factor() -> sp.Expr:
    """Return the fundamental color-singlet return factor ``C_F``."""

    return sp.Rational(4, 3)


def normalized_two_step_leakage() -> sp.Expr:
    """Return the normalized two-step boundary leakage scalar."""

    return sp.simplify(epsilon() ** 2 / sp.sqrt(1 + epsilon() ** 4))


def bcc_path_basis() -> tuple[sp.Matrix, sp.Matrix]:
    """Return the two orthonormal BCC path basis vectors ``(p1, p2)``."""

    return sp.Matrix([1, 0]), sp.Matrix([0, 1])


def bcc_symmetric_vector() -> sp.Matrix:
    """Return the normalized symmetric two-path vector."""

    p1, p2 = bcc_path_basis()
    return sp.simplify((p1 + p2) / sp.sqrt(2))


def bcc_antisymmetric_vector(*, normalized: bool = True) -> sp.Matrix:
    """Return the antisymmetric two-path vector."""

    p1, p2 = bcc_path_basis()
    vector = p1 - p2
    if normalized:
        return sp.simplify(vector / sp.sqrt(2))
    return vector


def bcc_symmetric_two_path_factor(*, coherent: bool = True) -> sp.Expr:
    """Return the symmetric two-path enhancement factor."""

    p1, p2 = bcc_path_basis()
    symmetric = bcc_symmetric_vector()
    leakage = p1 + p2 if coherent else p1
    return sp.simplify((symmetric.T * leakage)[0, 0])


def bcc_antisymmetric_projection_factor(*, normalized: bool = True) -> sp.Expr:
    """Return the projection of one external component onto the antisymmetric channel."""

    p1, _ = bcc_path_basis()
    antisymmetric = bcc_antisymmetric_vector(normalized=normalized)
    return sp.simplify((p1.T * antisymmetric)[0, 0])


def missing_color_generator_control() -> tuple[sp.Matrix, ...]:
    """Return a bad color-generator set with one generator removed."""

    return su3_fundamental_generators()[:-1]


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


@dataclass(frozen=True)
class QuarkClebschSummary:
    """Exact Q3 prefactor summary for later CKM assembly."""

    color_return_factor: sp.Expr
    normalized_cabibbo_leakage: sp.Expr
    symmetric_bcc_factor: sp.Expr
    antisymmetric_bcc_factor: sp.Expr


def quark_prefactor_summary() -> QuarkClebschSummary:
    """Return the exact Q3 prefactors."""

    return QuarkClebschSummary(
        color_return_factor=color_return_factor(),
        normalized_cabibbo_leakage=normalized_two_step_leakage(),
        symmetric_bcc_factor=bcc_symmetric_two_path_factor(),
        antisymmetric_bcc_factor=bcc_antisymmetric_projection_factor(),
    )


@dataclass(frozen=True)
class QuarkClebschAuditPayload:
    """Verdict payload for the V13 quark Clebsch gate."""

    final_verdict: str
    prefactors: QuarkClebschSummary
    color_contraction_matches: bool
    raw_generator_control_rejected: bool
    missing_generator_control_rejected: bool
    incoherent_bcc_control_rejected: bool
    unnormalized_antisymmetric_control_rejected: bool
    ckm_parked: bool
    interpretation: str


def quark_clebsch_audit_payload() -> QuarkClebschAuditPayload:
    """Return the V13 quark color/BCC Clebsch verdict."""

    v12 = quark_transfer_hierarchy_audit_payload()
    prefactors = quark_prefactor_summary()
    target_color = color_return_factor() * sp.eye(3)

    color_contraction_matches = _matrix_equal(color_return_contraction(), target_color)
    raw_generator_control_rejected = not _matrix_equal(
        color_return_contraction(su3_fundamental_generators(normalized=False)),
        target_color,
    )
    missing_generator_control_rejected = not _matrix_equal(
        color_return_contraction(missing_color_generator_control()),
        target_color,
    )
    incoherent_bcc_control_rejected = (
        sp.simplify(bcc_symmetric_two_path_factor(coherent=False) - sp.sqrt(2)) != 0
    )
    unnormalized_antisymmetric_control_rejected = (
        sp.simplify(bcc_antisymmetric_projection_factor(normalized=False) - 1 / sp.sqrt(2)) != 0
    )

    checks_pass = (
        v12.final_verdict == "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
        and color_contraction_matches
        and prefactors.color_return_factor == sp.Rational(4, 3)
        and sp.simplify(
            prefactors.normalized_cabibbo_leakage
            - epsilon() ** 2 / sp.sqrt(1 + epsilon() ** 4)
        )
        == 0
        and prefactors.symmetric_bcc_factor == sp.sqrt(2)
        and prefactors.antisymmetric_bcc_factor == 1 / sp.sqrt(2)
        and raw_generator_control_rejected
        and missing_generator_control_rejected
        and incoherent_bcc_control_rejected
        and unnormalized_antisymmetric_control_rejected
    )

    if checks_pass:
        final_verdict = "QUARK_CLEBSCH_Q3_PASS"
        interpretation = (
            "The exact SU(3) fundamental contraction gives C_F = 4/3, the "
            "normalized Cabibbo leakage is epsilon^2/sqrt(1+epsilon^4), and "
            "the BCC two-path channels give sqrt(2) and 1/sqrt(2). Raw-color, "
            "missing-generator, incoherent-path, and unnormalized-path controls "
            "are rejected. CKM assembly remains parked."
        )
    else:
        final_verdict = "QUARK_CLEBSCH_Q3_KILL"
        interpretation = (
            "The Q3 color contraction, BCC Clebsches, V12 prerequisite, or a "
            "negative control failed. CKM assembly remains parked."
        )

    return QuarkClebschAuditPayload(
        final_verdict=final_verdict,
        prefactors=prefactors,
        color_contraction_matches=color_contraction_matches,
        raw_generator_control_rejected=raw_generator_control_rejected,
        missing_generator_control_rejected=missing_generator_control_rejected,
        incoherent_bcc_control_rejected=incoherent_bcc_control_rejected,
        unnormalized_antisymmetric_control_rejected=unnormalized_antisymmetric_control_rejected,
        ckm_parked=True,
        interpretation=interpretation,
    )
