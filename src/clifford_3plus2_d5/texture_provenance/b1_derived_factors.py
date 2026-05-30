"""B1 — derived-factor ledger.

Classifies each CKM/PMNS texture factor that is *derived* from the chiral-16
quantum numbers or BCC geometry, with a machine-checked source. These factors
give the texture STRUCTURE (the CP phases and the mixing-angle structure); the
hierarchy is the free input handled in B2.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.texture_provenance.reuse import (
    bcc_antisymmetric_projection_factor,
    bcc_symmetric_two_path_factor,
    color_return_contraction,
    color_return_factor,
    leptonic_boundary_holonomy_audit_payload,
    quark_boundary_phase_angle,
    quark_gamma_sum,
    quark_shell_dimension_breakdown,
    selected_port_residual_components,
)


@dataclass(frozen=True)
class DerivedFactor:
    """A texture factor derived from quantum numbers / geometry."""

    name: str
    value: sp.Expr
    source: str
    check_passes: bool


def color_return_check() -> bool:
    """C_F: sum_A T^A T^A == (4/3) I_3 (SU(3) fundamental Casimir)."""

    contraction = color_return_contraction()
    return all(sp.simplify(e) == 0 for e in contraction - sp.Rational(4, 3) * sp.eye(3))


def coin_dimension_check() -> bool:
    """Coin base sqrt(5): Gamma_q^2 == 5 I and 5 == 2_BCC + 3_color."""

    gamma = quark_gamma_sum()
    closure = all(sp.simplify(e) == 0 for e in gamma * gamma - 5 * sp.eye(gamma.rows))
    breakdown = quark_shell_dimension_breakdown()
    dimension = breakdown["bcc_odd"] + breakdown["color_odd"] == 5
    return closure and dimension


def bcc_clebsch_check() -> bool:
    """BCC path Clebsches: sqrt(2) (symmetric) and 1/sqrt(2) (antisymmetric)."""

    symmetric = sp.simplify(bcc_symmetric_two_path_factor() - sp.sqrt(2)) == 0
    antisymmetric = sp.simplify(bcc_antisymmetric_projection_factor() - 1 / sp.sqrt(2)) == 0
    return symmetric and antisymmetric


def charged_lepton_clebsch() -> sp.Expr:
    """Return the charged-lepton Clebsch sqrt(3/2) = 1 / <a|e1>."""

    a_component = selected_port_residual_components()["a"]
    return sp.simplify(1 / a_component)


def charged_lepton_clebsch_check() -> bool:
    """sqrt(3/2) from the residual port e1 = sqrt(2/3) a + 1/sqrt(3) u."""

    return sp.simplify(charged_lepton_clebsch() - sp.sqrt(sp.Rational(3, 2))) == 0


def leptonic_phase_word_check() -> bool:
    """Leptonic phase WORD selection is derived (V10 boundary-loop holonomy)."""

    return (
        leptonic_boundary_holonomy_audit_payload().final_verdict
        == "LEPTONIC_PHASE_WORD_DERIVED_PASS"
    )


def derived_factors() -> tuple[DerivedFactor, ...]:
    """Return the catalogue of derived texture factors."""

    return (
        DerivedFactor(
            name="color_return_C_F",
            value=color_return_factor(),
            source="SU(3) fundamental Casimir sum_A T^A T^A",
            check_passes=color_return_check(),
        ),
        DerivedFactor(
            name="coin_base_sqrt5",
            value=sp.sqrt(5),
            source="Gamma_q^2 = 5 I, 5 = 2_BCC + 3_color (odd-shell dimension)",
            check_passes=coin_dimension_check(),
        ),
        DerivedFactor(
            name="bcc_symmetric_sqrt2",
            value=bcc_symmetric_two_path_factor(),
            source="symmetric BCC two-path Clebsch",
            check_passes=bcc_clebsch_check(),
        ),
        DerivedFactor(
            name="bcc_antisymmetric_inv_sqrt2",
            value=bcc_antisymmetric_projection_factor(),
            source="antisymmetric BCC two-path projection",
            check_passes=bcc_clebsch_check(),
        ),
        DerivedFactor(
            name="charged_lepton_sqrt_3_2",
            value=charged_lepton_clebsch(),
            source="residual port e1 = sqrt(2/3) a + 1/sqrt(3) u",
            check_passes=charged_lepton_clebsch_check(),
        ),
        DerivedFactor(
            name="leptonic_phase_word",
            value=sp.pi * sp.Rational(-5, 12),
            source="V10 boundary-loop holonomy (LEPTONIC_PHASE_WORD_DERIVED_PASS)",
            check_passes=leptonic_phase_word_check(),
        ),
        DerivedFactor(
            name="quark_cp_phase_atan_sqrt5",
            value=quark_boundary_phase_angle(),
            source="flat Cl_5 coin positive branch (given r=1)",
            check_passes=sp.simplify(quark_boundary_phase_angle() - sp.atan(sp.sqrt(5))) == 0,
        ),
    )


@dataclass(frozen=True)
class DerivedFactorsAuditPayload:
    """Verdict payload for the B1 derived-factor ledger."""

    final_verdict: str
    derived_factors: tuple[DerivedFactor, ...]
    all_checks_pass: bool
    interpretation: str


def derived_factors_audit_payload() -> DerivedFactorsAuditPayload:
    """Return the B1 verdict."""

    factors = derived_factors()
    all_pass = all(factor.check_passes for factor in factors)

    if all_pass:
        final_verdict = "DERIVED_FACTORS_CATALOGUED"
        interpretation = (
            "Every group-theoretic / geometric texture factor checks out against "
            "its source: C_F = 4/3 (SU(3) Casimir), coin base sqrt(5) from "
            "Gamma_q^2 = 5I = (2_BCC + 3_color), BCC sqrt(2) and 1/sqrt(2), "
            "charged-lepton sqrt(3/2) from the residual port, and the leptonic "
            "phase word from V10 holonomy. These give the CKM/PMNS structure: the "
            "CP phases atan(sqrt(5)) and 5 pi / 12 and the PMNS angle structure."
        )
    else:
        final_verdict = "DERIVED_FACTORS_CATALOGUED_KILL"
        interpretation = (
            "A claimed-derived factor failed its source check; it is not "
            "actually derived from the quantum numbers / geometry."
        )

    return DerivedFactorsAuditPayload(
        final_verdict=final_verdict,
        derived_factors=factors,
        all_checks_pass=all_pass,
        interpretation=interpretation,
    )
