"""A3-3 — the sector-specific structure lives in the coupling V_f, not in H_Q.

With the transfer boundary unified (A3-1, A3-2 use one common sterile chain), the
remaining quark structure must be carried by the coupling ``V_quark``: the
color-triplet SU(3) return ``C_F = sum_A T^A T^A = (4/3) I``, the BCC path
Clebsches ``sqrt(2)`` and ``1/sqrt(2)``, and the flat Cl_5 coin phase
``atan(sqrt(5))``.  The lepton coupling ``V_lepton`` is color-singlet (no C_F, no
color ports).  So the lepton/quark difference is entirely in ``V_f`` — exactly
the universality claim — while ``H_Q`` is common.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.unified_boundary.reuse import (
    bcc_antisymmetric_projection_factor,
    bcc_symmetric_two_path_factor,
    color_return_contraction,
    color_return_factor,
    color_split_is_quark_triplet_lepton_singlet,
    quark_boundary_phase_angle,
)


def color_return_is_casimir_identity() -> bool:
    """Return true when sum_A T^A T^A = (4/3) I_3 (color-triplet Casimir)."""

    contraction = color_return_contraction()
    expected = sp.Rational(4, 3) * sp.eye(3)
    return all(sp.simplify(entry) == 0 for entry in contraction - expected)


def color_factor_is_four_thirds() -> bool:
    """Return true when C_F = 4/3."""

    return sp.simplify(color_return_factor() - sp.Rational(4, 3)) == 0


def bcc_factors_are_sqrt2_and_inverse() -> bool:
    """Return true when the BCC path Clebsches are sqrt(2) and 1/sqrt(2)."""

    symmetric_ok = sp.simplify(bcc_symmetric_two_path_factor() - sp.sqrt(2)) == 0
    antisymmetric_ok = sp.simplify(bcc_antisymmetric_projection_factor() - 1 / sp.sqrt(2)) == 0
    return symmetric_ok and antisymmetric_ok


def coin_phase_is_atan_sqrt5() -> bool:
    """Return true when the flat Cl_5 coin phase is atan(sqrt(5))."""

    return sp.simplify(quark_boundary_phase_angle() - sp.atan(sp.sqrt(5))) == 0


def lepton_is_color_singlet_quark_triplet() -> bool:
    """Return true when leptons are color-singlet and quarks color-triplet (U3)."""

    return color_split_is_quark_triplet_lepton_singlet()


@dataclass(frozen=True)
class CouplingStructureAuditPayload:
    """Verdict payload for the A3-3 coupling-structure gate."""

    final_verdict: str
    color_return_is_casimir: bool
    color_factor_value: sp.Expr
    bcc_symmetric_factor: sp.Expr
    bcc_antisymmetric_factor: sp.Expr
    coin_phase: sp.Expr
    color_factor_is_four_thirds: bool
    bcc_factors_correct: bool
    coin_phase_correct: bool
    lepton_singlet_quark_triplet: bool
    interpretation: str


def coupling_structure_audit_payload() -> CouplingStructureAuditPayload:
    """Return the A3-3 verdict."""

    casimir_ok = color_return_is_casimir_identity()
    cf_ok = color_factor_is_four_thirds()
    bcc_ok = bcc_factors_are_sqrt2_and_inverse()
    coin_ok = coin_phase_is_atan_sqrt5()
    split_ok = lepton_is_color_singlet_quark_triplet()

    checks_pass = casimir_ok and cf_ok and bcc_ok and coin_ok and split_ok

    if checks_pass:
        final_verdict = "SECTOR_STRUCTURE_IN_COUPLINGS"
        interpretation = (
            "The quark-specific structure is V_quark quantum-number data: the "
            "color-triplet SU(3) return sum_A T^A T^A = (4/3) I (C_F = 4/3), the "
            "BCC path Clebsches sqrt(2) and 1/sqrt(2), and the flat Cl_5 coin "
            "phase atan(sqrt(5)). The lepton coupling is color-singlet. With H_Q "
            "common (A3-1/A3-2), the lepton/quark difference lives entirely in "
            "V_f. Deriving these factors from the chiral-16 geometry is deferred "
            "to A3b."
        )
    else:
        final_verdict = "SECTOR_STRUCTURE_IN_COUPLINGS_KILL"
        interpretation = (
            "The color return, BCC Clebsches, coin phase, or lepton-singlet / "
            "quark-triplet split did not check out. The sector structure is not "
            "cleanly carried by V_f."
        )

    return CouplingStructureAuditPayload(
        final_verdict=final_verdict,
        color_return_is_casimir=casimir_ok,
        color_factor_value=color_return_factor(),
        bcc_symmetric_factor=bcc_symmetric_two_path_factor(),
        bcc_antisymmetric_factor=bcc_antisymmetric_projection_factor(),
        coin_phase=quark_boundary_phase_angle(),
        color_factor_is_four_thirds=cf_ok,
        bcc_factors_correct=bcc_ok,
        coin_phase_correct=coin_ok,
        lepton_singlet_quark_triplet=split_ok,
        interpretation=interpretation,
    )
