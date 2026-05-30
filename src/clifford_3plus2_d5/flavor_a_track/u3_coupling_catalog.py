"""U3 — the per-sector couplings ``V_f`` are SM quantum-number projections.

Universality places the flavor difference between sectors entirely in the
coupling ``V_f``: which SM quantum numbers (color rep x weak rep x hypercharge)
connect the sector to the boundary.  This gate catalogs the six one-generation
fields and verifies that each field's chiral-16 multiplicity factors *exactly*
as ``color x weak``, with leptons color-singlet and quarks color-triplet — so
``V_f`` is the quantum-number projection, and the lepton/quark difference is the
color label (matching U2).

The explicit 32x32 projector construction on lepton's chiral-16 carrier (via
``joint_charge_decomposition`` + ``su3_c`` generators) is available but heavy and
is deferred to the full flavor program (roadmap A3); this gate works at the
quantum-number-multiplicity level, which is sufficient to confirm the structural
claim or to kill it.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.flavor_a_track.reuse import sm_field_multiplicity_table

# Field -> (color rep dim, weak rep dim, hypercharge Y). The chiral-16 complex
# multiplicity must equal color * weak.
SECTOR_QUANTUM_NUMBERS: dict[str, dict[str, object]] = {
    "Q": {"color": 3, "weak": 2, "Y": sp.Rational(1, 6)},
    "u^c": {"color": 3, "weak": 1, "Y": sp.Rational(-2, 3)},
    "d^c": {"color": 3, "weak": 1, "Y": sp.Rational(1, 3)},
    "L": {"color": 1, "weak": 2, "Y": sp.Rational(-1, 2)},
    "e^c": {"color": 1, "weak": 1, "Y": sp.Integer(1)},
    "nu^c": {"color": 1, "weak": 1, "Y": sp.Integer(0)},
}

QUARK_FIELDS = ("Q", "u^c", "d^c")
LEPTON_FIELDS = ("L", "e^c", "nu^c")


def field_multiplicity_table() -> dict[str, dict[str, object]]:
    """Return lepton's one-generation field multiplicity table (Y, multiplicity)."""

    return sm_field_multiplicity_table()


def multiplicities_factor_as_color_times_weak() -> bool:
    """Return true when every field's chiral-16 multiplicity is ``color * weak``."""

    table = field_multiplicity_table()
    for field, qn in SECTOR_QUANTUM_NUMBERS.items():
        expected = int(qn["color"]) * int(qn["weak"])
        actual = int(table[field]["complex_multiplicity"])
        if expected != actual:
            return False
    return True


def hypercharges_match_table() -> bool:
    """Return true when the catalog hypercharges match lepton's field table."""

    table = field_multiplicity_table()
    return all(
        sp.simplify(qn["Y"] - table[field]["Y"]) == 0
        for field, qn in SECTOR_QUANTUM_NUMBERS.items()
    )


def color_split_is_quark_triplet_lepton_singlet() -> bool:
    """Return true when quark fields are color-triplet and lepton fields color-singlet."""

    quarks_triplet = all(int(SECTOR_QUANTUM_NUMBERS[f]["color"]) == 3 for f in QUARK_FIELDS)
    leptons_singlet = all(int(SECTOR_QUANTUM_NUMBERS[f]["color"]) == 1 for f in LEPTON_FIELDS)
    return quarks_triplet and leptons_singlet


def coupling_catalog() -> dict[str, dict[str, object]]:
    """Return the per-sector ``V_f`` quantum-number catalog."""

    table = field_multiplicity_table()
    return {
        field: {
            "color": qn["color"],
            "weak": qn["weak"],
            "Y": qn["Y"],
            "chiral16_multiplicity": table[field]["complex_multiplicity"],
        }
        for field, qn in SECTOR_QUANTUM_NUMBERS.items()
    }


def coupling_catalog_verdict(
    multiplicities_factor: bool, hypercharges_match: bool, color_split: bool
) -> str:
    """Return the U3 verdict from the three checks. Pure decision (KILL-testable)."""

    if multiplicities_factor and hypercharges_match and color_split:
        return "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"
    return "COUPLINGS_NOT_QUANTUM_NUMBER_DETERMINED_KILL"


@dataclass(frozen=True)
class CouplingCatalogAuditPayload:
    """Verdict payload for the U3 coupling-catalog gate."""

    final_verdict: str
    coupling_catalog: dict[str, dict[str, object]]
    multiplicities_factor: bool
    hypercharges_match: bool
    color_split_correct: bool
    explicit_chiral16_projectors_deferred: bool
    interpretation: str


def coupling_catalog_audit_payload() -> CouplingCatalogAuditPayload:
    """Return the U3 coupling-catalog verdict."""

    factors = multiplicities_factor_as_color_times_weak()
    y_match = hypercharges_match_table()
    color_split = color_split_is_quark_triplet_lepton_singlet()

    final_verdict = coupling_catalog_verdict(factors, y_match, color_split)
    checks_pass = final_verdict == "COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS"

    if checks_pass:
        interpretation = (
            "Each one-generation field's chiral-16 multiplicity factors exactly "
            "as color x weak (Q=3x2, u^c=3x1, d^c=3x1, L=1x2, e^c=1x1, "
            "nu^c=1x1), with the catalog hypercharges matching lepton's field "
            "table. Quark fields are color-triplet, lepton fields color-singlet, "
            "so V_f is the SM quantum-number projection and the lepton/quark "
            "difference is the color label (consistent with U2). The explicit "
            "32x32 chiral-16 projector construction is deferred to A3."
        )
    else:
        interpretation = (
            "The field multiplicities do not factor as color x weak, the "
            "hypercharges do not match, or the quark-triplet/lepton-singlet "
            "color split fails. The couplings are not pure quantum-number "
            "projections; universality is not supported."
        )

    return CouplingCatalogAuditPayload(
        final_verdict=final_verdict,
        coupling_catalog=coupling_catalog(),
        multiplicities_factor=factors,
        hypercharges_match=y_match,
        color_split_correct=color_split,
        explicit_chiral16_projectors_deferred=checks_pass,
        interpretation=interpretation,
    )
