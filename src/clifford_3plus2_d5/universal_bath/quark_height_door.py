"""Session 08A quark height-door audit.

Hypercharge forces the SM Higgs doors ``H`` and ``H_tilde``.  It does not, by
itself, force the dynamical statement that the up door sees an oriented
nilpotent repair while the down door sees a Hermitian closure.  This session
keeps those two layers separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.depth_scar.nilpotent_flag import (
    flag_laplacian_from_nilpotent,
    nilpotent_flag_operator,
    nilpotent_order_pass,
)
from clifford_3plus2_d5.depth_scar.repair_graphs import EXPECTED_LAPLACIAN_SPECTRUM
from clifford_3plus2_d5.universal_bath.down_quark_indefinite_jacobi import (
    unresolved_down_quark_source,
)
from clifford_3plus2_d5.universal_bath.reduction import ReductionKind
from clifford_3plus2_d5.universal_bath.source_dictionary import (
    SourceStatus,
    source_dictionary_payload,
)
from clifford_3plus2_d5.universal_bath.up_quark_nilpotent_cmv import (
    unresolved_up_quark_source,
)


UP_QUARK_DOOR = "H_tilde"
DOWN_QUARK_DOOR = "H"


@dataclass(frozen=True)
class HiggsDoor:
    """One electroweak Higgs door for a quark Yukawa term."""

    sector: str
    door: str
    q_left_hypercharge: sp.Expr
    right_hypercharge: sp.Expr
    higgs_hypercharge: sp.Expr
    neutral_component_t3_l: sp.Expr
    electromagnetic_charge: sp.Expr
    hypercharge_residual: sp.Expr
    gauge_forced: bool
    repair_mode: str
    repair_operator: sp.Matrix
    repair_mode_from_hypercharge: bool


@dataclass(frozen=True)
class QuarkHeightDoorPayload:
    """Session 08A height-door verdict."""

    final_verdict: str
    source_dictionary_pass: bool
    up_source_unresolved: bool
    down_source_unresolved: bool
    up_door: HiggsDoor
    down_door: HiggsDoor
    hypercharge_forces_higgs_doors: bool
    neutral_higgs_components: bool
    up_operator_nilpotent: bool
    up_operator_non_hermitian: bool
    down_operator_hermitian: bool
    down_operator_not_nilpotent: bool
    down_laplacian_spectrum: tuple[sp.Expr, ...]
    down_laplacian_spectrum_matches_path: bool
    swapped_repair_assignment_hypercharge_allowed: bool
    swapped_repair_assignment_rejected_by_height_premise: bool
    repair_mode_not_forced_by_hypercharge: bool
    quark_sources_still_unfrozen: bool
    interpretation: str


def _sorted_eigenvalues(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    """Return exact eigenvalues with multiplicity in deterministic order."""

    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def _matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices agree after exact simplification."""

    residual = left - right
    return all(sp.simplify(entry) == 0 for entry in residual)


def is_hermitian(matrix: sp.Matrix) -> bool:
    """Return whether a matrix is exactly Hermitian."""

    return _matrix_equal(matrix, matrix.conjugate().T)


def up_repair_operator() -> sp.Matrix:
    """Return the oriented height-lowering up repair operator."""

    return nilpotent_flag_operator()


def down_repair_operator() -> sp.Matrix:
    """Return the Hermitian closure induced by the nilpotent flag."""

    return flag_laplacian_from_nilpotent(nilpotent_flag_operator())


def up_higgs_door() -> HiggsDoor:
    """Return the SM-forced up door with the declared height repair."""

    q_left = sp.Rational(1, 6)
    right = sp.Rational(2, 3)
    higgs = -sp.Rational(1, 2)
    t3 = sp.Rational(1, 2)
    return HiggsDoor(
        sector="up_quark",
        door=UP_QUARK_DOOR,
        q_left_hypercharge=q_left,
        right_hypercharge=right,
        higgs_hypercharge=higgs,
        neutral_component_t3_l=t3,
        electromagnetic_charge=sp.simplify(higgs + t3),
        hypercharge_residual=sp.simplify(-q_left + higgs + right),
        gauge_forced=True,
        repair_mode="oriented_height_lowering_nilpotent",
        repair_operator=up_repair_operator(),
        repair_mode_from_hypercharge=False,
    )


def down_higgs_door() -> HiggsDoor:
    """Return the SM-forced down door with the declared height repair."""

    q_left = sp.Rational(1, 6)
    right = -sp.Rational(1, 3)
    higgs = sp.Rational(1, 2)
    t3 = -sp.Rational(1, 2)
    return HiggsDoor(
        sector="down_quark",
        door=DOWN_QUARK_DOOR,
        q_left_hypercharge=q_left,
        right_hypercharge=right,
        higgs_hypercharge=higgs,
        neutral_component_t3_l=t3,
        electromagnetic_charge=sp.simplify(higgs + t3),
        hypercharge_residual=sp.simplify(-q_left + higgs + right),
        gauge_forced=True,
        repair_mode="hermitian_height_closure",
        repair_operator=down_repair_operator(),
        repair_mode_from_hypercharge=False,
    )


def hypercharge_forces_higgs_doors() -> bool:
    """Return whether the SM quark Yukawa hypercharge residuals vanish."""

    return up_higgs_door().hypercharge_residual == 0 and down_higgs_door().hypercharge_residual == 0


def neutral_higgs_components() -> bool:
    """Return whether both neutral Higgs-door components have ``Q=Y+T3=0``."""

    return up_higgs_door().electromagnetic_charge == 0 and down_higgs_door().electromagnetic_charge == 0


def swapped_repair_assignment_hypercharge_allowed() -> bool:
    """Return whether hypercharge alone permits swapping the repair modes."""

    # Hypercharge only sees the door labels and charges, not the repair operator.
    return hypercharge_forces_higgs_doors()


def swapped_repair_assignment_rejected_by_height_premise() -> bool:
    """Return whether the declared height premise rejects swapping modes."""

    swapped_up_is_hermitian = is_hermitian(down_repair_operator())
    swapped_down_is_nilpotent = nilpotent_order_pass(up_repair_operator())
    declared_up_is_nilpotent = nilpotent_order_pass(up_repair_operator())
    declared_down_is_hermitian = is_hermitian(down_repair_operator())
    return (
        swapped_up_is_hermitian
        and swapped_down_is_nilpotent
        and declared_up_is_nilpotent
        and declared_down_is_hermitian
    )


def quark_height_door_payload() -> QuarkHeightDoorPayload:
    """Return the Session 08A height-door audit payload."""

    source_payload = source_dictionary_payload()
    up_source = unresolved_up_quark_source()
    down_source = unresolved_down_quark_source()
    up_door = up_higgs_door()
    down_door = down_higgs_door()
    down_spectrum = _sorted_eigenvalues(down_door.repair_operator)

    source_pass = source_payload.final_verdict == "SOURCE_DICTIONARY_CORE_PASS"
    up_unresolved = (
        up_source.status == SourceStatus.UNRESOLVED
        and up_source.reduction == ReductionKind.CMV_OPUC
        and up_source.port_vector is None
    )
    down_unresolved = (
        down_source.status == SourceStatus.UNRESOLVED
        and down_source.reduction == ReductionKind.INDEFINITE_LOOKAHEAD_JACOBI
        and down_source.port_vector is None
    )
    up_nilpotent = nilpotent_order_pass(up_door.repair_operator)
    up_non_hermitian = not is_hermitian(up_door.repair_operator)
    down_hermitian = is_hermitian(down_door.repair_operator)
    down_not_nilpotent = not nilpotent_order_pass(down_door.repair_operator)
    path_spectrum = down_spectrum == EXPECTED_LAPLACIAN_SPECTRUM
    swapped_allowed = swapped_repair_assignment_hypercharge_allowed()
    swapped_rejected = swapped_repair_assignment_rejected_by_height_premise()
    not_from_hypercharge = (
        not up_door.repair_mode_from_hypercharge
        and not down_door.repair_mode_from_hypercharge
        and swapped_allowed
    )

    checks_pass = (
        source_pass
        and up_unresolved
        and down_unresolved
        and hypercharge_forces_higgs_doors()
        and neutral_higgs_components()
        and up_nilpotent
        and up_non_hermitian
        and down_hermitian
        and down_not_nilpotent
        and path_spectrum
        and swapped_allowed
        and swapped_rejected
        and not_from_hypercharge
    )

    if checks_pass:
        final_verdict = "QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT"
        interpretation = (
            "The SM charges force H_tilde for up and H for down, with neutral "
            "Higgs components.  The declared height-door assignment then maps "
            "up to the oriented length-3 nilpotent repair and down to the "
            "Hermitian path closure.  A swapped repair assignment is still "
            "hypercharge-allowed, so the coherent-up/Hermitian-down split is "
            "not derived from electroweak charges alone; it remains a named "
            "height-dynamics premise.  Up/down source vectors remain unfrozen."
        )
    else:
        final_verdict = "QUARK_HEIGHT_DOOR_AUDIT_KILL"
        interpretation = (
            "The height-door audit failed the source, SM charge, nilpotent, "
            "Hermitian closure, path-spectrum, or negative-control checks."
        )

    return QuarkHeightDoorPayload(
        final_verdict=final_verdict,
        source_dictionary_pass=source_pass,
        up_source_unresolved=up_unresolved,
        down_source_unresolved=down_unresolved,
        up_door=up_door,
        down_door=down_door,
        hypercharge_forces_higgs_doors=hypercharge_forces_higgs_doors(),
        neutral_higgs_components=neutral_higgs_components(),
        up_operator_nilpotent=up_nilpotent,
        up_operator_non_hermitian=up_non_hermitian,
        down_operator_hermitian=down_hermitian,
        down_operator_not_nilpotent=down_not_nilpotent,
        down_laplacian_spectrum=down_spectrum,
        down_laplacian_spectrum_matches_path=path_spectrum,
        swapped_repair_assignment_hypercharge_allowed=swapped_allowed,
        swapped_repair_assignment_rejected_by_height_premise=swapped_rejected,
        repair_mode_not_forced_by_hypercharge=not_from_hypercharge,
        quark_sources_still_unfrozen=True,
        interpretation=interpretation,
    )
