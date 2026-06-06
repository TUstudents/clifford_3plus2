"""Tests for Session 08A quark height-door audit."""

import sympy as sp

from clifford_3plus2_d5.depth_scar.repair_graphs import EXPECTED_LAPLACIAN_SPECTRUM
from clifford_3plus2_d5.universal_bath.quark_height_door import (
    DOWN_QUARK_DOOR,
    UP_QUARK_DOOR,
    down_higgs_door,
    down_repair_operator,
    hypercharge_forces_higgs_doors,
    is_hermitian,
    neutral_higgs_components,
    quark_height_door_payload,
    swapped_repair_assignment_hypercharge_allowed,
    swapped_repair_assignment_rejected_by_height_premise,
    up_higgs_door,
    up_repair_operator,
)


def _spectrum_tuple(matrix: sp.Matrix) -> tuple[sp.Expr, ...]:
    values: list[sp.Expr] = []
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        values.extend([sp.simplify(eigenvalue)] * multiplicity)
    return tuple(sorted(values, key=sp.default_sort_key))


def test_hypercharge_forces_h_and_h_tilde_doors() -> None:
    up = up_higgs_door()
    down = down_higgs_door()

    assert up.door == UP_QUARK_DOOR
    assert down.door == DOWN_QUARK_DOOR
    assert up.hypercharge_residual == 0
    assert down.hypercharge_residual == 0
    assert up.electromagnetic_charge == 0
    assert down.electromagnetic_charge == 0
    assert hypercharge_forces_higgs_doors()
    assert neutral_higgs_components()


def test_up_door_operator_is_oriented_nilpotent_not_hermitian() -> None:
    operator = up_repair_operator()

    assert operator**3 == sp.zeros(3, 3)
    assert operator**2 != sp.zeros(3, 3)
    assert not is_hermitian(operator)


def test_down_door_operator_is_hermitian_path_closure_not_nilpotent() -> None:
    operator = down_repair_operator()

    assert is_hermitian(operator)
    assert operator**2 != sp.zeros(3, 3)
    assert _spectrum_tuple(operator) == EXPECTED_LAPLACIAN_SPECTRUM


def test_hypercharge_does_not_select_repair_mode_by_itself() -> None:
    assert swapped_repair_assignment_hypercharge_allowed()
    assert swapped_repair_assignment_rejected_by_height_premise()


def test_quark_height_door_payload_reports_conditional_pass() -> None:
    payload = quark_height_door_payload()

    assert payload.final_verdict == "QUARK_HEIGHT_DOOR_AUDIT_CONDITIONAL_PASS"
    assert payload.source_dictionary_pass
    assert payload.up_source_unresolved
    assert payload.down_source_unresolved
    assert payload.hypercharge_forces_higgs_doors
    assert payload.neutral_higgs_components
    assert payload.up_operator_nilpotent
    assert payload.up_operator_non_hermitian
    assert payload.down_operator_hermitian
    assert payload.down_operator_not_nilpotent
    assert payload.down_laplacian_spectrum_matches_path
    assert payload.swapped_repair_assignment_hypercharge_allowed
    assert payload.swapped_repair_assignment_rejected_by_height_premise
    assert payload.repair_mode_not_forced_by_hypercharge
    assert payload.quark_sources_still_unfrozen
