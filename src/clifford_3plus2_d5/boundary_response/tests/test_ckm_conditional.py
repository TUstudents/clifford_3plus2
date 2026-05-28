"""Tests for the V14 conditional CKM assembly."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.ckm_conditional import (
    ckm_conditional_audit_payload,
    ckm_jarlskog,
    ckm_magnitude_matrix,
    ckm_phase_angle,
    ckm_rotation_12,
    ckm_rotation_13,
    ckm_rotation_23,
    ckm_sines,
    conditional_ckm_matrix,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_shell_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    quark_clebsch_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_transfer_hierarchy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def _assert_unitary_numeric(matrix: sp.Matrix, *, tolerance: float = 1e-25) -> None:
    residual = matrix.conjugate().T * matrix - sp.eye(matrix.rows)
    numeric_residual = residual.applyfunc(lambda entry: complex(sp.N(entry, 40)))
    assert max(abs(entry) for entry in numeric_residual) < tolerance


def test_ckm_sines_match_q1_q2_q3_formulas() -> None:
    sines = ckm_sines()
    assert sp.simplify(sines.s12 - sp.Rational(4, 3) * epsilon() ** 2 / sp.sqrt(1 + epsilon() ** 4)) == 0
    assert sp.simplify(sines.s23 - sp.sqrt(2) * epsilon() ** 4) == 0
    assert sp.simplify(sines.s13 - epsilon() ** 6 / sp.sqrt(2)) == 0


def test_ckm_phase_is_flat_quark_coin_phase() -> None:
    assert ckm_phase_angle() == sp.atan(sp.sqrt(5))
    assert float(sp.N(ckm_phase_angle() * 180 / sp.pi, 20)) == pytest.approx(
        65.9051574478893,
        abs=1e-12,
    )


def test_ckm_rotations_and_assembled_matrix_are_unitary() -> None:
    _assert_unitary_numeric(ckm_rotation_12())
    _assert_unitary_numeric(ckm_rotation_13())
    _assert_unitary_numeric(ckm_rotation_23())
    _assert_unitary_numeric(conditional_ckm_matrix())


def test_ckm_magnitude_matrix_matches_note_values() -> None:
    magnitude = ckm_magnitude_matrix()
    expected = (
        (0.97425, 0.22547, 0.00357),
        (0.22533, 0.97340, 0.04163),
        (0.00858, 0.04090, 0.99913),
    )
    for row_index, row in enumerate(expected):
        for col_index, expected_value in enumerate(row):
            assert float(sp.N(magnitude[row_index, col_index], 20)) == pytest.approx(
                expected_value,
                abs=1e-5,
            )


def test_ckm_jarlskog_matches_conditional_texture_value() -> None:
    assert float(sp.N(ckm_jarlskog(), 20)) == pytest.approx(2.98e-5, abs=2e-8)


def test_ckm_payload_reports_conditional_pass() -> None:
    payload = ckm_conditional_audit_payload()
    assert payload.final_verdict == "CKM_CONDITIONAL_ASSEMBLY_PASS"
    assert payload.prerequisites_pass
    assert payload.conditionally_assembled
    assert payload.phase_angle == sp.atan(sp.sqrt(5))
    assert float(sp.N(payload.jarlskog, 20)) == pytest.approx(2.98e-5, abs=2e-8)


def test_q1_q2_q3_regressions_remain_stable() -> None:
    assert quark_boundary_shell_audit_payload().final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
    assert (
        quark_transfer_hierarchy_audit_payload().final_verdict
        == "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
    )
    assert quark_clebsch_audit_payload().final_verdict == "QUARK_CLEBSCH_Q3_PASS"
