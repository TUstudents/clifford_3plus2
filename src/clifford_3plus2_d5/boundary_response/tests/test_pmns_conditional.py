"""Tests for the V9 conditional PMNS assembly."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    charged_lepton_leakage_audit_payload,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    leptonic_phase_word_audit_payload,
)
from clifford_3plus2_d5.boundary_response.pmns_conditional import (
    charged_lepton_rotation_matrix,
    conditional_pmns_matrix,
    pmns_conditional_audit_payload,
    pmns_mixing_observables,
    tbm_from_residual_basis,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon_fourth
from clifford_3plus2_d5.boundary_response.weyl_sterile import weyl_sterile_audit_payload


def _assert_unitary(matrix: sp.Matrix) -> None:
    residual = matrix.conjugate().T * matrix - sp.eye(matrix.rows)
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_tbm_from_residual_basis_is_unitary() -> None:
    _assert_unitary(tbm_from_residual_basis())


def test_charged_lepton_rotation_is_unitary() -> None:
    _assert_unitary(charged_lepton_rotation_matrix())


def test_conditional_pmns_matrix_is_unitary() -> None:
    matrix = conditional_pmns_matrix()
    residual = matrix.conjugate().T * matrix - sp.eye(3)
    numeric_residual = residual.applyfunc(lambda entry: complex(sp.N(entry, 30)))
    assert max(abs(entry) for entry in numeric_residual) < 1e-25


def test_conditional_pmns_theta13_matches_exact_leakage_prediction() -> None:
    observables = pmns_mixing_observables()
    expected = sp.Rational(3, 4) * epsilon_fourth()
    assert sp.simplify(observables.sin2_theta13 - expected) == 0


def test_conditional_pmns_angles_match_note_values() -> None:
    observables = pmns_mixing_observables()
    assert float(sp.N(observables.sin2_theta13)) == pytest.approx(0.0220779, abs=1e-7)
    assert float(sp.N(observables.sin2_theta12)) == pytest.approx(0.304610, abs=1e-6)
    assert float(sp.N(observables.sin2_theta23)) == pytest.approx(0.488712, abs=1e-6)


def test_conditional_pmns_cp_branches_are_conjugate() -> None:
    default = pmns_mixing_observables()
    conjugate = pmns_mixing_observables(conjugate_branch=True)
    assert sp.simplify(default.jarlskog + conjugate.jarlskog) == 0
    assert default.delta_cp_degrees == pytest.approx(261.6315, abs=1e-4)
    assert conjugate.delta_cp_degrees == pytest.approx(98.3685, abs=1e-4)


def test_conditional_pmns_payload_reports_conditional_pass() -> None:
    payload = pmns_conditional_audit_payload()
    assert payload.final_verdict == "PMNS_CONDITIONAL_ASSEMBLY_PASS"
    assert sp.simplify(payload.sin2_theta13 - sp.Rational(3, 4) * epsilon_fourth()) == 0
    assert not payload.phase_word_derived
    assert payload.conditional_on_phase_word
    assert payload.ckm_parked


def test_v6_v7_v8_regressions_remain_gated() -> None:
    assert weyl_sterile_audit_payload().final_verdict == "PRODUCT_STERILE_LIMIT_PASS"
    assert charged_lepton_leakage_audit_payload().final_verdict == "CHARGED_LEPTON_LEAKAGE_PASS"
    assert (
        leptonic_phase_word_audit_payload().final_verdict
        == "LEPTONIC_PHASE_WORD_CONDITIONAL_PASS"
    )
