"""Tests for ``sme_tensor_mapping.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.sme.sme_tensor_mapping import (
    dim5_sme_target_label,
    expected_cpt_class,
    h1_per_chirality_block,
    identity_pauli_polynomial_is_zero,
    lower_chirality_block_equals_upper,
    mapping_audit_payload,
    pauli_decomposition_per_chirality,
    t2g_tensor_entries,
)


def test_per_chirality_block_is_2x2() -> None:
    block = h1_per_chirality_block()
    assert block.shape == (2, 2)


def test_lower_chirality_block_equals_upper() -> None:
    assert lower_chirality_block_equals_upper()


def test_identity_pauli_polynomial_is_zero() -> None:
    assert identity_pauli_polynomial_is_zero()


def test_pauli_decomposition_matches_expected_form() -> None:
    coefficients = pauli_decomposition_per_chirality()
    _, kx, ky, kz = sp.symbols("epsilon kx ky kz", positive=True)
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    # Expected:
    #   sigma^I:  0
    #   sigma^x:  k_y k_z
    #   sigma^y: -k_x k_z
    #   sigma^z:  k_x k_y
    assert sp.expand(coefficients["I"]) == 0
    assert sp.expand(coefficients["x"] - ky * kz) == 0
    assert sp.expand(coefficients["y"] + kx * kz) == 0
    assert sp.expand(coefficients["z"] - kx * ky) == 0


def test_three_nonzero_t2g_tensor_entries() -> None:
    entries = t2g_tensor_entries()
    assert len(entries) == 3
    by_pauli = {e.pauli_index: e for e in entries}
    assert set(by_pauli) == {"x", "y", "z"}


def test_t2g_entries_have_expected_signs() -> None:
    entries = t2g_tensor_entries()
    by_pauli = {e.pauli_index: e for e in entries}
    assert by_pauli["x"].coefficient == 1
    assert by_pauli["x"].momentum_pair == ("y", "z")
    assert by_pauli["y"].coefficient == -1
    assert by_pauli["y"].momentum_pair == ("x", "z")
    assert by_pauli["z"].coefficient == 1
    assert by_pauli["z"].momentum_pair == ("x", "y")


def test_sme_target_label_is_d5() -> None:
    label = dim5_sme_target_label()
    assert "d^{(5)}" in label
    assert "axial-vector" in label
    assert "CPT-even" in label
    assert "CP-odd" in label


def test_expected_cpt_class_is_even() -> None:
    assert expected_cpt_class() == "CPT-even"


def test_mapping_audit_payload_consistent() -> None:
    payload = mapping_audit_payload()
    assert payload.identity_coefficient_vanishes
    assert payload.lower_block_equals_upper
    assert payload.nonzero_component_count == 3
    assert "MAPPING IDENTIFIED" in payload.verdict
    assert payload.cpt_class == "CPT-even"


def test_mapping_is_provisional_until_caveats_resolved() -> None:
    # Currently the module-level flags
    # FIELD_REDEFINITION_CHECKED and KM_HAMILTONIAN_NORMALIZATION_DERIVED
    # are both False, so the mapping should report itself as provisional.
    payload = mapping_audit_payload()
    assert payload.field_redefinition_checked is False
    assert payload.km_hamiltonian_normalization_derived is False
    assert payload.mapping_is_provisional is True
    assert "PROVISIONAL" in payload.verdict
    assert "Provisional" in payload.interpretation
