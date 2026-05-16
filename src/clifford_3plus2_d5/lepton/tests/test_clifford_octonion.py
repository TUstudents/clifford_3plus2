"""Tests for the Cl(0,8) / octonion stabilizer audit."""

from __future__ import annotations

import json
import subprocess

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_octonion import (
    chirality_projectors,
    cl02_complex_structure_candidates,
    cl08_even_commutant_basis,
    cl08_full_commutant_basis,
    cl08_gamma_matrices,
    clifford_octonion_audit_payload,
    clifford_relations_pass,
    octonion_derivation_basis,
    octonion_fano_triples,
    octonion_multiply,
    su3_stabilizer_basis,
    volume_element,
)


def _basis(index: int) -> sp.Matrix:
    vector = sp.zeros(8, 1)
    vector[index] = 1
    return vector


def test_cl08_gamma_relations_and_chirality() -> None:
    gammas = cl08_gamma_matrices()
    assert len(gammas) == 8
    assert all(gamma.shape == (16, 16) for gamma in gammas)
    assert clifford_relations_pass(gammas)

    omega = volume_element(gammas)
    assert omega * omega == sp.eye(16)
    p_plus, p_minus = chirality_projectors(gammas)
    assert (p_plus * p_plus).applyfunc(sp.simplify) == p_plus
    assert (p_minus * p_minus).applyfunc(sp.simplify) == p_minus
    assert (p_plus * p_minus).applyfunc(sp.simplify) == sp.zeros(16)
    assert (p_plus + p_minus).applyfunc(sp.simplify) == sp.eye(16)
    assert (p_plus.rank(), p_minus.rank()) == (8, 8)


def test_cl08_commutants_match_expected_dimensions() -> None:
    full = cl08_full_commutant_basis()
    even = cl08_even_commutant_basis()
    assert len(full) == 1
    assert len(even) == 2
    assert any(matrix == sp.eye(16) for matrix in full)

    omega = volume_element()
    assert any(matrix == omega for matrix in even)
    assert all((omega * matrix - matrix * omega).applyfunc(sp.simplify) == sp.zeros(16) for matrix in even)


def test_cl02_has_three_quaternionic_j_candidates() -> None:
    candidates = cl02_complex_structure_candidates()
    assert len(candidates) == 3
    assert all(candidate * candidate == -sp.eye(4) for candidate in candidates)
    for left_index, left in enumerate(candidates):
        for right in candidates[:left_index]:
            assert (left * right + right * left).applyfunc(sp.simplify) == sp.zeros(4)


def test_octonion_table_choice_and_basis_products() -> None:
    assert octonion_fano_triples() == (
        (1, 2, 3),
        (1, 4, 5),
        (1, 6, 7),
        (4, 2, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 5, 6),
    )
    one = _basis(0)
    for index in range(1, 8):
        assert octonion_multiply(_basis(index), _basis(index)) == -one

    assert octonion_multiply(_basis(1), _basis(2)) == _basis(3)
    assert octonion_multiply(_basis(2), _basis(1)) == -_basis(3)
    assert octonion_multiply(_basis(4), _basis(2)) == _basis(6)
    assert octonion_multiply(_basis(2), _basis(4)) == -_basis(6)


def test_octonion_alternativity_on_basis_samples() -> None:
    for left_index in (1, 2, 4, 7):
        left = _basis(left_index)
        for right_index in (1, 3, 5):
            right = _basis(right_index)
            assert octonion_multiply(octonion_multiply(left, left), right) == octonion_multiply(
                left,
                octonion_multiply(left, right),
            )
            assert octonion_multiply(octonion_multiply(right, left), left) == octonion_multiply(
                right,
                octonion_multiply(left, left),
            )


def test_octonion_derivations_reveal_g2_dimension() -> None:
    derivations = octonion_derivation_basis()
    assert len(derivations) == 14
    for derivation in derivations:
        assert derivation * _basis(0) == sp.zeros(8, 1)
        assert (derivation + derivation.T).applyfunc(sp.simplify) == sp.zeros(8)
        assert derivation[0, :] == sp.zeros(1, 8)
        assert derivation[:, 0] == sp.zeros(8, 1)


def test_su3_stabilizer_of_e7_has_dimension_eight() -> None:
    stabilizer = su3_stabilizer_basis(7)
    assert len(stabilizer) == 8
    for derivation in stabilizer:
        assert derivation * _basis(0) == sp.zeros(8, 1)
        assert derivation * _basis(7) == sp.zeros(8, 1)
        assert derivation[0, :] == sp.zeros(1, 8)
        assert derivation[:, 0] == sp.zeros(8, 1)
        assert derivation[7, :] == sp.zeros(1, 8)
        assert derivation[:, 7] == sp.zeros(8, 1)
    assert any(derivation[1:7, 1:7].rank() > 0 for derivation in stabilizer)


def test_clifford_octonion_audit_payload_is_stable() -> None:
    payload = clifford_octonion_audit_payload()
    assert payload["signature"] == "Cl(0,8)"
    assert payload["clifford_relations_pass"] is True
    assert payload["chirality_ranks"] == (8, 8)
    assert payload["full_commutant_dimension"] == 1
    assert payload["even_commutant_dimension"] == 2
    assert payload["cl02_j_candidate_count"] == 3
    assert payload["g2_derivation_dimension"] == 14
    assert payload["su3_stabilizer_dimension"] == 8
    assert "which Cl(0,2) unit is called J" in payload["choices"]


def test_clifford_octonion_audit_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "clifford_3plus2_d5.lepton.scripts.clifford_octonion_audit",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["signature"] == "Cl(0,8)"
    assert payload["g2_derivation_dimension"] == 14
    assert payload["su3_stabilizer_dimension"] == 8
