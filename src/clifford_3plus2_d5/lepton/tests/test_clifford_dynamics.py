"""Tests for the Session 14 rigid Clifford dynamics audit."""

from __future__ import annotations

import json
import subprocess

from clifford_3plus2_d5.lepton.clifford_dynamics import (
    StabilizerClass,
    audit_clifford_dynamics_candidate,
    chiral_so8_generator,
    chiral_so8_generators,
    chiral_block_matrix,
    clifford_dynamics_audit_entries,
    clifford_dynamics_audit_payload,
    finite_group_audit_payload,
    finite_group_closure,
    generated_lie_algebra_dimension,
    is_octonion_automorphism,
    iter_clifford_dynamics_candidates,
    known_octonion_automorphism,
    normalizes_lie_algebra,
)
from clifford_3plus2_d5.lepton.clifford_octonion import (
    cl08_gamma_matrices,
    octonion_derivation_basis,
    su3_stabilizer_basis,
)


def _candidate_by_name(name: str):
    for candidate in iter_clifford_dynamics_candidates():
        if candidate.name == name:
            return candidate
    raise AssertionError(f"missing candidate {name}")


def test_rigid_candidate_family_counts_are_stable() -> None:
    payload = clifford_dynamics_audit_payload()
    assert payload["candidate_count"] == 107
    assert payload["family_counts"] == {
        "identity": 1,
        "reflection": 8,
        "bivector": 28,
        "four_vector": 70,
    }
    assert payload["chirality_preserving_count"] == 99


def test_chiral_block_extraction_distinguishes_reflections_from_even_words() -> None:
    gammas = cl08_gamma_matrices()
    assert chiral_block_matrix(gammas[0], "+") is None
    assert chiral_block_matrix(gammas[0] * gammas[1], "+") is not None


def test_stabilizer_controls_are_explicit() -> None:
    identity_entry = audit_clifford_dynamics_candidate(_candidate_by_name("identity"))
    assert identity_entry.stabilizer_class == StabilizerClass.SU3_FIXING_E7
    assert identity_entry.octonion_automorphism

    nontrivial_g2 = known_octonion_automorphism()
    assert is_octonion_automorphism(nontrivial_g2)
    assert normalizes_lie_algebra(nontrivial_g2, octonion_derivation_basis())

    non_g2_spin_word = audit_clifford_dynamics_candidate(_candidate_by_name("gamma_1_2"))
    assert non_g2_spin_word.chirality_preserving
    assert non_g2_spin_word.stabilizer_class == StabilizerClass.SPIN8_BEYOND_G2

    chirality_swapper = audit_clifford_dynamics_candidate(_candidate_by_name("gamma_1"))
    assert not chirality_swapper.chirality_preserving
    assert chirality_swapper.stabilizer_class == StabilizerClass.NOT_CHIRALITY_PRESERVING


def test_clifford_four_vectors_find_discrete_su3_classes() -> None:
    fixing = audit_clifford_dynamics_candidate(_candidate_by_name("gamma_1_2_3_5"))
    flipping = audit_clifford_dynamics_candidate(_candidate_by_name("gamma_1_2_4_7"))
    assert fixing.stabilizer_class == StabilizerClass.SU3_FIXING_E7
    assert flipping.stabilizer_class == StabilizerClass.SU3_FLIPPING_E7


def test_audit_class_counts_are_stable() -> None:
    entries = clifford_dynamics_audit_entries()
    class_counts = {
        stabilizer_class.value: sum(1 for entry in entries if entry.stabilizer_class == stabilizer_class)
        for stabilizer_class in StabilizerClass
    }
    assert class_counts == {
        "su3_fixing_e7": 3,
        "su3_flipping_e7": 4,
        "g2_beyond_su3": 0,
        "spin8_beyond_g2": 92,
        "not_chirality_preserving": 8,
    }
    assert sum(1 for entry in entries if entry.octonion_automorphism) == 7
    assert sum(1 for entry in entries if entry.normalizes_g2) == 15
    assert sum(1 for entry in entries if entry.normalizes_su3) == 27


def test_lie_closure_dimension_is_zero_for_rigid_finite_representatives() -> None:
    payload = clifford_dynamics_audit_payload()
    assert payload["expected_g2_dimension"] == 14
    assert payload["expected_su3_dimension"] == 8
    assert payload["g2_algebra_closure_dimension"] == 0
    assert payload["su3_algebra_closure_dimension"] == 0

    g2_basis = octonion_derivation_basis()
    su3_basis = su3_stabilizer_basis(7)
    assert generated_lie_algebra_dimension(g2_basis, ambient_basis=g2_basis) == 14
    assert generated_lie_algebra_dimension(su3_basis, ambient_basis=su3_basis) == 8


def test_chiral_bivectors_are_so8_lie_generators() -> None:
    first = chiral_so8_generator(0, 1)
    assert first.shape == (8, 8)
    assert first + first.T == first.zeros(8, 8)

    generators = chiral_so8_generators()
    assert len(generators) == 28
    payload = clifford_dynamics_audit_payload()["lie_bivector_audit"]
    assert payload["so8_bivector_span_dimension"] == 28
    assert payload["individual_bivectors_in_g2_count"] == 0
    assert payload["individual_bivectors_in_su3_count"] == 0
    assert payload["g2_inside_bivector_span"] is True
    assert payload["su3_inside_bivector_span"] is True
    assert payload["g2_plus_bivectors_span_dimension"] == 28
    assert payload["su3_plus_bivectors_span_dimension"] == 28


def test_finite_group_closure_of_rigid_automorphisms_is_stable() -> None:
    payload = finite_group_audit_payload()
    assert payload["g2_rigid_automorphism_generator_count"] == 7
    assert payload["g2_rigid_finite_closure_order"] == 8
    assert payload["g2_rigid_finite_closure_status"] == "closed"
    assert payload["su3_fixing_generator_count"] == 3
    assert payload["su3_fixing_finite_closure_order"] == 4
    assert payload["su3_fixing_finite_closure_status"] == "closed"
    assert payload["su3_fixing_plus_flip_generator_count"] == 7
    assert payload["su3_fixing_plus_flip_finite_closure_order"] == 8
    assert payload["su3_fixing_plus_flip_finite_closure_status"] == "closed"

    identity_closure = finite_group_closure((known_octonion_automorphism(),))
    assert identity_closure.status == "closed"
    assert identity_closure.order == 3


def test_clifford_dynamics_audit_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "clifford_3plus2_d5.lepton.scripts.clifford_dynamics_audit",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["candidate_count"] == 107
    assert payload["octonion_automorphism_count"] == 7
    assert payload["g2_algebra_closure_dimension"] == 0
    assert payload["lie_bivector_audit"]["so8_bivector_span_dimension"] == 28
    assert payload["finite_group_audit"]["g2_rigid_finite_closure_order"] == 8
