from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.algebra.commutants import realify
from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.sm.classification import (
    NamedRealGate,
    canonical_gate_examples,
    certificate_to_dict,
    classify_real_gate,
    gate_classification_certificate,
)
from clifford_3plus2_d5.sm.embedding import complex_projector_3, matrix_unit


def test_canonical_gate_examples_match_expected_classes() -> None:
    certificate = gate_classification_certificate()

    assert certificate.gate_classification_check_passed
    assert certificate.commutant_basis_matches_expected
    assert certificate.safe_algebra_closure_passed
    assert not certificate.qca_geometric_gate_algebra_safe
    assert set(certificate.unsafe_gate_witnesses) == {
        "block_mixer",
        "rank_one_color_projector",
        "rank_one_weak_projector",
        "real_conjugation",
    }
    assert all(gate.matches_expected for gate in certificate.gates)


def test_safe_projectors_and_j_projectors_are_safe() -> None:
    examples = {gate.name: gate for gate in canonical_gate_examples()}

    assert classify_real_gate(examples["P_3"].matrix) == "safe_sm_commutant"
    assert classify_real_gate(examples["P_2"].matrix) == "safe_sm_commutant"
    assert classify_real_gate(examples["J_P_3"].matrix) == "safe_sm_commutant"
    assert classify_real_gate(examples["J_P_2"].matrix) == "safe_sm_commutant"


def test_block_mixer_is_rejected() -> None:
    mixer = realify(matrix_unit(0, 3) + matrix_unit(3, 0))

    assert classify_real_gate(mixer) == "block_mixing_fail"


def test_rank_one_color_projector_is_rejected() -> None:
    projector = realify(sp.diag(1, 0, 0, 0, 0))

    assert classify_real_gate(projector) == "color_breaking_fail"


def test_rank_one_weak_projector_is_rejected() -> None:
    projector = realify(sp.diag(0, 0, 0, 1, 0))

    assert classify_real_gate(projector) == "weak_breaking_fail"


def test_antilinear_gate_is_rejected() -> None:
    conjugation = identity(5).row_join(sp.zeros(5)).col_join(
        sp.zeros(5).row_join(-identity(5))
    )

    assert classify_real_gate(conjugation) == "antilinear_fail"


def test_malformed_gate_is_unknown() -> None:
    assert classify_real_gate(sp.eye(5)) == "unknown_fail"


def test_custom_safe_gate_set_can_be_all_safe_without_bridge_claim() -> None:
    gates = (
        NamedRealGate("P_3", realify(complex_projector_3()), "safe_sm_commutant"),
    )
    certificate = gate_classification_certificate(gates)

    assert certificate.gate_classification_check_passed
    assert certificate.qca_geometric_gate_algebra_safe
    assert not certificate.load_bearing_qca_bridge


def test_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(gate_classification_certificate())

    assert payload["gate_classification_check_passed"] is True
    assert payload["commutant_basis_matches_expected"] is True
    assert payload["safe_algebra_closure_passed"] is True
    assert payload["qca_geometric_gate_algebra_safe"] is False
    assert payload["load_bearing_qca_bridge"] is False
    classifications = {gate["name"]: gate["classification"] for gate in payload["gates"]}
    assert classifications["P_3"] == "safe_sm_commutant"
    assert classifications["block_mixer"] == "block_mixing_fail"
    assert classifications["rank_one_color_projector"] == "color_breaking_fail"
    assert classifications["rank_one_weak_projector"] == "weak_breaking_fail"
    assert classifications["real_conjugation"] == "antilinear_fail"
