from __future__ import annotations

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.search.forced_j import (
    certificate_to_dict,
    is_complex_structure,
    is_real_orthogonal,
    j_certificate,
)
from clifford_3plus2_d5.obstruction_r10.search.gate_words import (
    rank_one_pair_rotations,
    standard_period_four_primitives,
)


def test_standard_j_is_exact_complex_structure() -> None:
    carrier = standard_real_carrier()
    j = carrier.complex_structure

    assert is_real_orthogonal(j, carrier.metric)
    assert is_complex_structure(j, carrier.metric)
    assert j.det() == 1
    assert j * j == -identity(10)


def test_standard_j_certificate_is_candidate_only_by_default() -> None:
    certificate = j_certificate()

    assert certificate.generated_by_gate_word
    assert certificate.gate_word == ("global_clock_tick",)
    assert certificate.is_real_orthogonal
    assert certificate.determinant == 1
    assert certificate.squares_to_minus_identity
    assert certificate.equals_standard_j
    assert not certificate.rank_one_pair_rotations_addressable
    assert not certificate.qca_forces_j
    assert certificate.verdict == "candidate_only"


def test_forced_j_requires_explicit_rule_forced_word() -> None:
    certificate = j_certificate(qca_rule_forces_word=True)

    assert certificate.qca_forces_j
    assert certificate.verdict == "forced_j"


def test_rank_one_pair_addressability_falsifies_candidate() -> None:
    primitives = standard_period_four_primitives() + rank_one_pair_rotations()
    certificate = j_certificate(primitives=primitives, qca_rule_forces_word=True)

    assert certificate.generated_by_gate_word
    assert certificate.rank_one_pair_rotations_addressable
    assert not certificate.qca_forces_j
    assert certificate.verdict == "falsified"


def test_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(j_certificate())

    assert payload == {
        "candidate_name": "standard_clock_j",
        "generated_by_gate_word": True,
        "gate_word": ["global_clock_tick"],
        "is_real_orthogonal": True,
        "determinant": 1,
        "squares_to_minus_identity": True,
        "equals_standard_j": True,
        "rank_one_pair_rotations_addressable": False,
        "qca_forces_j": False,
        "forced_j_verdict": "candidate_only",
        "forced_j_check_passed": True,
        "load_bearing_qca_bridge": False,
    }
