from __future__ import annotations

from clifford_3plus2_d5.search.real_qca_branch import (
    certificate_to_dict,
    controls_with_off_block,
    controls_with_rank_one_color,
    controls_with_rank_one_weak,
    rank_one_pair_rotation_primitives,
    real_qca_branch_certificate,
    standard_real_qca_primitives,
)


def test_default_real_qca_branch_is_candidate_only() -> None:
    certificate = real_qca_branch_certificate()

    assert certificate.branch_name == "phase_8a_stronger_real_qca_first"
    assert certificate.candidate_word_found
    assert certificate.candidate_word == ("global_clock_tick",)
    assert certificate.word_depth == 1
    assert certificate.finite_depth
    assert certificate.translation_invariant
    assert certificate.generates_j
    assert certificate.generates_split
    assert not certificate.j_forced_by_rule_space
    assert not certificate.split_forced_by_rule_space
    assert not certificate.forbidden_rank_one_controls_present
    assert not certificate.forbidden_off_block_controls_present
    assert certificate.addressability_algebra_safe
    assert certificate.normalizer_verdict == "candidate_only"
    assert certificate.real_qca_branch_verdict == "candidate_only"
    assert not certificate.load_bearing_qca_bridge


def test_declared_period_four_word_generates_j_but_does_not_force_j() -> None:
    certificate = real_qca_branch_certificate(
        primitives=standard_real_qca_primitives(),
        max_depth=1,
        rule_data_source="synthetic_rule_space",
        qca_rule_forces_j=True,
    )

    assert certificate.generates_j
    assert certificate.forced_j_certificate["qca_forces_j"] is True
    assert not certificate.j_forced_by_rule_space
    assert certificate.normalizer_verdict == "weak_orbit"
    assert certificate.real_qca_branch_verdict == "candidate_only"


def test_rank_one_pair_primitive_falsifies_branch() -> None:
    primitives = standard_real_qca_primitives() + rank_one_pair_rotation_primitives()
    certificate = real_qca_branch_certificate(primitives=primitives)

    assert certificate.forbidden_rank_one_controls_present
    assert certificate.forced_j_certificate["rank_one_pair_rotations_addressable"] is True
    assert certificate.real_qca_branch_verdict == "falsified"


def test_rank_one_color_control_falsifies_branch() -> None:
    certificate = real_qca_branch_certificate(
        addressable_operators=controls_with_rank_one_color()
    )

    assert certificate.forbidden_rank_one_controls_present
    assert not certificate.addressability_algebra_safe
    assert certificate.real_qca_branch_verdict == "falsified"


def test_rank_one_weak_control_falsifies_branch() -> None:
    certificate = real_qca_branch_certificate(
        addressable_operators=controls_with_rank_one_weak()
    )

    assert certificate.forbidden_rank_one_controls_present
    assert not certificate.addressability_algebra_safe
    assert certificate.real_qca_branch_verdict == "falsified"


def test_off_block_control_falsifies_branch() -> None:
    certificate = real_qca_branch_certificate(addressable_operators=controls_with_off_block())

    assert certificate.forbidden_off_block_controls_present
    assert not certificate.addressability_algebra_safe
    assert certificate.real_qca_branch_verdict == "falsified"


def test_normalizer_certificate_is_embedded() -> None:
    certificate = real_qca_branch_certificate()

    assert certificate.normalizer_certificate["forcedness_verdict"] == "candidate_only"
    assert certificate.normalizer_certificate["j_unique_or_forced"] is False
    assert certificate.normalizer_certificate["split_unique_or_forced"] is False


def test_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(real_qca_branch_certificate())

    assert payload["candidate_word_found"] is True
    assert payload["candidate_word"] == ["global_clock_tick"]
    assert payload["finite_depth"] is True
    assert payload["translation_invariant"] is True
    assert payload["generates_j"] is True
    assert payload["generates_split"] is True
    assert payload["j_forced_by_rule_space"] is False
    assert payload["split_forced_by_rule_space"] is False
    assert payload["normalizer_verdict"] == "candidate_only"
    assert payload["real_qca_branch_verdict"] == "candidate_only"
    assert payload["real_qca_branch_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False
