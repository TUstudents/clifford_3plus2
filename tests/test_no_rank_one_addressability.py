from __future__ import annotations

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.projectors import mode_axis_projector
from clifford_3plus2_d5.search.addressability import (
    AddressableOperator,
    certificate_to_dict,
    classify_addressable_operator,
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
    structural_split_certificate,
)


def test_standard_block_projectors_are_safe_candidate_only() -> None:
    certificate = structural_split_certificate()

    assert certificate.projector_identities_passed
    assert certificate.projectors_commute_with_j
    assert certificate.projector_3_rank == 6
    assert certificate.projector_2_rank == 4
    assert certificate.addressability_algebra_safe
    assert not certificate.qca_supplies_structural_3plus2_split
    assert certificate.structural_split_verdict == "candidate_only"
    assert not certificate.load_bearing_qca_bridge


def test_rank_one_color_projector_falsifies_structural_split() -> None:
    operators = standard_block_projectors() + rank_one_color_projector_controls()
    certificate = structural_split_certificate(operators=operators)

    assert certificate.rank_one_color_projectors_addressable
    assert not certificate.rank_one_weak_projectors_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.structural_split_verdict == "falsified"


def test_rank_one_weak_projector_falsifies_structural_split() -> None:
    operators = standard_block_projectors() + rank_one_weak_projector_controls()
    certificate = structural_split_certificate(operators=operators)

    assert not certificate.rank_one_color_projectors_addressable
    assert certificate.rank_one_weak_projectors_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.structural_split_verdict == "falsified"


def test_off_block_mixer_falsifies_structural_split() -> None:
    operators = standard_block_projectors() + (off_block_mixer_control(),)
    certificate = structural_split_certificate(operators=operators)

    assert certificate.off_block_controls_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.structural_split_verdict == "falsified"


def test_block_diagonal_non_scalar_control_falsifies_structural_split() -> None:
    non_scalar = mode_axis_projector(0) + 2 * mode_axis_projector(1)
    operators = standard_block_projectors() + (
        AddressableOperator("color_cartan_like_control", non_scalar),
    )
    certificate = structural_split_certificate(operators=operators)

    assert certificate.non_scalar_block_controls_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.structural_split_verdict == "falsified"


def test_classification_distinguishes_safe_and_forbidden_controls() -> None:
    assert classify_addressable_operator(identity(10)) == "safe_block_projector"
    assert classify_addressable_operator(mode_axis_projector(0)) == "rank_one_color_projector"
    assert classify_addressable_operator(mode_axis_projector(3)) == "rank_one_weak_projector"
    assert classify_addressable_operator(off_block_mixer_control().matrix) == "off_block_mixer"


def test_synthetic_rule_forced_split_is_not_full_bridge() -> None:
    certificate = structural_split_certificate(
        qca_rule_forces_split=True,
        qca_structural_origin="qca_rule_data",
    )

    assert certificate.qca_supplies_structural_3plus2_split
    assert certificate.structural_split_verdict == "structural_split"
    assert not certificate.load_bearing_qca_bridge


def test_rule_forced_flag_without_structural_origin_does_not_pass() -> None:
    certificate = structural_split_certificate(qca_rule_forces_split=True)

    assert not certificate.qca_supplies_structural_3plus2_split
    assert certificate.structural_split_verdict == "candidate_only"


def test_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(structural_split_certificate())

    assert payload["projector_identities_passed"] is True
    assert payload["projectors_commute_with_j"] is True
    assert payload["projector_3_rank"] == 6
    assert payload["projector_2_rank"] == 4
    assert payload["qca_supplies_structural_3plus2_split"] is False
    assert payload["rank_one_color_projectors_addressable"] is False
    assert payload["rank_one_weak_projectors_addressable"] is False
    assert payload["addressability_algebra_safe"] is True
    assert payload["structural_split_verdict"] == "candidate_only"
    assert payload["structural_split_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False
