from __future__ import annotations

from clifford_3plus2_d5.search.addressability import (
    off_block_mixer_control,
    rank_one_color_projector_controls,
    rank_one_weak_projector_controls,
    standard_block_projectors,
)
from clifford_3plus2_d5.search.normalizer import (
    certificate_to_dict,
    centralizer_basis,
    full_u5_like_addressability_controls,
    normalizer_certificate,
    orthogonal_centralizer_basis,
)


def test_unconstrained_orthogonal_normalizer_is_too_large() -> None:
    assert len(centralizer_basis(())) == 100
    assert len(orthogonal_centralizer_basis(())) == 45


def test_default_normalizer_certificate_is_candidate_only() -> None:
    certificate = normalizer_certificate()

    assert certificate.centralizer_dimension == 52
    assert certificate.orthogonal_normalizer_dimension == 21
    assert certificate.addressability_algebra_dimension == 2
    assert certificate.candidate_j_valid
    assert certificate.candidate_split_valid
    assert not certificate.candidate_j_preserved_by_normalizer
    assert certificate.normalizer_preserves_declared_split
    assert certificate.continuous_j_alternatives_not_excluded
    assert certificate.rank_three_projector_family_not_excluded
    assert not certificate.j_unique_or_forced
    assert not certificate.split_unique_or_forced
    assert certificate.addressability_algebra_safe
    assert certificate.forcedness_verdict == "candidate_only"
    assert not certificate.load_bearing_qca_bridge


def test_empty_rule_data_does_not_preserve_the_declared_split() -> None:
    certificate = normalizer_certificate(rule_data_operators=())
    payload = certificate_to_dict(certificate)

    assert certificate.orthogonal_normalizer_dimension == 45
    assert not certificate.normalizer_preserves_declared_split
    assert certificate.forcedness_verdict == "candidate_only"
    assert payload["normalizer_check_passed"] is False


def test_source_backed_split_without_j_forced_is_only_weak_orbit() -> None:
    certificate = normalizer_certificate(
        rule_data_source="synthetic_qca_rule_data",
        qca_rule_forces_j=True,
        qca_rule_forces_split=True,
    )

    assert not certificate.j_unique_or_forced
    assert certificate.split_unique_or_forced
    assert certificate.forcedness_verdict == "weak_orbit"
    assert not certificate.load_bearing_qca_bridge


def test_rank_one_color_projectors_falsify_forcedness() -> None:
    operators = standard_block_projectors() + rank_one_color_projector_controls()
    certificate = normalizer_certificate(addressable_operators=operators)

    assert certificate.rank_one_color_projectors_addressable
    assert not certificate.rank_one_weak_projectors_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.forcedness_verdict == "falsified"


def test_rank_one_weak_projectors_falsify_forcedness() -> None:
    operators = standard_block_projectors() + rank_one_weak_projector_controls()
    certificate = normalizer_certificate(addressable_operators=operators)

    assert not certificate.rank_one_color_projectors_addressable
    assert certificate.rank_one_weak_projectors_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.forcedness_verdict == "falsified"


def test_off_block_control_falsifies_forcedness() -> None:
    operators = standard_block_projectors() + (off_block_mixer_control(),)
    certificate = normalizer_certificate(addressable_operators=operators)

    assert certificate.off_block_controls_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.forcedness_verdict == "falsified"


def test_full_u5_like_controls_falsify_forcedness() -> None:
    operators = standard_block_projectors() + full_u5_like_addressability_controls()
    certificate = normalizer_certificate(addressable_operators=operators)

    assert certificate.rank_one_color_projectors_addressable
    assert certificate.rank_one_weak_projectors_addressable
    assert certificate.off_block_controls_addressable
    assert not certificate.addressability_algebra_safe
    assert certificate.forcedness_verdict == "falsified"


def test_certificate_serialization_is_stable() -> None:
    payload = certificate_to_dict(normalizer_certificate())

    assert payload["rule_data_source"] == "candidate_only"
    assert payload["centralizer_dimension"] == 52
    assert payload["orthogonal_normalizer_dimension"] == 21
    assert payload["addressability_algebra_dimension"] == 2
    assert payload["candidate_j_valid"] is True
    assert payload["candidate_split_valid"] is True
    assert payload["candidate_j_preserved_by_normalizer"] is False
    assert payload["normalizer_preserves_declared_split"] is True
    assert payload["continuous_j_alternatives_not_excluded"] is True
    assert payload["rank_three_projector_family_not_excluded"] is True
    assert payload["forcedness_verdict"] == "candidate_only"
    assert payload["normalizer_check_passed"] is True
    assert payload["load_bearing_qca_bridge"] is False
