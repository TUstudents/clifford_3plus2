"""Tests for the finite S3 projector audit."""

import sympy as sp

from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    central_projector_ranks,
    central_projectors,
    central_s3_does_not_force_rank_two,
    rank_five_is_not_unique,
    rank_five_projector_ranks,
    s3_projector_audit_payload,
    standard_copy_projector,
    standard_copy_projector_rank,
)


def test_central_projectors_have_regular_s3_dimensions() -> None:
    projectors = central_projectors()
    assert projectors["trivial"] ** 2 == projectors["trivial"]
    assert projectors["sign"] ** 2 == projectors["sign"]
    assert projectors["standard_isotypic"] ** 2 == projectors["standard_isotypic"]
    assert projectors["trivial"] + projectors["sign"] + projectors["standard_isotypic"] == sp.eye(6)
    assert central_projector_ranks() == {
        "0": 0,
        "trivial": 1,
        "sign": 1,
        "standard_isotypic": 4,
        "trivial_plus_sign": 2,
        "trivial_plus_standard": 5,
        "sign_plus_standard": 5,
        "regular": 6,
    }


def test_rank_two_is_defect_selected_not_central() -> None:
    projector = standard_copy_projector()
    assert standard_copy_projector_rank() == 2
    assert projector**2 == projector
    assert central_s3_does_not_force_rank_two()


def test_rank_five_exists_but_is_not_unique() -> None:
    assert rank_five_projector_ranks() == {
        "regular_minus_trivial": 5,
        "regular_minus_sign": 5,
    }
    assert rank_five_is_not_unique()


def test_s3_projector_payload_localizes_down_count_gap() -> None:
    payload = s3_projector_audit_payload()
    assert payload.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
    assert payload.count_vector_available
    assert not payload.count_vector_forced_by_s3_alone
    assert payload.rank_two_requires_defect_polarization
    assert payload.rank_five_not_unique
