"""Tests for down-sector primitive-label subset counts."""

import sympy as sp

from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    color_only_middle_control,
    compressed_partition_cannot_derive_bcc_count,
    down_baseline_clebsch_vector,
    down_baseline_counts,
    down_baseline_mass_ratio_predictions,
    down_candidate_clebsch_vector,
    down_candidate_counts,
    down_candidate_mass_ratio_predictions,
    down_subset_audit_payload,
    odd_shell_plus_one_is_open,
    primitive_channel_subsets,
    s3_projector_bottom_control,
)


def test_primitive_channel_candidate_subsets_have_counts_6_2_5() -> None:
    subsets = primitive_channel_subsets()
    assert tuple(subsets) == ("d", "s", "b")
    assert down_candidate_counts() == {"d": 6, "s": 2, "b": 5}


def test_s3_baseline_counts_are_6_2_4() -> None:
    assert down_baseline_counts() == {"d": 6, "s": 2, "b": 4}


def test_down_candidate_clebsches_are_sqrt_n_over_six() -> None:
    assert down_candidate_clebsch_vector() == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(5, 6)),
    )


def test_down_s3_baseline_clebsches_match_old_projector_value() -> None:
    assert down_baseline_clebsch_vector() == (
        sp.Integer(1),
        1 / sp.sqrt(3),
        sp.sqrt(sp.Rational(2, 3)),
    )
    assert s3_projector_bottom_control() == down_baseline_clebsch_vector()


def test_down_ratio_formulas_are_exact() -> None:
    eta = sp.Symbol("eta", positive=True)
    baseline = down_baseline_mass_ratio_predictions(eta)
    candidate = down_candidate_mass_ratio_predictions(eta)
    assert sp.simplify(baseline["m_d/m_s"] - sp.sqrt(3) * eta**2) == 0
    assert sp.simplify(baseline["m_s/m_b"] - eta**2 / sp.sqrt(2)) == 0
    assert sp.simplify(candidate["m_d/m_s"] - sp.sqrt(3) * eta**2) == 0
    assert sp.simplify(candidate["m_s/m_b"] - sp.sqrt(sp.Rational(2, 5)) * eta**2) == 0


def test_down_controls_and_open_burden_are_recorded() -> None:
    assert compressed_partition_cannot_derive_bcc_count()
    assert color_only_middle_control() != down_candidate_clebsch_vector()
    assert odd_shell_plus_one_is_open()


def test_down_subset_payload_passes_conditionally() -> None:
    payload = down_subset_audit_payload()
    assert payload.final_verdict == "DOWN_S3_BASELINE_ODD_SHELL_CANDIDATE_PASS"
    assert payload.quark_shell_prerequisite_pass
    assert payload.baseline_counts == {"d": 6, "s": 2, "b": 4}
    assert payload.candidate_counts == {"d": 6, "s": 2, "b": 5}
    assert payload.compressed_partition_rejected
    assert payload.s3_projector_baseline_confirmed
    assert payload.plus_one_open
