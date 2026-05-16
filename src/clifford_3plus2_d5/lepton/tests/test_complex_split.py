"""Tests for the complex-linear split-first lab."""

from __future__ import annotations

from clifford_3plus2_d5.lepton.complex_carrier import (
    complex_c3_carrier_check_passed,
    complex_c5_carrier_check_passed,
)
from clifford_3plus2_d5.lepton.complex_primitives import (
    c3_full_irreducible_control,
    c3_rank_one_locking_control,
    c3_seeded_split_control,
    c3_synthetic_split_candidate,
    c5_full_irreducible_control,
    c5_conjugated_synthetic_split_candidate,
    c5_dense_conjugated_synthetic_split_candidate,
    c5_rank_one_locking_control,
    c5_seeded_split_control,
    c5_synthetic_split_candidate,
)
from clifford_3plus2_d5.lepton.complex_profiles import (
    complex_c3_split_profile,
    complex_c5_discovered_split_profile,
    complex_c5_split_profile,
)
from clifford_3plus2_d5.lepton.complex_scans import (
    complex_c3_summary,
    complex_c5_discovered_summary,
    complex_c5_summary,
    run_complex_c3_split_scan,
    run_complex_c5_discovered_split_scan,
    run_complex_c5_split_scan,
)
from clifford_3plus2_d5.lepton.complex_verdict import rule_to_complex_split_verdict


def _verdict(candidate_name: str) -> str:
    [row] = [
        item
        for item in run_complex_c3_split_scan(max_candidates=6)
        if item.rule_name == candidate_name
    ]
    return row.verdict


def _c5_verdict(candidate_name: str) -> str:
    [row] = [
        item
        for item in run_complex_c5_split_scan(max_candidates=6)
        if item.rule_name == candidate_name
    ]
    return row.verdict


def test_complex_carriers_pass() -> None:
    assert complex_c3_carrier_check_passed()
    assert complex_c5_carrier_check_passed()


def test_seeded_split_control_is_labeled_seeded() -> None:
    profile = complex_c3_split_profile()
    result = rule_to_complex_split_verdict(c3_seeded_split_control().layers, profile)
    assert result.verdict.value == "seeded_split_control"
    assert result.target_split_present
    assert result.seeded_split_present
    assert result.block_dimensions == (1, 4)


def test_full_irreducible_control_fails_no_split() -> None:
    profile = complex_c3_split_profile()
    result = rule_to_complex_split_verdict(c3_full_irreducible_control().layers, profile)
    assert result.verdict.value == "falsified_no_split"
    assert result.generated_algebra_dimension == 9
    assert result.center_dimension == 1
    assert not result.target_split_present


def test_rank_one_locking_control_fails_locking() -> None:
    profile = complex_c3_split_profile()
    result = rule_to_complex_split_verdict(c3_rank_one_locking_control().layers, profile)
    assert result.verdict.value == "falsified_rank_one_locking"
    assert result.target_split_present
    assert result.forbidden_rank_one_count >= 1


def test_synthetic_split_candidate_passes() -> None:
    profile = complex_c3_split_profile()
    result = rule_to_complex_split_verdict(c3_synthetic_split_candidate().layers, profile)
    assert result.verdict.value == "split_candidate"
    assert result.reason == "passed"
    assert result.generated_algebra_dimension == 5
    assert result.center_dimension == 2
    assert result.block_dimensions == (1, 4)
    assert result.block_commutativity == (True, False)
    assert not result.seeded_split_present


def test_complex_c3_scan_is_deterministic() -> None:
    rows = run_complex_c3_split_scan(max_candidates=4)
    assert tuple(row.verdict for row in rows) == (
        "seeded_split_control",
        "falsified_no_split",
        "falsified_rank_one_locking",
        "split_candidate",
    )
    assert _verdict("c3_synthetic_c_plus_m2_candidate") == "split_candidate"


def test_complex_c3_summary_counts_controls_and_candidate() -> None:
    summary = complex_c3_summary(max_candidates=4)
    assert summary.candidate_count == 4
    assert summary.split_candidate_count == 1
    assert summary.seeded_control_count == 1


def test_c5_seeded_split_control_is_labeled_seeded() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(c5_seeded_split_control().layers, profile)
    assert result.verdict.value == "seeded_split_control"
    assert result.target_split_present
    assert result.seeded_split_present
    assert result.block_dimensions == (9, 4)


def test_c5_full_irreducible_control_fails_no_split() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(c5_full_irreducible_control().layers, profile)
    assert result.verdict.value == "falsified_no_split"
    assert result.generated_algebra_dimension == 25
    assert result.center_dimension == 1
    assert not result.target_split_present


def test_c5_rank_one_locking_control_fails_locking() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(c5_rank_one_locking_control().layers, profile)
    assert result.verdict.value == "falsified_rank_one_locking"
    assert result.target_split_present
    assert result.forbidden_rank_one_count >= 1


def test_c5_synthetic_split_candidate_passes() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(c5_synthetic_split_candidate().layers, profile)
    assert result.verdict.value == "split_candidate"
    assert result.reason == "passed"
    assert result.generated_algebra_dimension == 13
    assert result.center_dimension == 2
    assert result.block_dimensions == (9, 4)
    assert result.block_commutativity == (False, False)
    assert not result.seeded_split_present


def test_complex_c5_scan_is_deterministic() -> None:
    rows = run_complex_c5_split_scan(max_candidates=4)
    assert tuple(row.verdict for row in rows) == (
        "seeded_split_control",
        "falsified_no_split",
        "falsified_rank_one_locking",
        "split_candidate",
    )
    assert _c5_verdict("c5_synthetic_m3_plus_m2_candidate") == "split_candidate"


def test_complex_c5_summary_counts_controls_and_candidate() -> None:
    summary = complex_c5_summary(max_candidates=4)
    assert summary.candidate_count == 4
    assert summary.split_candidate_count == 1
    assert summary.seeded_control_count == 1


def test_c5_discovered_profile_accepts_canonical_synthetic() -> None:
    profile = complex_c5_discovered_split_profile()
    result = rule_to_complex_split_verdict(c5_synthetic_split_candidate().layers, profile)
    assert result.verdict.value == "split_candidate"
    assert result.canonical_split_matched
    assert result.discovered_projector_ranks == (3, 2)
    assert result.block_dimensions == (9, 4)


def test_c5_fixed_profile_rejects_conjugated_synthetic() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(
        c5_conjugated_synthetic_split_candidate().layers,
        profile,
    )
    assert result.verdict.value == "falsified_no_split"
    assert not result.target_split_present


def test_c5_discovered_profile_accepts_conjugated_synthetic() -> None:
    profile = complex_c5_discovered_split_profile()
    result = rule_to_complex_split_verdict(
        c5_conjugated_synthetic_split_candidate().layers,
        profile,
    )
    assert result.verdict.value == "split_candidate"
    assert not result.canonical_split_matched
    assert result.discovered_projector_ranks == (3, 2)
    assert result.block_dimensions == (9, 4)
    assert result.block_commutativity == (False, False)


def test_complex_c5_discovered_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(max_candidates=5)
    assert tuple(row.verdict for row in rows) == (
        "seeded_split_control",
        "falsified_no_split",
        "falsified_rank_one_locking",
        "split_candidate",
        "split_candidate",
    )
    conjugated = rows[4]
    assert conjugated.rule_name == "c5_synthetic_m3_plus_m2_candidate_conjugated_swap_2_3"
    assert conjugated.discovered_projector_ranks == (3, 2)
    assert not conjugated.canonical_split_matched


def test_complex_c5_discovered_summary_counts_controls_and_candidates() -> None:
    summary = complex_c5_discovered_summary(max_candidates=5)
    assert summary.candidate_count == 5
    assert summary.split_candidate_count == 2
    assert summary.non_synthetic_split_candidate_count == 0
    assert summary.canonical_split_candidate_count == 1
    assert summary.noncanonical_split_candidate_count == 1
    assert summary.seeded_control_count == 1


def test_complex_c5_discovered_phase_permutation_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="phase-permutation",
        max_candidates=4,
    )
    assert tuple(row.verdict for row in rows) == (
        "not_solved",
        "falsified_no_split",
        "not_solved",
        "not_solved",
    )
    assert all(dict(row.metadata)["synthetic"] == "false" for row in rows)
    assert all(row.discovered_projector_ranks == () for row in rows)


def test_complex_c5_discovered_monomial_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="monomial",
        max_candidates=3,
    )
    assert tuple(row.verdict for row in rows) == (
        "falsified_no_split",
        "falsified_rank_one_locking",
        "falsified_no_split",
    )
    assert all(dict(row.metadata)["synthetic"] == "false" for row in rows)
    assert all(row.verdict != "split_candidate" for row in rows)


def test_complex_c5_discovered_finite_order_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="finite-order",
        max_candidates=4,
    )
    assert tuple(row.verdict for row in rows) == (
        "falsified_rank_one_locking",
        "falsified_rank_one_locking",
        "not_solved",
        "falsified_rank_one_locking",
    )
    assert all(dict(row.metadata)["synthetic"] == "false" for row in rows)
    assert all(row.verdict != "split_candidate" for row in rows)


def test_complex_c5_discovered_monomial_summary_reports_no_non_synthetic_split() -> None:
    summary = complex_c5_discovered_summary(
        family="monomial",
        max_candidates=3,
    )
    assert summary.candidate_count == 3
    assert summary.split_candidate_count == 0
    assert summary.non_synthetic_split_candidate_count == 0
    assert summary.seeded_control_count == 0
    assert summary.max_generated_algebra_dimension == 25


def test_c5_discovered_profile_accepts_dense_conjugated_control() -> None:
    profile = complex_c5_discovered_split_profile()
    result = rule_to_complex_split_verdict(
        c5_dense_conjugated_synthetic_split_candidate().layers,
        profile,
    )
    assert result.verdict.value == "split_candidate"
    assert not result.canonical_split_matched
    assert result.discovered_projector_ranks == (3, 2)
    assert result.block_dimensions == (9, 4)


def test_c5_fixed_profile_rejects_dense_conjugated_control() -> None:
    profile = complex_c5_split_profile()
    result = rule_to_complex_split_verdict(
        c5_dense_conjugated_synthetic_split_candidate().layers,
        profile,
    )
    assert result.verdict.value == "falsified_no_split"
    assert not result.target_split_present


def test_complex_c5_discovered_dense_control_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="dense-conjugated-control",
        max_candidates=1,
    )
    assert len(rows) == 1
    [row] = rows
    assert row.verdict == "split_candidate"
    assert row.discovered_projector_ranks == (3, 2)
    assert not row.canonical_split_matched
    metadata = dict(row.metadata)
    assert metadata["synthetic"] == "true"
    assert metadata["control"] == "true"
    assert metadata["dense"] == "true"


def test_complex_c5_discovered_dense_hadamard_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="dense-hadamard",
        max_candidates=3,
    )
    assert tuple(row.verdict for row in rows) == ("not_solved", "not_solved", "not_solved")
    assert tuple(row.reason for row in rows) == (
        "center_too_large",
        "center_too_large",
        "center_too_large",
    )
    assert all(dict(row.metadata)["dense"] == "true" for row in rows)
    assert all(dict(row.metadata)["control"] == "false" for row in rows)


def test_complex_c5_discovered_dense_householder_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="dense-householder",
        max_candidates=3,
    )
    assert tuple(row.verdict for row in rows) == ("not_solved", "not_solved", "not_solved")
    assert tuple(row.center_dimension for row in rows) == (10, 10, 10)
    assert all(dict(row.metadata)["synthetic"] == "false" for row in rows)


def test_complex_c5_discovered_dense_fourier_lite_scan_is_deterministic() -> None:
    rows = run_complex_c5_discovered_split_scan(
        family="dense-fourier-lite",
        max_candidates=3,
    )
    assert tuple(row.verdict for row in rows) == (
        "falsified_no_split",
        "falsified_rank_one_locking",
        "falsified_rank_one_locking",
    )
    assert all(dict(row.metadata)["dense"] == "true" for row in rows)
    assert all(row.verdict != "split_candidate" for row in rows)


def test_complex_c5_discovered_dense_all_summary_reports_no_non_control_split() -> None:
    summary = complex_c5_discovered_summary(
        family="dense-all",
        max_candidates=9,
    )
    assert summary.candidate_count == 9
    assert summary.split_candidate_count == 0
    assert summary.non_synthetic_split_candidate_count == 0
    assert summary.seeded_control_count == 0
