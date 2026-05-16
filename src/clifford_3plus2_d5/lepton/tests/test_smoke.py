"""Smoke tests for the lepton laboratory.

These tests live inside the lepton package by design. Run them directly with:

    uv run pytest src/clifford_3plus2_d5/lepton/tests -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import sympy as sp

from clifford_3plus2_d5.lepton import (
    audit_first_domain_wall_candidate,
    audit_first_physical_domain_wall_candidate,
    lab_a_carrier_check_passed,
    lab_b_carrier_check_passed,
    lab_b_physical_wall_carrier_check_passed,
    lab_b_singlet_doublet_projectors_for_mode,
    iter_lab_b_domain_wall_candidates,
    iter_lab_b_physical_domain_wall_candidates,
    lab_b_domain_wall_context,
    run_lab_b_domain_wall_scan,
    run_lab_b_physical_domain_wall_scan,
    run_lab_b_strict_scan,
    run_lab_b_structural_wall_scan,
    su2_l_u1_y_generators_r6_for_singlet,
)


def test_carrier_checks_pass() -> None:
    assert lab_a_carrier_check_passed()
    assert lab_b_carrier_check_passed()
    assert lab_b_physical_wall_carrier_check_passed()


def test_lab_b_split_frames_have_expected_shape() -> None:
    for singlet_mode in range(3):
        projectors = lab_b_singlet_doublet_projectors_for_mode(singlet_mode)
        assert tuple(projector.rank() for projector in projectors) == (2, 4)
        assert len(su2_l_u1_y_generators_r6_for_singlet(singlet_mode)) == 4


def test_lab_b_strict_reproduces_multiple_j_outcome() -> None:
    [row] = run_lab_b_strict_scan(max_candidates=1)
    assert row.verdict == "candidate_only_j_not_forced"
    assert row.generated_algebra_dimension == 10
    assert row.center_dimension == 4
    assert row.central_j_candidate_count == 4
    assert row.commutant_verdict == "passed_multiple_aligned"
    assert not row.load_bearing_qca_bridge


def test_lab_b_structural_wall_reproduces_multiple_j_outcome() -> None:
    [row] = run_lab_b_structural_wall_scan(max_candidates=1, max_pairs=4)
    assert row.verdict == "candidate_only_j_not_forced"
    assert row.center_dimension == 4
    assert row.central_j_candidate_count == 4
    assert dict(row.metadata)["transition_tier"] == "split_reassignment"


def test_lab_b_domain_wall_fast_positive_is_not_global_bridge() -> None:
    [row] = run_lab_b_domain_wall_scan(
        max_candidates=1,
        max_pairs=1,
        verify_center_exact=False,
    )
    assert row.verdict == "domain_wall_candidate"
    assert row.center_dimension == 2
    assert row.central_j_candidate_count == 2
    assert row.load_bearing_domain_wall_candidate
    assert not row.load_bearing_qca_bridge


def test_lab_b_domain_wall_checks_independent_right_frame() -> None:
    candidate = next(iter_lab_b_domain_wall_candidates(max_candidates=1, max_pairs=1))
    assert candidate.wall_context is not None
    assert candidate.wall_context.consistency_certified()
    assert dict(candidate.metadata)["right_singlet_mode"] == "1"
    with pytest.raises(ValueError, match="declared right frame"):
        lab_b_domain_wall_context(
            transition=candidate.wall_context.transition,
            left_singlet_mode=candidate.left_singlet_mode,
            right_singlet_mode=2,
        )


def test_lab_b_physical_domain_wall_uses_complementary_site_projectors() -> None:
    candidate = next(iter_lab_b_physical_domain_wall_candidates(max_candidates=1, max_pairs=1))
    assert candidate.wall_context is not None
    assert candidate.wall_context.side_projectors_must_be_complementary
    assert candidate.wall_context.consistency_certified()
    bad_transition_context = replace(candidate.wall_context, transition=2 * sp.eye(12))
    assert not bad_transition_context.consistency_certified()
    bad_projector_context = replace(
        candidate.wall_context,
        right_side_projector=candidate.wall_context.left_side_projector,
    )
    assert not bad_projector_context.consistency_certified()
    [row] = run_lab_b_physical_domain_wall_scan(max_candidates=1, max_pairs=1)
    assert row.verdict == "domain_wall_candidate"
    assert row.center_dimension == 2
    assert row.central_j_candidate_count == 2
    assert row.load_bearing_domain_wall_candidate
    assert not row.load_bearing_qca_bridge
    assert dict(row.metadata)["domain_model"] == "two_site_r12"
    assert dict(row.metadata)["center_contract"] == "known_center_only"


def test_first_physical_domain_wall_bounded_audit() -> None:
    audit = audit_first_physical_domain_wall_candidate(max_pairs=1, timeout_seconds=1)
    assert audit.wall_context_consistent
    assert audit.generated_algebra_closed
    assert audit.verdict == "domain_wall_candidate"
    assert audit.central_j_candidate_count == 2
    assert audit.load_bearing_domain_wall_candidate
    assert not audit.load_bearing_qca_bridge
    assert audit.exact_center_status in {
        "passed_exact",
        "failed",
        "not_solved_timeout",
        "not_solved_no_result",
        "not_solved_error",
    }
    if audit.exact_center_status == "passed_exact":
        assert audit.actual_center_dimension == 2
        assert audit.center_idempotents_solved
        assert audit.central_idempotent_ranks == (0, 12)


def test_first_domain_wall_exact_audit() -> None:
    audit = audit_first_domain_wall_candidate(max_pairs=1)
    assert audit.wall_context_consistent
    assert audit.generated_algebra_dimension == 18
    assert audit.generated_algebra_closed
    assert audit.actual_center_dimension == 2
    assert audit.center_idempotents_solved
    assert audit.central_idempotent_ranks == (0, 6)
    assert audit.verdict == "domain_wall_candidate"
    assert audit.central_j_candidate_count == 2
    assert audit.load_bearing_domain_wall_candidate
    assert not audit.load_bearing_qca_bridge
