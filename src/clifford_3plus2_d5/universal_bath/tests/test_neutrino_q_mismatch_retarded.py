"""Tests for Session 12 q-mismatch and retarded active-plane closure."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.universal_bath.neutrino_bcc_moments import (
    bb_edge_blocks,
)
from clifford_3plus2_d5.universal_bath.neutrino_family_port_graph import matrix_equal
from clifford_3plus2_d5.universal_bath.neutrino_q_mismatch_retarded import (
    delta_q,
    finite_gap_feedback_is_scalar,
    hard_gap_feedback_is_zero,
    mixed_normal_directions,
    mixed_normal_schur_feedback,
    q_mismatch_gap,
    q_mismatch_retarded_payload,
    recurrent_two_step_return,
    recurrent_visible_powers_differ,
    retarded_feedback_limit_is_zero,
    retarded_visible_powers_match_survival,
    retarded_weyl_equation_passes,
    retarded_weyl_normalization_passes,
    same_normal_directions,
)


def test_bcc_hops_split_by_relative_depth_charge() -> None:
    """The BCC edge hop set splits into q=0 and q=+-2 sectors."""

    same = same_normal_directions()
    mixed = mixed_normal_directions()

    assert len(same) == 4
    assert len(mixed) == 4
    assert all(delta_q(direction) == 0 for direction in same)
    assert {delta_q(direction) for direction in mixed} == {-2, 2}


def test_q_mismatch_gap_and_schur_feedback() -> None:
    """A q^2 penalty gives scalar finite feedback and zero hard-gap feedback."""

    z = sp.symbols("z")
    g = sp.symbols("g", positive=True)
    feedback = mixed_normal_schur_feedback(z, g)

    assert sp.simplify(q_mismatch_gap(g) - 4 * g) == 0
    assert matrix_equal(feedback, sp.eye(2) / (2 * (z - 4 * g)))
    assert finite_gap_feedback_is_scalar()
    assert hard_gap_feedback_is_zero()


def test_retarded_weyl_branch_and_high_threshold_limit() -> None:
    """The outgoing half-line Weyl branch is normalized and decouples at high gap."""

    assert retarded_weyl_equation_passes()
    assert retarded_weyl_normalization_passes()
    assert retarded_feedback_limit_is_zero()


def test_retarded_compression_and_recurrent_control() -> None:
    """Retarded triangular closure preserves q=0 powers; recurrence does not."""

    assert retarded_visible_powers_match_survival(shells=4, powers=(0, 1, 2, 3))
    assert recurrent_visible_powers_differ(shells=4)

    recurrent_return = recurrent_two_step_return(bb_edge_blocks())
    expected = sp.Matrix(
        [
            [-(1 + sp.I) / 4, 0],
            [0, -(1 - sp.I) / 4],
        ]
    )
    assert matrix_equal(recurrent_return, expected)


def test_session_12_payload() -> None:
    """Session 12 closes the conditional q-mismatch/retarded-compression gate."""

    payload = q_mismatch_retarded_payload()

    assert payload.final_verdict == "NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS"
    assert payload.same_normal_direction_count == 4
    assert payload.mixed_normal_direction_count == 4
    assert payload.total_norm_is_identity
    assert payload.family_radial_projector_matches_incidence
    assert payload.family_active_projector_matches_incidence
    assert payload.hard_gap_feedback_zero
    assert payload.finite_gap_feedback_scalar
    assert payload.retarded_weyl_equation_passes
    assert payload.retarded_weyl_normalization_passes
    assert payload.retarded_feedback_limit_zero
    assert payload.retarded_visible_powers_match_survival
    assert payload.recurrent_wedge_return_nonzero
    assert payload.recurrent_visible_powers_differ
    assert payload.session_10_family_graph_passes
    assert payload.session_11_active_plane_passes
    assert "single_clock_locking_field" in payload.remaining_declared_inputs[0]
