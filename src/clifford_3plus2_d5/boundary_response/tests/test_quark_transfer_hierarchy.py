"""Tests for the V12 quark transfer-depth hierarchy gate."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_shell_audit_payload,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    EXPECTED_TRANSITION_DEPTHS,
    matches_ckm_depth_ordering,
    nonadditive_transition_depth_control,
    odd_depth_family_embedding_control,
    permuted_family_embedding_control,
    quark_family_depths,
    quark_transfer_hierarchy_audit_payload,
    quark_transition_amplitude,
    quark_transition_depths,
    transition_depths_are_additive,
    transition_depths_are_even,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def test_quark_family_depth_embedding_is_exact() -> None:
    assert quark_family_depths() == {1: 0, 2: 2, 3: 6}


def test_quark_transition_depths_match_q2_hierarchy() -> None:
    assert quark_transition_depths() == EXPECTED_TRANSITION_DEPTHS


def test_quark_transition_amplitudes_are_epsilon_powers() -> None:
    assert sp.simplify(quark_transition_amplitude(1, 2) - epsilon() ** 2) == 0
    assert sp.simplify(quark_transition_amplitude(2, 3) - epsilon() ** 4) == 0
    assert sp.simplify(quark_transition_amplitude(1, 3) - epsilon() ** 6) == 0
    assert sp.simplify(quark_transition_amplitude(3, 1) - epsilon() ** 6) == 0


def test_invalid_quark_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        quark_transition_amplitude(1, 1)
    with pytest.raises(ValueError, match="unknown"):
        quark_transition_amplitude(1, 4)


def test_transition_depth_controls() -> None:
    depths = quark_transition_depths()
    assert transition_depths_are_even(depths)
    assert transition_depths_are_additive(depths)
    assert matches_ckm_depth_ordering(depths)

    odd_depths = quark_transition_depths(odd_depth_family_embedding_control())
    assert not transition_depths_are_even(odd_depths)

    assert not transition_depths_are_additive(nonadditive_transition_depth_control())

    permuted_depths = quark_transition_depths(permuted_family_embedding_control())
    assert not matches_ckm_depth_ordering(permuted_depths)


def test_quark_transfer_hierarchy_payload_reports_q2_pass() -> None:
    payload = quark_transfer_hierarchy_audit_payload()
    assert payload.final_verdict == "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
    assert payload.family_depths == {1: 0, 2: 2, 3: 6}
    assert payload.transition_depths == EXPECTED_TRANSITION_DEPTHS
    assert sp.simplify(payload.transition_amplitudes[(1, 2)] - epsilon() ** 2) == 0
    assert sp.simplify(payload.transition_amplitudes[(2, 3)] - epsilon() ** 4) == 0
    assert sp.simplify(payload.transition_amplitudes[(1, 3)] - epsilon() ** 6) == 0
    assert payload.even_depths
    assert payload.additive_depths
    assert payload.ckm_ordering_matches
    assert payload.odd_depth_control_rejected
    assert payload.nonadditive_control_rejected
    assert payload.permuted_label_control_rejected
    assert payload.ckm_parked


def test_v11_regression_remains_stable() -> None:
    assert quark_boundary_shell_audit_payload().final_verdict == "QUARK_BOUNDARY_SHELL_Q1_PASS"
