"""Tests for the V29 unit outward-continuation normalization gate."""

from __future__ import annotations

from itertools import permutations

import pytest
import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import epsilon, transfer_matrix
from clifford_3plus2_d5.boundary_response.unit_continuation import (
    REMAINING_DECLARED_INPUTS_AFTER_UNIT_CONTINUATION,
    anisotropic_outward_matching,
    continuation_gram,
    decaying_factor_from_matching,
    double_outward_matching,
    residual_basis_continuation_couplings,
    residual_basis_continuation_matrix,
    residual_permutation_matching,
    residual_permutation_matchings_are_unit_gauge_equivalent,
    scaled_outward_matching,
    transfer_matrix_from_matching,
    uniform_continuation_coefficient,
    unit_continuation_audit_payload,
    unit_outward_matching,
)


def test_unit_outward_matching_has_identity_gram() -> None:
    matching = unit_outward_matching()
    assert matching == sp.eye(3)
    assert continuation_gram(matching) == sp.eye(3)
    assert uniform_continuation_coefficient(matching) == sp.Integer(1)


def test_unit_matching_has_unit_couplings_in_residual_basis() -> None:
    matching = unit_outward_matching()
    assert residual_basis_continuation_matrix(matching) == sp.eye(3)
    assert residual_basis_continuation_couplings(matching) == (
        sp.Integer(1),
        sp.Integer(1),
        sp.Integer(1),
    )


def test_unit_matching_recovers_v26_transfer_matrix_and_epsilon() -> None:
    matching = unit_outward_matching()
    assert transfer_matrix_from_matching(matching) == transfer_matrix()
    assert sp.simplify(decaying_factor_from_matching(matching) - epsilon()) == 0


def test_scaled_matching_changes_transfer_root() -> None:
    matching = scaled_outward_matching(2)
    assert continuation_gram(matching) == 4 * sp.eye(3)
    assert uniform_continuation_coefficient(matching) == sp.Integer(4)
    assert sp.simplify(decaying_factor_from_matching(matching) - epsilon()) != 0


def test_double_outward_continuation_changes_norm_and_root() -> None:
    matching = double_outward_matching()
    assert matching.shape == (6, 3)
    assert continuation_gram(matching) == 2 * sp.eye(3)
    assert uniform_continuation_coefficient(matching) == sp.Integer(2)
    assert sp.simplify(decaying_factor_from_matching(matching) - epsilon()) != 0


def test_anisotropic_matching_is_rejected_as_scalar_quotient() -> None:
    matching = anisotropic_outward_matching()
    assert uniform_continuation_coefficient(matching) is None
    with pytest.raises(ValueError, match="scalar continuation quotient"):
        transfer_matrix_from_matching(matching)


def test_residual_label_permutations_are_gauge_equivalent_unit_matchings() -> None:
    assert residual_permutation_matchings_are_unit_gauge_equivalent()
    for perm in permutations((0, 1, 2)):
        matching = residual_permutation_matching(tuple(perm))
        assert continuation_gram(matching) == sp.eye(3)
        assert uniform_continuation_coefficient(matching) == sp.Integer(1)
        assert sp.simplify(decaying_factor_from_matching(matching) - epsilon()) == 0


def test_invalid_matching_controls_are_not_unit_gauge_equivalent() -> None:
    assert uniform_continuation_coefficient(scaled_outward_matching(3)) != 1
    assert uniform_continuation_coefficient(double_outward_matching()) != 1
    assert uniform_continuation_coefficient(anisotropic_outward_matching()) is None


def test_unit_continuation_payload_reports_pass() -> None:
    payload = unit_continuation_audit_payload()
    assert payload.final_verdict == "UNIT_OUTWARD_CONTINUATION_NORMALIZATION_PASS"
    assert payload.matching_shape == (3, 3)
    assert payload.continuation_gram_matrix == sp.eye(3)
    assert payload.residual_basis_couplings == (1, 1, 1)
    assert payload.uniform_continuation_coefficient == 1
    assert payload.derived_transfer_matrix == transfer_matrix()
    assert sp.simplify(payload.derived_decaying_factor - epsilon()) == 0
    assert payload.scaled_control_rejected
    assert payload.double_control_coefficient == 2
    assert payload.double_control_rejected
    assert payload.anisotropic_control_rejected
    assert payload.permutation_matchings_gauge_equivalent
    assert payload.v26_recovered
    assert (
        payload.remaining_declared_inputs
        == REMAINING_DECLARED_INPUTS_AFTER_UNIT_CONTINUATION
    )
    assert payload.remaining_declared_inputs == (
        "physical_vacuum_order_parameter_exists",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
