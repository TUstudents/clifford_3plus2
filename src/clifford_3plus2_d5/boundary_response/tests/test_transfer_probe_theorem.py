"""Tests for the V25 transfer-probe compatibility theorem."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    transfer_polynomial,
)
from clifford_3plus2_d5.boundary_response.transfer_probe_theorem import (
    REMAINING_DECLARED_INPUTS_AFTER_TRANSFER_PROBE,
    probe_from_transfer_factor,
    reciprocal_branch_rejected,
    reciprocal_transfer_branch,
    scaled_probe_control_residual,
    scaled_probe_from_transfer_factor,
    transfer_factor_from_probe,
    transfer_probe_theorem_audit_payload,
    transfer_probe_uniqueness_residual,
)
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    weyl_fixed_point_residual,
)


def test_probe_from_epsilon_is_existing_transfer_probe() -> None:
    assert sp.simplify(probe_from_transfer_factor(epsilon()) - 2 * sp.sqrt(2)) == 0
    assert sp.simplify(probe_from_transfer_factor(epsilon()) - transfer_probe()) == 0


def test_epsilon_satisfies_residual_transfer_recurrence() -> None:
    assert sp.simplify(transfer_polynomial(epsilon())) == 0


def test_weyl_function_at_derived_probe_returns_epsilon() -> None:
    probe = probe_from_transfer_factor(epsilon())
    assert sp.simplify(transfer_factor_from_probe(probe) - epsilon()) == 0
    assert weyl_fixed_point_residual(probe) == 0


def test_reciprocal_branch_is_non_decaying_and_rejected() -> None:
    reciprocal = reciprocal_transfer_branch(epsilon())
    assert sp.simplify(reciprocal - (1 + sp.sqrt(2))) == 0
    assert bool(sp.N(reciprocal) > 1)
    assert reciprocal_branch_rejected(epsilon())


def test_probe_is_unique_for_fixed_decaying_transfer_factor() -> None:
    z = sp.symbols("z")
    residual = transfer_probe_uniqueness_residual(epsilon(), z)
    assert sp.simplify(residual - (z - 2 * sp.sqrt(2))) == 0
    assert sp.solve(sp.Eq(residual, 0), z) == [2 * sp.sqrt(2)]


def test_scaled_chain_control_keeps_unit_normalization_visible() -> None:
    hopping = sp.symbols("t", positive=True)
    scaled_probe = scaled_probe_from_transfer_factor(epsilon(), hopping)
    assert sp.simplify(scaled_probe - hopping * 2 * sp.sqrt(2)) == 0
    assert sp.simplify(
        scaled_probe_control_residual(epsilon(), hopping)
        - (hopping - 1) * 2 * sp.sqrt(2)
    ) == 0


def test_transfer_probe_theorem_payload_reports_pass() -> None:
    payload = transfer_probe_theorem_audit_payload()
    assert payload.final_verdict == "TRANSFER_PROBE_COMPATIBILITY_PASS"
    assert payload.recurrence_residual == 0
    assert sp.simplify(payload.derived_probe - transfer_probe()) == 0
    assert sp.simplify(payload.weyl_value_at_probe - epsilon()) == 0
    assert payload.fixed_point_residual_at_probe == 0
    assert sp.simplify(payload.reciprocal_branch - (1 + sp.sqrt(2))) == 0
    assert payload.reciprocal_branch_rejected
    assert sp.simplify(payload.symbolic_uniqueness_residual.subs("z", payload.derived_probe)) == 0
    assert payload.unit_normalization_load_bearing
    assert payload.remaining_declared_inputs == REMAINING_DECLARED_INPUTS_AFTER_TRANSFER_PROBE
    assert payload.remaining_declared_inputs == (
        "vacuum_framing",
        "unit_sterile_chain_normalization",
        "regular_boundary_fiber_or_max_entropy_prior",
    )
