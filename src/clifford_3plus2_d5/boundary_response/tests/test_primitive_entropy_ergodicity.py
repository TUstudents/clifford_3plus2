"""Tests for the V19 primitive max-entropy ergodicity gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.primitive_entropy_ergodicity import (
    compressed_macro_entropy,
    compressed_macro_entropy_derivative,
    compressed_macro_entropy_max_ratio,
    compressed_macro_probabilities,
    primitive_entropy_derivative,
    primitive_entropy_ergodicity_audit_payload,
    primitive_entropy_max_ratio,
    primitive_entropy_probabilities,
    primitive_shannon_entropy,
)
from clifford_3plus2_d5.boundary_response.quark_boundary_shell import (
    quark_boundary_phase_angle,
)
from clifford_3plus2_d5.boundary_response.quark_coin_rigidity import (
    isotropic_quark_phase_angle,
)


def test_primitive_probabilities_normalize_exactly() -> None:
    r = sp.Symbol("r", positive=True)
    probabilities = primitive_entropy_probabilities(r)
    assert len(probabilities) == 6
    assert sp.simplify(sum(probabilities) - 1) == 0


def test_primitive_entropy_has_closed_form_and_derivative() -> None:
    r = sp.Symbol("r", positive=True)
    denominator = 1 + 5 * r**2
    expected_entropy = sp.log(denominator) - (5 * r**2 / denominator) * sp.log(r**2)
    expected_derivative = -10 * r * sp.log(r**2) / denominator**2
    assert sp.simplify(primitive_shannon_entropy(r) - expected_entropy) == 0
    assert sp.simplify(primitive_entropy_derivative(r) - expected_derivative) == 0
    assert sp.simplify(sp.diff(primitive_shannon_entropy(r), r) - expected_derivative) == 0


def test_primitive_entropy_maximum_is_six_channel_uniform_distribution() -> None:
    ratio = primitive_entropy_max_ratio()
    probabilities = primitive_entropy_probabilities(ratio)
    assert ratio == 1
    assert all(sp.simplify(probability - sp.Rational(1, 6)) == 0 for probability in probabilities)
    assert sp.simplify(primitive_shannon_entropy(ratio) - sp.log(6)) == 0
    assert sp.simplify(primitive_entropy_derivative(ratio)) == 0


def test_primitive_entropy_maximum_recovers_ckm_phase() -> None:
    ratio = primitive_entropy_max_ratio()
    assert sp.simplify(isotropic_quark_phase_angle(ratio) - quark_boundary_phase_angle()) == 0


def test_compressed_macro_entropy_control_maximizes_at_normalized_odd_collective() -> None:
    r = sp.Symbol("r", positive=True)
    ratio = compressed_macro_entropy_max_ratio()
    probabilities = compressed_macro_probabilities(ratio)
    assert sp.simplify(ratio - 1 / sp.sqrt(5)) == 0
    assert all(sp.simplify(probability - sp.Rational(1, 2)) == 0 for probability in probabilities)
    assert sp.simplify(compressed_macro_entropy(ratio) - sp.log(2)) == 0
    assert sp.simplify(compressed_macro_entropy_derivative(ratio)) == 0
    assert sp.simplify(
        sp.diff(compressed_macro_entropy(r), r) - compressed_macro_entropy_derivative(r)
    ) == 0


def test_compressed_macro_entropy_control_gives_pi_over_four_not_ckm() -> None:
    phase = isotropic_quark_phase_angle(compressed_macro_entropy_max_ratio())
    assert sp.simplify(phase - sp.pi / 4) == 0
    assert sp.simplify(phase - quark_boundary_phase_angle()) != 0


def test_primitive_entropy_payload_reports_conditional_pass() -> None:
    payload = primitive_entropy_ergodicity_audit_payload()
    assert payload.final_verdict == "MAX_ENTROPY_PRIMITIVE_ERGODICITY_CONDITIONAL_PASS"
    assert payload.primitive_max_ratio == 1
    assert sp.simplify(payload.primitive_max_phase - quark_boundary_phase_angle()) == 0
    assert sp.simplify(payload.compressed_max_ratio - 1 / sp.sqrt(5)) == 0
    assert sp.simplify(payload.compressed_max_phase - sp.pi / 4) == 0
    assert payload.ckm_phase_recovered
    assert payload.compressed_control_rejected
    assert payload.entropy_partition_is_extra_principle
