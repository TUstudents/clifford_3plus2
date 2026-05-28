"""Tests for the V13 quark color/BCC Clebsch gate."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.boundary_response.quark_clebsch_factors import (
    bcc_antisymmetric_projection_factor,
    bcc_symmetric_two_path_factor,
    color_return_contraction,
    color_return_factor,
    missing_color_generator_control,
    normalized_two_step_leakage,
    quark_clebsch_audit_payload,
    quark_prefactor_summary,
    su3_fundamental_generators,
)
from clifford_3plus2_d5.boundary_response.quark_transfer_hierarchy import (
    quark_transfer_hierarchy_audit_payload,
)
from clifford_3plus2_d5.boundary_response.transfer import epsilon


def _assert_matrix_equal(left: sp.Matrix, right: sp.Matrix) -> None:
    residual = left - right
    assert all(sp.simplify(entry) == 0 for entry in residual)


def test_su3_fundamental_generators_are_hermitian_and_traceless() -> None:
    generators = su3_fundamental_generators()
    assert len(generators) == 8
    for generator in generators:
        _assert_matrix_equal(generator.conjugate().T, generator)
        assert sp.simplify(generator.trace()) == 0


def test_color_return_contraction_is_fundamental_casimir() -> None:
    _assert_matrix_equal(color_return_contraction(), sp.Rational(4, 3) * sp.eye(3))
    assert color_return_factor() == sp.Rational(4, 3)


def test_normalized_two_step_leakage_matches_q3_formula() -> None:
    expected = epsilon() ** 2 / sp.sqrt(1 + epsilon() ** 4)
    assert sp.simplify(normalized_two_step_leakage() - expected) == 0


def test_bcc_two_path_clebsches_are_exact() -> None:
    assert bcc_symmetric_two_path_factor() == sp.sqrt(2)
    assert bcc_antisymmetric_projection_factor() == 1 / sp.sqrt(2)


def test_q3_negative_controls_are_rejected() -> None:
    raw_contraction = color_return_contraction(su3_fundamental_generators(normalized=False))
    assert raw_contraction != sp.Rational(4, 3) * sp.eye(3)

    missing_contraction = color_return_contraction(missing_color_generator_control())
    assert missing_contraction != sp.Rational(4, 3) * sp.eye(3)

    assert sp.simplify(bcc_symmetric_two_path_factor(coherent=False) - sp.sqrt(2)) != 0
    assert sp.simplify(bcc_antisymmetric_projection_factor(normalized=False) - 1 / sp.sqrt(2)) != 0


def test_quark_prefactor_summary_reports_exact_values() -> None:
    summary = quark_prefactor_summary()
    assert summary.color_return_factor == sp.Rational(4, 3)
    assert sp.simplify(
        summary.normalized_cabibbo_leakage
        - epsilon() ** 2 / sp.sqrt(1 + epsilon() ** 4)
    ) == 0
    assert summary.symmetric_bcc_factor == sp.sqrt(2)
    assert summary.antisymmetric_bcc_factor == 1 / sp.sqrt(2)


def test_quark_clebsch_payload_reports_q3_pass() -> None:
    payload = quark_clebsch_audit_payload()
    assert payload.final_verdict == "QUARK_CLEBSCH_Q3_PASS"
    assert payload.color_contraction_matches
    assert payload.raw_generator_control_rejected
    assert payload.missing_generator_control_rejected
    assert payload.incoherent_bcc_control_rejected
    assert payload.unnormalized_antisymmetric_control_rejected
    assert payload.ckm_parked


def test_v12_regression_remains_stable() -> None:
    assert (
        quark_transfer_hierarchy_audit_payload().final_verdict
        == "QUARK_TRANSFER_HIERARCHY_Q2_PASS"
    )
