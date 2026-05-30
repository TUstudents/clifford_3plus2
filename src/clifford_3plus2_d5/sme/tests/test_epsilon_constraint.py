"""Tests for ``epsilon_constraint.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.sme.epsilon_constraint import (
    DIM5_FERMION_BOUND_GEV_INVERSE,
    GEV_INVERSE_TO_METRES,
    OBSERVABLE_THRESHOLD_METRES,
    PLANCK_LENGTH_METRES,
    dim5_fermion_bound_in_metres,
    epsilon_bound_per_component,
    epsilon_bound_tightest_face,
    epsilon_constraint_payload,
    scale_verdict,
)
from clifford_3plus2_d5.sme.sme_tensor_mapping import t2g_tensor_entries


def test_planck_length_constant() -> None:
    # CODATA Planck length ~ 1.6 × 10⁻³⁵ m, within 1%.
    assert abs(float(PLANCK_LENGTH_METRES) - 1.616e-35) < 1e-37


def test_gev_inverse_to_metres_constant() -> None:
    # 1 GeV⁻¹ ≈ 1.97 × 10⁻¹⁶ m via ℏc.
    assert abs(float(GEV_INVERSE_TO_METRES) - 1.973e-16) < 1e-18


def test_dim5_bound_in_metres_is_product() -> None:
    # bound_m should equal bound_GeV⁻¹ × GeV⁻¹→m conversion.
    bound_m = dim5_fermion_bound_in_metres()
    expected = float(DIM5_FERMION_BOUND_GEV_INVERSE) * float(GEV_INVERSE_TO_METRES)
    assert abs(float(bound_m) - expected) / expected < 1e-6


def test_per_component_bound_for_unit_coefficient() -> None:
    entries = t2g_tensor_entries()
    bound_m = float(dim5_fermion_bound_in_metres())
    for entry in entries:
        result = float(epsilon_bound_per_component(entry))
        # |coefficient| = 1, so ε bound = bound_m exactly.
        assert abs(result - bound_m) / bound_m < 1e-6


def test_tightest_face_equals_single_bound_when_coefficients_unit() -> None:
    # All three T_{2g} coefficients have magnitude 1; tightest equals single.
    tight = float(epsilon_bound_tightest_face())
    single = float(dim5_fermion_bound_in_metres())
    assert abs(tight - single) / single < 1e-6


def test_scale_verdict_sub_planck() -> None:
    sub_planck = PLANCK_LENGTH_METRES * sp.Rational(1, 2)
    assert scale_verdict(sub_planck) == "SUB-PLANCK KILL"


def test_scale_verdict_planck_consistent() -> None:
    near_planck = PLANCK_LENGTH_METRES * sp.Integer(5)
    assert scale_verdict(near_planck) == "PLANCK-CONSISTENT"


def test_scale_verdict_unfalsifiable() -> None:
    # 10⁻³³ m is ~ 100 ℓ_P, above 10 ℓ_P and below 10⁻²⁵ m.
    middle = sp.Float("1e-33")
    assert scale_verdict(middle) == "UNFALSIFIABLE PASS"


def test_scale_verdict_observable() -> None:
    observable = OBSERVABLE_THRESHOLD_METRES * sp.Integer(100)
    assert scale_verdict(observable) == "OBSERVABLE POSITIVE"


def test_payload_verdict_is_unfalsifiable_pass() -> None:
    payload = epsilon_constraint_payload()
    assert payload.scale_verdict == "UNFALSIFIABLE PASS"


def test_payload_bound_is_about_two_e_minus_33() -> None:
    payload = epsilon_constraint_payload()
    bound = float(payload.epsilon_bound_metres)
    # Expected ~ 1.97 × 10⁻³³ m.
    assert 1e-33 < bound < 1e-32


def test_payload_orders_above_planck_is_about_two() -> None:
    payload = epsilon_constraint_payload()
    orders = float(payload.epsilon_bound_orders_above_planck)
    # log10(2e-33 / 1.6e-35) ≈ 2.09.
    assert 1.5 < orders < 2.5


def test_payload_carries_explanation_string() -> None:
    payload = epsilon_constraint_payload()
    assert isinstance(payload.verdict_class_explanation, str)
    assert len(payload.verdict_class_explanation) > 50


def test_payload_flags_representative_bound_until_kr_verified() -> None:
    payload = epsilon_constraint_payload()
    # The bound is the representative 1e-17 GeV⁻¹; KR entry-ids unverified.
    assert payload.kr_entry_ids_verified is False
    assert payload.bound_is_representative is True
