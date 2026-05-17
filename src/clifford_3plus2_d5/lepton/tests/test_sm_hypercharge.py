"""Session 19b tests for the SM hypercharge spectrum."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.continuum import matrix_zero
from clifford_3plus2_d5.lepton.patisalam_sm import (
    b_minus_l_generator_from_su4,
    hypercharge_generator,
    t3_r_generator_from_su2_r,
)
from clifford_3plus2_d5.lepton.sm_hypercharge import (
    EXPECTED_HYPERCHARGE_SPECTRUM,
    EXPECTED_JOINT_Y_T3L_TABLE,
    b_minus_l_observable,
    charge_observables_commute,
    generator_commutes_with_j,
    hypercharge_audit_payload,
    hypercharge_observable,
    hypercharge_spectrum,
    joint_y_t3l_table,
    match_normalization,
    normalized_b_minus_l_generator,
    normalized_hypercharge_spectrum,
    normalized_t3_l_generator,
    normalized_t3_l_observable,
    normalized_t3_r_generator,
    observable_is_symmetric,
    physical_hypercharge_generator,
    raw_hypercharge_observable,
    raw_hypercharge_spectrum,
    sm_field_multiplicity_table,
    t3_l_generator,
    t3_l_observable,
    t3_r_observable,
)


def _commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def test_generators_commute_with_j() -> None:
    assert generator_commutes_with_j(hypercharge_generator())
    assert generator_commutes_with_j(physical_hypercharge_generator())
    assert generator_commutes_with_j(t3_l_generator())
    assert generator_commutes_with_j(t3_r_generator_from_su2_r())
    assert generator_commutes_with_j(b_minus_l_generator_from_su4())


def test_charge_observables_are_symmetric_and_commute() -> None:
    observables = (
        raw_hypercharge_observable(),
        hypercharge_observable(),
        t3_l_observable(),
        normalized_t3_l_observable(),
        t3_r_observable(),
        b_minus_l_observable(),
    )
    assert all(observable_is_symmetric(observable) for observable in observables)
    assert charge_observables_commute()
    assert matrix_zero(_commutator(hypercharge_observable(), normalized_t3_l_observable()))


def test_raw_hypercharge_requires_component_normalization() -> None:
    raw = raw_hypercharge_spectrum()
    matched, scale = match_normalization(raw)
    assert matched is False
    assert scale is None
    assert raw == {
        sp.Rational(7, 4): 1,
        sp.Rational(-1, 4): 1,
        sp.Rational(-3, 4): 2,
        sp.Rational(3, 4): 3,
        sp.Rational(-5, 4): 3,
        sp.Rational(1, 4): 6,
    }


def test_physical_component_normalizations_are_patisalam_standard() -> None:
    assert normalized_t3_r_generator() == (
        sp.Rational(1, 2) * t3_r_generator_from_su2_r()
    ).applyfunc(sp.simplify)
    assert normalized_b_minus_l_generator() == (
        sp.Rational(2, 3) * b_minus_l_generator_from_su4()
    ).applyfunc(sp.simplify)
    assert physical_hypercharge_generator() == (
        normalized_t3_r_generator()
        + sp.Rational(1, 2) * normalized_b_minus_l_generator()
    ).applyfunc(sp.simplify)
    assert normalized_t3_l_generator() == (
        sp.Rational(1, 2) * t3_l_generator()
    ).applyfunc(sp.simplify)


def test_normalized_hypercharge_spectrum_matches_one_generation() -> None:
    matched, scale = match_normalization(hypercharge_spectrum())
    assert matched is True
    assert scale == 1
    assert normalized_hypercharge_spectrum() == EXPECTED_HYPERCHARGE_SPECTRUM


def test_joint_y_t3l_table_matches_one_generation() -> None:
    assert joint_y_t3l_table() == EXPECTED_JOINT_Y_T3L_TABLE
    table = sm_field_multiplicity_table()
    assert table["Q"]["complex_multiplicity"] == 6
    assert table["u^c"]["complex_multiplicity"] == 3
    assert table["d^c"]["complex_multiplicity"] == 3
    assert table["L"]["complex_multiplicity"] == 2
    assert table["e^c"]["complex_multiplicity"] == 1
    assert table["nu^c"]["complex_multiplicity"] == 1


def test_hypercharge_audit_payload_is_stable() -> None:
    payload = hypercharge_audit_payload()
    assert payload["real_dimension"] == 32
    assert payload["complex_dimension"] == 16
    assert payload["y_raw_commutes_with_j"] is True
    assert payload["y_physical_commutes_with_j"] is True
    assert payload["observables_symmetric"] is True
    assert payload["charge_observables_commute"] is True
    assert payload["raw_common_scale_matches_sm"] is False
    assert payload["component_normalization_required"] is True
    assert payload["b_minus_l_normalization_factor"] == sp.Rational(2, 3)
    assert payload["t3_r_normalization_factor"] == sp.Rational(1, 2)
    assert payload["normalization_factor"] == 1
    assert payload["normalized_hypercharge_spectrum"] == EXPECTED_HYPERCHARGE_SPECTRUM
    assert payload["matches_sm_hypercharge_spectrum"] is True
    assert payload["joint_y_t3l_table"] == EXPECTED_JOINT_Y_T3L_TABLE
    assert payload["matches_sm_joint_table"] is True
