"""Session 19a tests for extracting the SM algebra from Pati-Salam."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.clifford_lie import (
    basis_span_dimension,
    matrix_in_span,
)
from clifford_3plus2_d5.lepton.clifford_patisalam import (
    patisalam_all_commute_with_chosen_j,
    su2_l_generators_from_spin04,
    su4_generators_from_spin06,
)
from clifford_3plus2_d5.lepton.continuum import matrix_zero
from clifford_3plus2_d5.lepton.patisalam_sm import (
    b_minus_l_centralizer_in_su4,
    b_minus_l_generator_from_su4,
    hypercharge_generator,
    sm_algebra_audit_payload,
    sm_commutators_pass,
    sm_gauge_generators,
    sm_generator_is_valid,
    su3_c_closes,
    su3_c_generators_from_su4,
    t3_r_generator_from_su2_r,
)


def _commutator(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return (left * right - right * left).applyfunc(sp.simplify)


def test_b_minus_l_has_su3_plus_u1_centralizer() -> None:
    su4 = su4_generators_from_spin06()
    b_minus_l = b_minus_l_generator_from_su4()
    centralizer = b_minus_l_centralizer_in_su4()

    assert sm_generator_is_valid(b_minus_l)
    assert basis_span_dimension(su4) == 15
    assert basis_span_dimension(centralizer) == 9
    assert matrix_in_span(b_minus_l, centralizer)
    assert all(matrix_zero(_commutator(generator, b_minus_l)) for generator in centralizer)


def test_su3_c_is_trace_orthogonal_centralizer_slice() -> None:
    b_minus_l = b_minus_l_generator_from_su4()
    su3 = su3_c_generators_from_su4()

    assert len(su3) == 8
    assert basis_span_dimension(su3) == 8
    assert not matrix_in_span(b_minus_l, su3)
    assert all(sm_generator_is_valid(generator) for generator in su3)
    assert all(matrix_zero(_commutator(generator, b_minus_l)) for generator in su3)
    assert su3_c_closes()


def test_hypercharge_is_patisalam_combination_and_commutes_with_sm_factors() -> None:
    b_minus_l = b_minus_l_generator_from_su4()
    t3_r = t3_r_generator_from_su2_r()
    hypercharge = hypercharge_generator()

    assert hypercharge == (t3_r + sp.Rational(1, 2) * b_minus_l).applyfunc(sp.simplify)
    assert sm_generator_is_valid(hypercharge)
    assert sm_commutators_pass() == {
        "su3_commutes_with_su2_l": True,
        "su3_commutes_with_y": True,
        "su2_l_commutes_with_y": True,
    }


def test_sm_total_dimension_and_j_compatibility() -> None:
    sm = sm_gauge_generators()
    assert len(sm) == 12
    assert basis_span_dimension(sm) == 12
    assert basis_span_dimension(su2_l_generators_from_spin04()) == 3
    assert patisalam_all_commute_with_chosen_j(sm)


def test_sm_algebra_audit_payload_is_stable() -> None:
    payload = sm_algebra_audit_payload()
    assert payload["su4_dimension"] == 15
    assert payload["b_minus_l_centralizer_dimension"] == 9
    assert payload["su3_c_dimension"] == 8
    assert payload["su3_c_closes"] is True
    assert payload["su2_l_dimension"] == 3
    assert payload["u1_y_dimension"] == 1
    assert payload["sm_total_dimension"] == 12
    assert payload["su3_commutes_with_su2_l"] is True
    assert payload["su3_commutes_with_y"] is True
    assert payload["su2_l_commutes_with_y"] is True
    assert payload["chosen_j_commutes_with_sm"] is True
    assert payload["hypercharge_nonzero"] is True
    assert payload["hypercharge_skew_symmetric"] is True
    assert payload["hypercharge_independent_from_su3_su2"] is True
