"""Tests for ``spin8_triality.py`` — Spin(8) embedding and triality Cartan."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.triality.spin8_triality import (
    apply_triality_to_cartan_vector,
    cartan_coordinates,
    is_in_cartan_span,
    spin8_bivector_indices,
    spin8_cartan_mutually_commute,
    spin8_cartan_on_chiral16,
    spin8_cartan_pairs,
    spin8_generator_on_chiral16,
    spin8_generators_on_chiral16,
    triality_cartan_fixes_only_diagonal,
    triality_cartan_is_not_identity,
    triality_cartan_is_order_three,
    triality_cartan_is_orthogonal,
    triality_cartan_matrix,
)


def test_spin8_has_exactly_28_generators() -> None:
    assert len(spin8_bivector_indices()) == 28
    generators = spin8_generators_on_chiral16()
    assert len(generators) == 28
    assert all(generator.shape == (32, 32) for generator in generators)


def test_spin8_generators_are_skew_symmetric_on_chiral16() -> None:
    for generator in spin8_generators_on_chiral16():
        assert (generator + generator.T).applyfunc(sp.simplify) == sp.zeros(32)


def test_spin8_cartan_has_four_mutually_commuting_elements() -> None:
    cartan = spin8_cartan_on_chiral16()
    assert len(cartan) == 4
    assert spin8_cartan_pairs() == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert spin8_cartan_mutually_commute()


def test_triality_cartan_matrix_is_orthogonal_order_three_nonidentity() -> None:
    matrix = triality_cartan_matrix()
    assert matrix.shape == (4, 4)
    assert triality_cartan_is_orthogonal()
    assert triality_cartan_is_order_three()
    assert triality_cartan_is_not_identity()
    assert matrix.det() == 1


def test_triality_cartan_fixed_subspace_has_expected_dimension() -> None:
    # A non-identity orthogonal order-3 map on R^4 has eigenvalues
    # 1, 1, omega, omega-bar.  The real +1 eigenspace is 2-dimensional.
    assert triality_cartan_fixes_only_diagonal() == 2


def test_apply_triality_to_cartan_vector_is_linear() -> None:
    u = sp.Matrix(4, 1, [sp.Rational(1, 3), sp.Integer(0), sp.Integer(-2), sp.Rational(5, 7)])
    v = sp.Matrix(4, 1, [sp.Integer(1), sp.Rational(-1, 2), sp.Integer(0), sp.Integer(4)])
    image_u = apply_triality_to_cartan_vector(u)
    image_v = apply_triality_to_cartan_vector(v)
    image_sum = apply_triality_to_cartan_vector((u + sp.Integer(3) * v).applyfunc(sp.simplify))
    expected = (image_u + sp.Integer(3) * image_v).applyfunc(sp.simplify)
    assert (image_sum - expected).applyfunc(sp.simplify) == sp.zeros(4, 1)


def test_apply_triality_to_cartan_vector_cycles_unit_vectors() -> None:
    e0 = sp.Matrix(4, 1, [1, 0, 0, 0])
    image_e0 = apply_triality_to_cartan_vector(e0)
    image2_e0 = apply_triality_to_cartan_vector(image_e0)
    image3_e0 = apply_triality_to_cartan_vector(image2_e0)
    assert (image3_e0 - e0).applyfunc(sp.simplify) == sp.zeros(4, 1)
    # tau is not identity even on a single basis vector
    assert (image_e0 - e0).applyfunc(sp.simplify) != sp.zeros(4, 1)


def test_cartan_coordinates_round_trip_on_each_basis_element() -> None:
    cartan = spin8_cartan_on_chiral16()
    for index, generator in enumerate(cartan):
        coordinates = cartan_coordinates(generator)
        assert coordinates is not None
        expected = sp.Matrix(4, 1, [sp.Integer(0)] * 4)
        expected[index] = sp.Integer(1)
        assert (coordinates - expected).applyfunc(sp.simplify) == sp.zeros(4, 1)


def test_cartan_coordinates_rejects_non_cartan_generators() -> None:
    # (0, 2) is a Spin(8) generator but not in the Cartan span.
    off_cartan = spin8_generator_on_chiral16(0, 2)
    assert cartan_coordinates(off_cartan) is None
    assert not is_in_cartan_span(off_cartan)
