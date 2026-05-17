from __future__ import annotations

import sympy as sp
import pytest

from clifford_3plus2_d5.algebra.commutants import (
    IncrementalMatrixSpan,
    RationalMatrixSpan,
    Sqrt3IMatrixSpan,
    commutes_with_sm_gauge,
    complex_from_real,
    is_complex_block_scalar_3plus2,
    is_complex_linear_real,
    is_sm_commutant_matrix,
    matrix_span_rank,
    realify,
    safe_commutant_basis,
    safe_commutant_closure_examples_pass,
    safe_commutant_element,
    safe_real_commutant_basis,
    sm_commutant_basis_from_linear_system,
    sm_commutant_basis_matches_expected,
)
from clifford_3plus2_d5.algebra.real_carrier import standard_real_carrier
from clifford_3plus2_d5.obstruction_r10.sm.embedding import (
    complex_projector_2,
    complex_projector_3,
    hypercharge_generator,
    matrix_unit,
    sm_gauge_generators,
    su2_generators,
    su3_generators,
)


def test_sm_generators_have_exact_expected_shapes() -> None:
    generators = sm_gauge_generators()

    assert len(su3_generators()) == 6
    assert len(su2_generators()) == 3
    assert len(generators) == 10
    assert all(generator.matrix.shape == (5, 5) for generator in generators)
    assert hypercharge_generator().matrix == sp.diag(
        sp.Rational(-1, 3),
        sp.Rational(-1, 3),
        sp.Rational(-1, 3),
        sp.Rational(1, 2),
        sp.Rational(1, 2),
    )


def test_safe_commutant_basis_is_projector_pair() -> None:
    p3, p2 = safe_commutant_basis()

    assert p3 == complex_projector_3()
    assert p2 == complex_projector_2()
    assert p3 + p2 == sp.eye(5)
    assert p3 * p2 == sp.zeros(5)
    assert commutes_with_sm_gauge(p3)
    assert commutes_with_sm_gauge(p2)


def test_linear_system_commutant_basis_matches_projector_span() -> None:
    computed = sm_commutant_basis_from_linear_system()
    expected = safe_commutant_basis()

    assert matrix_span_rank(computed) == 2
    assert matrix_span_rank(computed + expected) == 2
    assert sm_commutant_basis_matches_expected()


def test_incremental_matrix_span_tracks_rank_and_coordinates() -> None:
    first = sp.Matrix([[1, 0], [0, 0]])
    second = sp.Matrix([[0, 1], [0, 0]])
    dependent = 2 * first - second
    span = IncrementalMatrixSpan(rows=2, cols=2)

    assert span.add(first)
    assert span.add(second)
    assert not span.add(dependent)
    assert span.rank == 2
    assert span.coordinates(dependent) == (2, -1)
    assert span.contains(first + second)
    assert not span.contains(sp.Matrix([[0, 0], [1, 0]]))


def test_rational_matrix_span_tracks_sparse_rank_and_coordinates() -> None:
    first = sp.Matrix([[1, 0], [0, 0]])
    second = sp.Matrix([[0, 1], [0, 0]])
    dependent = sp.Rational(3, 2) * first - sp.Rational(1, 2) * second
    span = RationalMatrixSpan(rows=2, cols=2)

    assert RationalMatrixSpan.supports(dependent)
    assert span.add(first)
    assert span.add(second)
    assert not span.add(dependent)
    assert span.rank == 2
    assert span.coordinates(dependent) == (sp.Rational(3, 2), sp.Rational(-1, 2))
    assert span.contains(first + second)
    assert not span.contains(sp.Matrix([[0, 0], [1, 0]]))


def test_sqrt3_i_matrix_span_tracks_field_coefficients() -> None:
    first = sp.Matrix([[1, 0], [0, 0]])
    second = sp.Matrix([[sp.sqrt(3), 0], [0, 0]])
    third = sp.Matrix([[sp.I, 0], [0, 0]])
    outside = sp.Matrix([[0, 1], [0, 0]])
    span = Sqrt3IMatrixSpan(rows=2, cols=2)

    assert span.add(first)
    assert not span.add(second)
    assert not span.add(third)
    assert span.rank == 1
    assert span.coordinates(second) == (sp.sqrt(3),)
    assert span.coordinates(third) == (sp.I,)
    assert not span.contains(outside)


def test_block_scalar_commutes_with_sm_gauge() -> None:
    matrix = safe_commutant_element(sp.Rational(2), sp.I + sp.Rational(1, 2))

    assert is_complex_block_scalar_3plus2(matrix)
    assert is_sm_commutant_matrix(matrix)


def test_rank_one_color_projector_is_not_in_commutant() -> None:
    projector = sp.diag(1, 0, 0, 0, 0)

    assert not is_complex_block_scalar_3plus2(projector)
    assert not commutes_with_sm_gauge(projector)


def test_rank_one_weak_projector_is_not_in_commutant() -> None:
    projector = sp.diag(0, 0, 0, 1, 0)

    assert not is_complex_block_scalar_3plus2(projector)
    assert not commutes_with_sm_gauge(projector)


def test_block_mixer_is_not_in_commutant() -> None:
    mixer = matrix_unit(0, 3) + matrix_unit(3, 0)

    assert not is_complex_block_scalar_3plus2(mixer)
    assert not commutes_with_sm_gauge(mixer)


def test_realification_matches_phase_1_complex_structure_and_projectors() -> None:
    carrier = standard_real_carrier()

    assert realify(sp.I * sp.eye(5)) == carrier.complex_structure
    assert realify(complex_projector_3()) == carrier.projector_3
    assert realify(complex_projector_2()) == carrier.projector_2


def test_complex_from_real_round_trips_exact_complex_matrix() -> None:
    matrix = safe_commutant_element(sp.I, sp.Rational(3, 2))
    real_matrix = realify(matrix)

    assert is_complex_linear_real(real_matrix)
    assert complex_from_real(real_matrix) == matrix


def test_real_conjugation_is_not_complex_linear() -> None:
    conjugation = sp.eye(5).row_join(sp.zeros(5)).col_join(sp.zeros(5).row_join(-sp.eye(5)))

    assert not is_complex_linear_real(conjugation)
    assert complex_from_real(conjugation) is None


def test_safe_real_commutant_basis_is_complex_linear() -> None:
    basis = safe_real_commutant_basis()

    assert len(basis) == 4
    assert all(is_complex_linear_real(matrix) for matrix in basis)


def test_safe_commutant_closure_examples_pass() -> None:
    assert safe_commutant_closure_examples_pass()


def test_matrix_unit_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError):
        matrix_unit(5, 0)
