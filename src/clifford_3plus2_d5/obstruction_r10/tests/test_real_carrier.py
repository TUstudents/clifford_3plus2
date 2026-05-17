from __future__ import annotations

import sympy as sp
import pytest

from clifford_3plus2_d5.algebra.matrices import (
    commutator,
    epsilon,
    identity,
    is_zero_matrix,
    skew_generator,
)
from clifford_3plus2_d5.algebra.real_carrier import (
    carrier_certificate,
    phase_1_check_passed,
    split_projectors_3_2,
    standard_basis,
    standard_real_carrier,
)


def test_epsilon_is_exact_quarter_turn() -> None:
    eps = epsilon()

    assert eps == sp.Matrix([[0, -1], [1, 0]])
    assert eps * eps == -identity(2)


def test_standard_basis_order_is_fixed() -> None:
    assert standard_basis() == (
        "x_1",
        "x_2",
        "x_3",
        "x_4",
        "x_5",
        "y_1",
        "y_2",
        "y_3",
        "y_4",
        "y_5",
    )


def test_real_carrier_dimension_and_metric() -> None:
    carrier = standard_real_carrier()

    assert carrier.dimension == 10
    assert carrier.mode_dimension == 5
    assert carrier.metric.shape == (10, 10)
    assert carrier.metric == identity(10)
    assert carrier.metric == carrier.metric.T
    assert all(value.is_real for value in carrier.metric)


def test_complex_structure_identities_are_exact() -> None:
    carrier = standard_real_carrier()
    j = carrier.complex_structure

    assert j.shape == (10, 10)
    assert j * j == -identity(10)
    assert j.T * j == identity(10)
    assert j.det() == 1


def test_split_projector_identities_are_exact() -> None:
    p3, p2 = split_projectors_3_2()
    j = standard_real_carrier().complex_structure

    assert p3 + p2 == identity(10)
    assert p3 * p3 == p3
    assert p2 * p2 == p2
    assert p3 * p2 == sp.zeros(10)
    assert p3.rank() == 6
    assert p2.rank() == 4
    assert is_zero_matrix(commutator(j, p3))
    assert is_zero_matrix(commutator(j, p2))


def test_primitive_skew_generators_are_exact() -> None:
    generator = skew_generator(0, 7)

    assert generator.shape == (10, 10)
    assert generator.T + generator == sp.zeros(10)
    assert set(generator) == {-1, 0, 1}


@pytest.mark.parametrize(
    ("row", "column"),
    [(0, 0), (-1, 2), (0, 10)],
)
def test_invalid_skew_generators_raise(row: int, column: int) -> None:
    with pytest.raises(ValueError):
        skew_generator(row, column)


def test_phase_1_certificate_is_non_load_bearing() -> None:
    certificate = carrier_certificate()

    assert phase_1_check_passed()
    assert certificate["phase_1_real_carrier_check_passed"] is True
    assert certificate["qca_forces_j"] is False
    assert certificate["load_bearing_qca_bridge"] is False
