"""Tests for killing literal nilpotent Taylor as the Yukawa matrix."""

import sympy as sp

from clifford_3plus2_d5.radial_response.literal_nilpotent_yukawa import (
    left_rotation_metric_is_diagonal,
    literal_nilpotent_yukawa_payload,
    nilpotent_exponential_matrix,
    normalized_singular_value_ratios_at_unit_x,
    singular_values_at_unit_x,
)


def test_literal_unit_shift_exponential_has_order_one_singular_values() -> None:
    matrix = nilpotent_exponential_matrix()
    assert matrix == sp.Matrix(
        [
            [1, 1, sp.Rational(1, 2)],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )
    assert singular_values_at_unit_x() == (sp.Integer(2), sp.Integer(1), sp.Rational(1, 2))
    assert normalized_singular_value_ratios_at_unit_x() == (
        sp.Integer(1),
        sp.Rational(1, 2),
        sp.Rational(1, 4),
    )


def test_literal_unit_shift_exponential_has_left_rotation() -> None:
    assert not left_rotation_metric_is_diagonal()


def test_literal_nilpotent_yukawa_payload_kills_matrix_interpretation() -> None:
    payload = literal_nilpotent_yukawa_payload()
    assert payload.final_verdict == "LITERAL_NILPOTENT_YUKAWA_KILL"
    assert not payload.left_rotation_metric_diagonal
    assert not payload.literal_matrix_can_be_yukawa
