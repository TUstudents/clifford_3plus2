from __future__ import annotations

from fractions import Fraction

from clifford_3plus2_d5.sm.hypercharge import (
    format_fraction,
    hypercharge,
    hypercharge_by_sector,
    hypercharge_check_passed,
    standard_hypercharge_formula,
)


def test_standard_hypercharge_formula_is_exact() -> None:
    formula = standard_hypercharge_formula()

    assert formula.constant == Fraction(0)
    assert formula.color_coefficient == Fraction(-1, 3)
    assert formula.weak_coefficient == Fraction(1, 2)


def test_hypercharge_values_match_one_generation_table() -> None:
    assert hypercharge(0, 0) == Fraction(0)
    assert hypercharge(0, 2) == Fraction(1)
    assert hypercharge(1, 1) == Fraction(1, 6)
    assert hypercharge(2, 0) == Fraction(-2, 3)
    assert hypercharge(2, 2) == Fraction(1, 3)
    assert hypercharge(3, 1) == Fraction(-1, 2)


def test_hypercharge_by_sector_is_stable_and_exact() -> None:
    assert hypercharge_by_sector() == {
        (0, 0): Fraction(0),
        (0, 2): Fraction(1),
        (1, 1): Fraction(1, 6),
        (2, 0): Fraction(-2, 3),
        (2, 2): Fraction(1, 3),
        (3, 1): Fraction(-1, 2),
    }
    assert hypercharge_check_passed()


def test_fraction_formatting_uses_exact_strings() -> None:
    assert format_fraction(Fraction(1, 6)) == "1/6"
    assert format_fraction(Fraction(-2, 3)) == "-2/3"
    assert format_fraction(Fraction(1)) == "1"
