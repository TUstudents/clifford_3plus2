from __future__ import annotations

from fractions import Fraction

from clifford_3plus2_d5.branching import (
    EXPECTED_HYPERCHARGES,
    EXPECTED_SECTOR_ORDER,
    branching_check_passed,
    branching_table,
    solve_hypercharge_formula,
    total_multiplicity,
)
from clifford_3plus2_d5.exterior import even_basis_3plus2, sector_multiplicities


def test_even_basis_has_spinor_dimension_16() -> None:
    assert len(even_basis_3plus2()) == 16
    assert sum(sector_multiplicities().values()) == 16


def test_hypercharge_formula_is_solved_exactly() -> None:
    formula = solve_hypercharge_formula()

    assert formula.a == Fraction(0)
    assert formula.b == Fraction(-1, 3)
    assert formula.c == Fraction(1, 2)
    assert formula.evaluate(0, 0) == Fraction(0)
    assert formula.evaluate(0, 2) == Fraction(1)
    assert formula.evaluate(1, 1) == Fraction(1, 6)


def test_branching_table_matches_expected_generation() -> None:
    table = branching_table()

    assert total_multiplicity() == 16
    assert tuple((sector.n3, sector.n2) for sector in table) == EXPECTED_SECTOR_ORDER
    assert {
        (sector.n3, sector.n2): sector.hypercharge for sector in table
    } == EXPECTED_HYPERCHARGES
    assert {(sector.n3, sector.n2): sector.label for sector in table} == {
        (0, 0): "nu^c",
        (0, 2): "e^c",
        (1, 1): "Q",
        (2, 0): "u^c",
        (2, 2): "d^c",
        (3, 1): "L",
    }
    assert branching_check_passed()
