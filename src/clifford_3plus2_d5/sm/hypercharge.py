"""Exact hypercharge bookkeeping for Lambda^even(C^3 plus C^2)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from clifford_3plus2_d5.branching import EXPECTED_HYPERCHARGES


@dataclass(frozen=True)
class HyperchargeFormula:
    constant: Fraction = Fraction(0)
    color_coefficient: Fraction = Fraction(-1, 3)
    weak_coefficient: Fraction = Fraction(1, 2)

    def evaluate(self, n3: int, n2: int) -> Fraction:
        return self.constant + self.color_coefficient * n3 + self.weak_coefficient * n2


def standard_hypercharge_formula() -> HyperchargeFormula:
    return HyperchargeFormula()


def hypercharge(n3: int, n2: int) -> Fraction:
    return standard_hypercharge_formula().evaluate(n3, n2)


def hypercharge_by_sector() -> dict[tuple[int, int], Fraction]:
    return dict(EXPECTED_HYPERCHARGES)


def hypercharge_check_passed() -> bool:
    return all(hypercharge(n3, n2) == value for (n3, n2), value in EXPECTED_HYPERCHARGES.items())


def format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
