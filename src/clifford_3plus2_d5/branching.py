"""Textbook Spin(10) branching arithmetic for Lambda^even(C^3 oplus C^2)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from clifford_3plus2_d5.exterior import sector_multiplicities


@dataclass(frozen=True)
class HyperchargeFormula:
    """Y = A + B N_3 + C N_2."""

    a: Fraction
    b: Fraction
    c: Fraction

    def evaluate(self, n3: int, n2: int) -> Fraction:
        return self.a + self.b * n3 + self.c * n2


@dataclass(frozen=True, order=True)
class BranchingSector:
    n3: int
    n2: int
    multiplicity: int
    hypercharge: Fraction
    label: str


EXPECTED_SECTOR_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (2, 2),
    (3, 1),
)

SECTOR_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "nu^c",
    (0, 2): "e^c",
    (1, 1): "Q",
    (2, 0): "u^c",
    (2, 2): "d^c",
    (3, 1): "L",
}

EXPECTED_HYPERCHARGES: dict[tuple[int, int], Fraction] = {
    (0, 0): Fraction(0),
    (0, 2): Fraction(1),
    (1, 1): Fraction(1, 6),
    (2, 0): Fraction(-2, 3),
    (2, 2): Fraction(1, 3),
    (3, 1): Fraction(-1, 2),
}


def solve_hypercharge_formula() -> HyperchargeFormula:
    """Solve Y(0,0)=0, Y(0,2)=1, and Y(1,1)=1/6 exactly."""

    a = Fraction(0)
    c = Fraction(1, 2)
    b = Fraction(1, 6) - a - c
    return HyperchargeFormula(a=a, b=b, c=c)


def branching_table() -> tuple[BranchingSector, ...]:
    """Return the textbook one-generation Spin(10) branching table."""

    formula = solve_hypercharge_formula()
    multiplicities = sector_multiplicities()
    return tuple(
        BranchingSector(
            n3=n3,
            n2=n2,
            multiplicity=multiplicities[(n3, n2)],
            hypercharge=formula.evaluate(n3, n2),
            label=SECTOR_LABELS[(n3, n2)],
        )
        for n3, n2 in EXPECTED_SECTOR_ORDER
    )


def total_multiplicity() -> int:
    return sum(sector.multiplicity for sector in branching_table())


def branching_check_passed() -> bool:
    table = branching_table()
    return (
        total_multiplicity() == 16
        and tuple((sector.n3, sector.n2) for sector in table) == EXPECTED_SECTOR_ORDER
        and all(
            sector.hypercharge == EXPECTED_HYPERCHARGES[(sector.n3, sector.n2)]
            for sector in table
        )
    )
