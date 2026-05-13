from __future__ import annotations

import argparse
from fractions import Fraction

from clifford_3plus2_d5.branching import (
    branching_check_passed,
    branching_table,
    solve_hypercharge_formula,
    total_multiplicity,
)


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check textbook Spin(10) branching arithmetic.")
    parser.add_argument("--check", action="store_true", help="Exit nonzero if the check fails.")
    args = parser.parse_args()

    formula = solve_hypercharge_formula()
    passed = branching_check_passed()

    print("This verifies standard Spin(10) branching only.")
    print("It does not prove the QCA supplies the split.")
    print(
        "hypercharge_formula: "
        f"Y = {_format_fraction(formula.a)} "
        f"+ ({_format_fraction(formula.b)}) N_3 "
        f"+ ({_format_fraction(formula.c)}) N_2"
    )
    print(f"total_multiplicity: {total_multiplicity()}")

    for sector in branching_table():
        print(
            "sector: "
            f"N_3={sector.n3}, "
            f"N_2={sector.n2}, "
            f"multiplicity={sector.multiplicity}, "
            f"Y={_format_fraction(sector.hypercharge)}, "
            f"label={sector.label}"
        )

    print(f"branching_check_passed: {str(passed).lower()}")
    print("load_bearing_qca_bridge: false")
    return 0 if passed or not args.check else 1


if __name__ == "__main__":
    raise SystemExit(main())
