"""Run the Target-C right-charge origin audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_right_charge_origin_audit


def main() -> None:
    """Print the CUSP Target-C right-charge origin payload."""

    audit = cusp_right_charge_origin_audit()
    print("left charges =", audit.left_charges)
    print("cusp conductor =", audit.conductor)
    print("weak lift factor =", audit.weak_lift_factor)
    print("color lift control factor =", audit.color_lift_control_factor)
    print("down origin rule =", audit.down_origin_rule)
    print("up origin rule =", audit.up_origin_rule)
    print("down right charges =", audit.down_right_charges)
    print("up right charges =", audit.up_right_charges)
    print("down diagonal exponents =", audit.down_diagonal_exponents)
    print("up diagonal exponents =", audit.up_diagonal_exponents)
    print("trivial lift control =", audit.trivial_lift_control)
    print("color lift control =", audit.color_lift_control)
    print("all right charges nonnegative =", audit.all_right_charges_nonnegative)
    print("uses mass fits =", audit.uses_mass_fits)
    print("right-charge origin pass =", audit.right_charge_origin_pass)


if __name__ == "__main__":
    main()
