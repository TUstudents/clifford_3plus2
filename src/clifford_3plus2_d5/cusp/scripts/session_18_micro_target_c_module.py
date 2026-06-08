"""Run MicroCUSP Session G: Target-C module audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_target_c_module_audit


def main() -> None:
    """Print the microscopic Target-C module audit."""

    audit = microscopic_target_c_module_audit()
    print("primitive semigroup =", audit.primitive_return_semigroup)
    print("family basis =", audit.family_module_basis)
    print("module valuations heavy-to-light =", audit.module_valuations_heavy_to_light)
    print("left charges light-to-heavy =", audit.left_charges_light_to_heavy)
    print("conductor =", audit.conductor)
    print("down right charges =", audit.down_right_charges)
    print("up right charges =", audit.up_right_charges)
    print("down diagonal exponents =", audit.down_diagonal_exponents)
    print("up diagonal exponents =", audit.up_diagonal_exponents)
    print("up exponent matrix =", audit.up_exponent_matrix)
    print("down exponent matrix =", audit.down_exponent_matrix)
    print("wrong conductor controls =", audit.wrong_conductor_controls)
    print("lift controls =", audit.lift_controls)
    print("uses diagonal targets =", audit.uses_diagonal_targets)
    print("uses mass fits =", audit.uses_mass_fits)
    print("session G pass =", audit.session_g_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
