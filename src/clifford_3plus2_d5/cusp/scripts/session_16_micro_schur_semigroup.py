"""Run MicroCUSP Session E: Schur-return semigroup audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_schur_semigroup_audit


def main() -> None:
    """Print the microscopic Schur-semigroup audit."""

    audit = microscopic_schur_semigroup_audit()
    print("return moments through six =", audit.return_moments_through_six)
    print("M1 zero =", audit.m1_zero)
    print("M2 nonzero =", audit.m2_nonzero)
    print("M3 nonzero =", audit.m3_nonzero)
    print("primitive return semigroup =", audit.primitive_return_semigroup)
    print("recirculation algebra =", audit.recirculation_algebra)
    print("maximal ideal generators =", audit.maximal_ideal_generators)
    print("family module basis =", audit.family_module_basis)
    print("low valuations =", audit.low_valuations)
    print("controls =", audit.controls)
    print("controls rejected =", audit.controls_rejected)
    print("session E pass =", audit.session_e_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
