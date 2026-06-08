"""Run MicroCUSP global gate: SM quotient versus independent centers."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import sm_global_quotient_cusp_audit


def main() -> None:
    """Print the SM global quotient audit."""

    audit = sm_global_quotient_cusp_audit()
    print("independent tick vectors =", audit.independent_tick_vectors)
    print("quotient diagonal tick vector =", audit.quotient_diagonal_tick_vector)
    print("independent primitive lengths =", audit.independent_primitive_lengths)
    print("independent low valuations =", audit.independent_low_valuations)
    print("independent family basis =", audit.independent_family_basis)
    print("independent axes pass =", audit.independent_axes_pass)
    print("quotient primitive length =", audit.quotient_diagonal_primitive_length)
    print("quotient low valuations =", audit.quotient_diagonal_low_valuations)
    print("quotient family basis =", audit.quotient_diagonal_family_basis)
    print("U1-collapsed low valuations =", audit.u1_collapsed_low_valuations)
    print("quotient controls rejected =", audit.quotient_controls_rejected)
    print("session global quotient pass =", audit.session_global_quotient_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
