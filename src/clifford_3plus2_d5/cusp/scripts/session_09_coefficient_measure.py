"""Run the Target-D cusp-module coefficient measure audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_coefficient_measure_audit


def main() -> None:
    """Print the CUSP Target-D coefficient measure payload."""

    audit = cusp_coefficient_measure_audit()
    print("amplitude rule =", audit.amplitude_rule)
    print("up exponent matrix =", audit.up_exponent_matrix)
    print("down exponent matrix =", audit.down_exponent_matrix)
    print("up amplitudes =", audit.up_amplitudes)
    print("down amplitudes =", audit.down_amplitudes)
    print("all amplitudes positive =", audit.all_amplitudes_positive)
    print("uses fitted amplitudes =", audit.uses_fitted_amplitudes)
    print("CP invariant =", audit.cp_invariant)
    print("CP invariant numeric =", audit.cp_invariant.evalf(15))
    print("real-control invariant =", audit.real_control_invariant)
    print("rephased invariant =", audit.rephased_cp_invariant)
    print(
        "holonomy topology selected by boundary =",
        audit.holonomy_topology_selected_by_boundary,
    )
    print("coefficient measure pass =", audit.coefficient_measure_pass)


if __name__ == "__main__":
    main()
