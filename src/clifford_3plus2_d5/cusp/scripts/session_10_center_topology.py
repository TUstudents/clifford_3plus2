"""Run the Target-D finite center-topology audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_center_topology_audit


def main() -> None:
    """Print the CUSP Target-D center-topology payload."""

    audit = cusp_center_topology_audit()
    print("up center powers =", audit.up_center_powers)
    print("down center powers =", audit.down_center_powers)
    print("up rule =", audit.up_rule.name)
    print("down rule =", audit.down_rule.name)
    print("up controls =", audit.up_control_powers)
    print("down controls =", audit.down_control_powers)
    print("up controls rejected =", audit.up_controls_rejected)
    print(
        "down controls rejected except conjugate =",
        audit.down_controls_rejected_except_conjugate,
    )
    print("down direct row/col zero =", audit.down_direct_row_col_zero)
    print("down unit pairing normalized =", audit.down_unit_pairing_normalized)
    print("finite topology selection pass =", audit.finite_topology_selection_pass)
    print("boundary microphysics selected =", audit.boundary_microphysics_selected)


if __name__ == "__main__":
    main()
