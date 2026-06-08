"""Run MicroCUSP Session H: Target-D topology audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_target_d_topology_audit


def main() -> None:
    """Print the microscopic Target-D topology audit."""

    audit = microscopic_target_d_topology_audit()
    print("family flag edges =", audit.family_flag_edges)
    print("center labels =", audit.center_labels)
    print("up center powers =", audit.up_center_powers)
    print("down center powers =", audit.down_center_powers)
    print("up amplitudes =", audit.up_amplitudes)
    print("down amplitudes =", audit.down_amplitudes)
    print("CP invariant =", audit.cp_invariant)
    print("real control invariant =", audit.real_control_invariant)
    print("one-sector controls =", audit.one_sector_control_invariants)
    print("rephased invariant =", audit.rephased_cp_invariant)
    print("topology controls rejected =", audit.topology_controls_rejected)
    print("session H pass =", audit.session_h_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
