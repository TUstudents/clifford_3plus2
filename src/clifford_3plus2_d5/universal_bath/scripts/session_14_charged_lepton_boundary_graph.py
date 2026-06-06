"""Run the Session 14 charged-lepton minimal boundary graph certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.charged_lepton_boundary_graph import (
    charged_lepton_boundary_graph_payload,
)


def main() -> None:
    """Print Session 14 payload."""

    payload = charged_lepton_boundary_graph_payload()
    print("source dictionary pass =", payload.source_dictionary_pass)
    print("CMV head pass =", payload.cmv_head_pass)
    print("torsion gate pass =", payload.torsion_gate_pass)
    print("holonomy gate pass =", payload.holonomy_gate_pass)
    print("theta base =", payload.theta_base)
    print("torsion angle =", payload.torsion_angle)
    print("theta =", payload.theta)
    print("left coupling =")
    print(payload.left_coupling)
    print("right coupling =")
    print(payload.right_coupling)
    print("residue =")
    print(payload.residue)
    print("target residue =")
    print(payload.target_residue)
    print("residue matches target =", payload.residue_matches_target)
    print("selected port (u,a,b) =")
    print(payload.selected_port_uab)
    print("residue on selected port =")
    print(payload.residue_on_selected_port)
    print("normalized square-root vector (u,a,b) =")
    print(payload.normalized_square_root_vector_uab)
    print("trace weight =", payload.trace_weight)
    print("traceless weight =", payload.traceless_weight)
    print("trace/traceless equipartition =", payload.trace_traceless_equipartition)
    print("Koide parameter =", payload.koide_parameter)
    print("Koide parameter is 2/3 =", payload.koide_parameter_is_two_thirds)
    print("one trace path control rejected =", payload.one_trace_path_control_rejected)
    print("one-sided Hermitian control rejected =", payload.one_sided_hermitian_control_rejected)
    print("torsion angle not derived by graph =", payload.torsion_angle_not_derived_by_graph)
    print("remaining declared inputs =", payload.remaining_declared_inputs)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
