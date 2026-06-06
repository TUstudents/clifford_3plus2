"""Run the Session 19 charged-lepton trace/torsion origin audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.charged_lepton_trace_torsion_origin import (
    charged_lepton_trace_torsion_origin_payload,
)


def main() -> None:
    """Print Session 19 payload."""

    payload = charged_lepton_trace_torsion_origin_payload()
    print("boundary graph pass =", payload.boundary_graph_pass)
    print("torsion gate pass =", payload.torsion_gate_pass)
    print("holonomy gate pass =", payload.holonomy_gate_pass)
    print("source occupation weights =", payload.source_occupation_weights)
    print("required trace path count =", payload.required_trace_path_count)
    print("supplied trace path count =", payload.supplied_trace_path_count)
    print(
        "supplied trace paths match required count =",
        payload.supplied_trace_paths_match_required_count,
    )
    print("trace weight formula =", payload.trace_weight_formula)
    print("traceless weight formula =", payload.traceless_weight_formula)
    print("two trace paths force equipartition =", payload.two_trace_paths_force_equipartition)
    print("one trace path control rejected =", payload.one_trace_path_control_rejected)
    print("three trace path control rejected =", payload.three_trace_path_control_rejected)
    print(
        "trace path origin derived from BCC/Higgs =",
        payload.trace_path_origin_derived_from_bcc_higgs,
    )
    print("torsion occupation weight =", payload.torsion_occupation_weight)
    print("torsion used as rotation angle =", payload.torsion_used_as_rotation_angle)
    print("torsion weight inserted into theta =", payload.torsion_weight_inserted_into_theta)
    print("occupation-to-angle map derived =", payload.occupation_to_angle_map_derived)
    print("CMV phase word pass =", payload.cmv_phase_word_pass)
    print("CMV phase word is not torsion angle =", payload.cmv_phase_word_is_not_torsion_angle)
    print("minimal graph accepts supplied theta =", payload.minimal_graph_accepts_supplied_theta)
    print("Koide still exact with supplied theta =", payload.koide_still_exact_with_supplied_theta)
    print("remaining declared inputs =", payload.remaining_declared_inputs)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
