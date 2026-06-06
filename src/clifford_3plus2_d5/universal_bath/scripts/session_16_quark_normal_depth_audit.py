"""Run the Session 16 quark normal-depth placement audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_normal_depth_audit import (
    quark_normal_depth_audit_payload,
)


def main() -> None:
    """Print Session 16 payload."""

    payload = quark_normal_depth_audit_payload()
    print("quark source assembly pass =", payload.quark_source_assembly_pass)
    print("nilpotent flag pass =", payload.nilpotent_flag_pass)
    print("microscopic locality pass =", payload.microscopic_locality_pass)
    print("port height filtration =", payload.port_height_filtration)
    print(
        "port height filtration microscopically derived =",
        payload.port_height_filtration_microscopically_derived,
    )
    print(
        "one-tick geometry microscopically derived =",
        payload.one_tick_geometry_microscopically_derived,
    )
    print("repair flag =")
    print(payload.repair_flag)
    print("depth operator in port basis =")
    print(payload.depth_operator_port_basis)
    print("hand-written diagonal control =")
    print(payload.hand_written_diagonal_control)
    print("depth operator not diagonal =", payload.depth_operator_not_diagonal_in_port_basis)
    print("defect mode depths =", payload.defect_mode_depths)
    print("doubled port heights =", payload.doubled_port_heights)
    print("port heights are not depth spectrum =", payload.port_heights_are_not_depth_spectrum)
    print("up dictionary normal depth =", payload.up_dictionary_normal_depth)
    print("down dictionary normal depth =", payload.down_dictionary_normal_depth)
    print("dictionary depths still unfrozen =", payload.dictionary_depths_still_unfrozen)
    print("graph depths do not freeze source depths =", payload.graph_depths_do_not_freeze_source_depths)
    print("normal depth premise still open =", payload.normal_depth_premise_still_open)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
