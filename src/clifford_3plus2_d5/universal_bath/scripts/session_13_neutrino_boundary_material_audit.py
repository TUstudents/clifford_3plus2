"""Run the Session 13 boundary-material origin audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_boundary_material_audit import (
    boundary_material_audit_payload,
)


def main() -> None:
    """Print Session 13 payload."""

    payload = boundary_material_audit_payload()
    print("local mismatch coordinate unique =", payload.local_mismatch_coordinate_unique)
    print("positive penalty if constraint admitted =", payload.positive_penalty_if_constraint_admitted)
    print("bare blocks have no gap parameter =", payload.bare_blocks_have_no_gap_parameter)
    print("retarded completion uses same emission norm =", payload.retarded_completion_uses_same_emission_norm)
    print("recurrent completion uses same emission norm =", payload.recurrent_completion_uses_same_emission_norm)
    print("retarded and recurrent visible powers differ =", payload.retarded_and_recurrent_visible_powers_differ)
    print("recurrent two-step return nonzero =", payload.recurrent_two_step_return_nonzero)
    print("outgoing asymptotics forced by bare blocks =", payload.outgoing_asymptotics_forced_by_bare_blocks)
    print("positive locking forced by bare blocks =", payload.positive_locking_forced_by_bare_blocks)
    print("Session 12 conditional graph passes =", payload.session_12_conditional_graph_passes)
    print("next required object =", payload.next_required_object)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
