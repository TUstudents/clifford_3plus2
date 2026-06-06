"""Run the Session 11 selected active-plane incidence certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.neutrino_active_plane import (
    active_plane_incidence_payload,
)


def main() -> None:
    """Print Session 11 payload."""

    payload = active_plane_incidence_payload()
    print("selected port =")
    print(payload.selected_port)
    print("selected port components (a,u,b) =", payload.selected_port_components_aub)
    print("detraced radial line =")
    print(payload.detraced_radial_line)
    print("opposite edge line =")
    print(payload.opposite_edge_line)
    print("active projector =")
    print(payload.active_projector)
    print("radial projector =")
    print(payload.radial_projector)
    print("active projector rank =", payload.active_projector_rank)
    print("radial projector rank =", payload.radial_projector_rank)
    print("detraced line matches a =", payload.detraced_line_matches_a)
    print("opposite edge line matches b =", payload.opposite_edge_line_matches_b)
    print("active projector matches Session 10 =", payload.active_projector_matches_session_10)
    print("active + radial = I =", payload.active_radial_resolution_identity)
    print("active projector selected-S2 invariant =", payload.active_projector_selected_s2_invariant)
    print("active projector not full-S3 invariant =", payload.active_projector_not_full_s3_invariant)
    print("selected-S2 symmetry alone not sufficient =", payload.selected_s2_symmetry_alone_not_sufficient)
    print("raw selected-port line control rejected =", payload.raw_selected_port_line_control_rejected)
    print("Session 10 family moment gate passes =", payload.session_10_family_moment_gate_passes)
    print("remaining declared inputs =", payload.remaining_declared_inputs)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
