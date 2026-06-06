"""Run the Session 22 quark current-parity selector certificate."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_current_parity_selector import (
    quark_current_parity_selector_payload,
)


def main() -> None:
    """Print Session 22 payload."""

    payload = quark_current_parity_selector_payload()
    print("selected S2 residual action (u,a,b) =", payload.selected_s2_residual_action_uab)
    print("parity by line =", payload.parity_by_line)
    print("even projector is u+a =", payload.even_projector_is_u_plus_a)
    print("odd projector is b =", payload.odd_projector_is_b)
    print("active plane pass =", payload.active_plane_pass)
    print("current line standard =", payload.current_line_standard)
    print("current line residual (u,a,b) =", payload.current_line_residual_uab)
    print("current line is b =", payload.current_line_is_b)
    print("current line is selected pair current =", payload.current_line_is_selected_pair_current)
    print("current line is odd =", payload.current_line_is_odd)
    print(
        "selected scalar has no current component =",
        payload.selected_scalar_has_no_current_component,
    )
    print("active plane alone insufficient =", payload.active_plane_alone_insufficient)
    print("u rejected as scalar/even =", payload.u_rejected_as_scalar_even)
    print("a rejected as radial/even =", payload.a_rejected_as_radial_even)
    print("current parity selects b =", payload.current_parity_selects_b)
    print("Session 21 source premise reduced =", payload.session21_source_premise_reduced)
    print("remaining physical premise =", payload.remaining_physical_premise)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
