"""Run the Session 17 active hidden color-return microcanonical audit."""

from __future__ import annotations

from clifford_3plus2_d5.universal_bath.quark_active_color_microcanonical import (
    quark_active_color_microcanonical_payload,
)


def main() -> None:
    """Print Session 17 payload."""

    payload = quark_active_color_microcanonical_payload()
    print("quark source assembly pass =", payload.quark_source_assembly_pass)
    print("color lift pass =", payload.color_lift_pass)
    print("primitive shell pass =", payload.primitive_shell_pass)
    print("microcanonical reduction pass =", payload.microcanonical_reduction_pass)
    print("primitive channel names =", payload.primitive_channel_names)
    print("primitive shell breakdown =", payload.primitive_shell_breakdown)
    print("active shell breakdown =", payload.active_shell_breakdown)
    print("active shell matches primitive shell =", payload.active_shell_matches_primitive_shell)
    print("spectator shell dimension =", payload.spectator_shell_dimension)
    print("spectator is compressed control =", payload.spectator_is_compressed_control)
    print("equal degeneracy density uniform =", payload.equal_degeneracy_density_uniform)
    print("equal degeneracy weights =", payload.equal_degeneracy_weights)
    print("equal weights cover all primitive labels =", payload.equal_weights_cover_all_primitive_labels)
    print("compressed macro ratio =", payload.compressed_macro_ratio)
    print("compressed macro phase =", payload.compressed_macro_phase)
    print("compressed macro control rejected =", payload.compressed_macro_control_rejected)
    print(
        "active return selected inside microcanonical shell =",
        payload.active_return_selected_inside_microcanonical_shell,
    )
    print("gauge alone selected active =", payload.gauge_alone_selected_active)
    print("remaining declared inputs =", payload.remaining_declared_inputs_after_reduction)
    print(
        "active color blocker reduced to microcanonical prior =",
        payload.active_color_blocker_reduced_to_microcanonical_prior,
    )
    print("source freeze ready =", payload.source_freeze_ready)
    print("verdict =", payload.final_verdict)
    print(payload.interpretation)


if __name__ == "__main__":
    main()
