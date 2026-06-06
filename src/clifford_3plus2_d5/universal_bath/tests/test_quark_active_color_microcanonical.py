"""Tests for Session 17 active hidden color-return microcanonical audit."""

import sympy as sp

from clifford_3plus2_d5.universal_bath.quark_active_color_microcanonical import (
    MICROCANONICAL_ACTIVE_RETURN_PREMISE,
    active_shell_matches_primitive_shell,
    equal_degeneracy_label_weights,
    equal_weights_cover_all_primitive_labels,
    primitive_channel_names,
    quark_active_color_microcanonical_payload,
    spectator_is_compressed_control,
)


def test_equal_degeneracy_weights_cover_all_six_primitive_labels() -> None:
    assert len(primitive_channel_names()) == 6
    assert equal_degeneracy_label_weights() == (
        sp.Rational(1, 6),
        sp.Rational(1, 6),
        sp.Rational(1, 6),
        sp.Rational(1, 6),
        sp.Rational(1, 6),
        sp.Rational(1, 6),
    )
    assert equal_weights_cover_all_primitive_labels()


def test_active_shell_matches_primitive_shell_and_spectator_is_compressed() -> None:
    assert active_shell_matches_primitive_shell()
    assert spectator_is_compressed_control()


def test_quark_active_color_microcanonical_payload_reports_conditional_pass() -> None:
    payload = quark_active_color_microcanonical_payload()

    assert payload.final_verdict == "QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS"
    assert payload.quark_source_assembly_pass
    assert payload.color_lift_pass
    assert payload.primitive_shell_pass
    assert payload.microcanonical_reduction_pass
    assert payload.primitive_shell_breakdown == {
        "even_direct": 1,
        "bcc_odd": 2,
        "color_odd": 3,
        "odd_total": 5,
        "total": 6,
    }
    assert payload.active_shell_breakdown == payload.primitive_shell_breakdown
    assert payload.spectator_shell_dimension == 3
    assert payload.spectator_is_compressed_control
    assert payload.equal_degeneracy_density_uniform
    assert payload.equal_weights_cover_all_primitive_labels
    assert sp.simplify(payload.compressed_macro_ratio - 1 / sp.sqrt(5)) == 0
    assert payload.compressed_macro_control_rejected
    assert payload.active_return_selected_inside_microcanonical_shell
    assert not payload.gauge_alone_selected_active
    assert MICROCANONICAL_ACTIVE_RETURN_PREMISE in payload.remaining_declared_inputs_after_reduction
    assert payload.active_color_blocker_reduced_to_microcanonical_prior
    assert not payload.source_freeze_ready
