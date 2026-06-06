"""Tests for the two-channel scalar repair isometry gate."""

import sympy as sp

from clifford_3plus2_d5.radial_response.two_channel_isometry import (
    asymmetric_two_channel_amplitudes,
    equal_successor_amplitudes,
    leakage_repair_amplitude,
    total_probability,
    two_channel_isometry_payload,
    two_channel_repair_amplitude,
)


def test_two_equal_no_leakage_channels_force_one_over_sqrt_two() -> None:
    amplitudes = equal_successor_amplitudes(2)
    assert amplitudes == (1 / sp.sqrt(2), 1 / sp.sqrt(2))
    assert two_channel_repair_amplitude() == 1 / sp.sqrt(2)
    assert total_probability(amplitudes) == 1


def test_channel_count_controls_do_not_derive_two_channel_amplitude() -> None:
    assert equal_successor_amplitudes(1) == (sp.Integer(1),)
    assert equal_successor_amplitudes(3) == (
        1 / sp.sqrt(3),
        1 / sp.sqrt(3),
        1 / sp.sqrt(3),
    )


def test_leakage_and_asymmetry_leave_amplitude_unforced() -> None:
    ell = sp.Symbol("ell", nonnegative=True)
    weight = sp.Symbol("w", positive=True)
    assert leakage_repair_amplitude(ell) == sp.sqrt((1 - ell) / 2)
    assert asymmetric_two_channel_amplitudes(weight) != (1 / sp.sqrt(2), 1 / sp.sqrt(2))


def test_two_channel_payload_passes() -> None:
    payload = two_channel_isometry_payload()
    assert payload.final_verdict == "TWO_CHANNEL_REPAIR_ISOMETRY_PASS"
    assert payload.two_channel_amplitudes == (1 / sp.sqrt(2), 1 / sp.sqrt(2))
    assert payload.asymmetric_control_is_free
