"""Tests for the R13 spectral-measure selection study."""

import sympy as sp

from clifford_3plus2_d5.radial_response.silver_transfer_inheritance import inherited_eta
from clifford_3plus2_d5.radial_response.spectral_measure_selection import (
    DiscreteSpectralMeasure,
    down_baseline_target_measure,
    down_candidate_target_measure,
    existing_r12_baths_match_target,
    jacobi_control_matches_target,
    jacobi_from_measure,
    jacobi_self_energy,
    measure_from_jacobi,
    measure_is_positive,
    measure_round_trips,
    p3_repair_jacobi,
    silver_tail_jacobi,
    spectral_measure_selection_payload,
    stieltjes_self_energy,
    up_target_measure,
    weights_are_normalized,
)


def test_stieltjes_self_energy_for_simple_measure() -> None:
    z = sp.Symbol("z")
    measure = DiscreteSpectralMeasure(
        "two_point",
        (sp.Integer(0), sp.Integer(2)),
        (sp.Rational(1, 4), sp.Rational(3, 4)),
    )
    assert weights_are_normalized(measure)
    assert sp.simplify(
        stieltjes_self_energy(measure, z) - (z - sp.Rational(1, 2)) / (z * (z - 2))
    ) == 0


def test_jacobi_round_trip_for_simple_measure() -> None:
    z = sp.Symbol("z")
    measure = DiscreteSpectralMeasure(
        "two_point",
        (sp.Integer(0), sp.Integer(2)),
        (sp.Rational(1, 4), sp.Rational(3, 4)),
    )
    jacobi = jacobi_from_measure(measure)
    assert jacobi == sp.Matrix(
        [
            [sp.Rational(3, 2), sp.sqrt(3) / 2],
            [sp.sqrt(3) / 2, sp.Rational(1, 2)],
        ]
    )
    assert sp.simplify(stieltjes_self_energy(measure, z) - jacobi_self_energy(jacobi, z)) == 0
    reconstructed = measure_from_jacobi(jacobi, z)
    assert reconstructed.poles == measure.poles
    assert reconstructed.weights == measure.weights


def test_target_measures_are_positive_and_use_inherited_eta() -> None:
    eta = inherited_eta()
    up = up_target_measure()
    down_baseline = down_baseline_target_measure()
    down_candidate = down_candidate_target_measure()
    expected_up_poles = (sp.Rational(1, 4) * eta**6, eta**3 / sp.sqrt(2), sp.Integer(1))
    assert all(
        sp.simplify(actual - expected) == 0
        for actual, expected in zip(up.poles, expected_up_poles, strict=True)
    )
    assert up.weights == (sp.Rational(1, 25), sp.Rational(8, 25), sp.Rational(16, 25))
    assert down_baseline.weights == (sp.Rational(1, 2), sp.Rational(1, 6), sp.Rational(1, 3))
    assert down_candidate.weights == (sp.Rational(6, 13), sp.Rational(2, 13), sp.Rational(5, 13))
    assert all(measure_is_positive(measure) for measure in (up, down_baseline, down_candidate))


def test_target_measures_jacobi_round_trip() -> None:
    for measure in (
        up_target_measure(),
        down_baseline_target_measure(),
        down_candidate_target_measure(),
    ):
        assert measure_round_trips(measure)
        jacobi = jacobi_from_measure(measure)
        assert jacobi == jacobi.T
        assert all(float(sp.N(jacobi[index, index + 1])) > 0 for index in range(jacobi.rows - 1))


def test_existing_forward_controls_do_not_select_targets() -> None:
    for measure in (
        up_target_measure(),
        down_baseline_target_measure(),
        down_candidate_target_measure(),
    ):
        assert not existing_r12_baths_match_target(measure)
        assert not jacobi_control_matches_target(p3_repair_jacobi(), measure)
        assert not jacobi_control_matches_target(silver_tail_jacobi(), measure)


def test_spectral_measure_selection_payload_reports_reconstruction_only() -> None:
    payload = spectral_measure_selection_payload()
    assert payload.final_verdict == "RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY"
    assert payload.positive_measures
    assert payload.jacobi_round_trips
    assert not payload.r12_baths_match_targets
    assert not payload.p3_repair_matches_targets
    assert not payload.silver_tail_matches_targets
    assert not payload.minimal_unitary_matches_up_at_z2
    assert not payload.simple_forward_bath_selected
