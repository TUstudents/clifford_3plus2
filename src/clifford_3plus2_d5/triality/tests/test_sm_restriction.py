"""Tests for ``sm_restriction.py`` — the K1/K2 kill test."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.triality.reuse import physical_hypercharge_generator
from clifford_3plus2_d5.triality.sm_restriction import (
    g_sm_8_cartan_basis_coords,
    k1_failure_witnesses,
    k1_passes,
    kill_test_audit_payload,
    restricted_hypercharge_cartan_coords,
    restricted_hypercharge_generator,
    restricted_hypercharge_residual_norm_squared,
    su3_c_cartan_coords,
    su3_c_cartan_indices,
    y_prime_complex_spectrum,
    y_prime_observable,
    y_prime_real_spectrum,
)


def test_restricted_hypercharge_cartan_coords_are_expected() -> None:
    coords = restricted_hypercharge_cartan_coords()
    expected = sp.Matrix(
        4,
        1,
        [sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 3), sp.Rational(1, 2)],
    )
    assert (coords - expected).applyfunc(sp.simplify) == sp.zeros(4, 1)


def test_restricted_hypercharge_differs_from_physical() -> None:
    residual = restricted_hypercharge_residual_norm_squared()
    assert residual != 0


def test_restricted_hypercharge_generator_is_skew_symmetric() -> None:
    generator = restricted_hypercharge_generator()
    assert generator.shape == (32, 32)
    assert (generator + generator.T).applyfunc(sp.simplify) == sp.zeros(32)


def test_restricted_hypercharge_is_strict_subset_of_physical() -> None:
    full = physical_hypercharge_generator()
    restricted = restricted_hypercharge_generator()
    diff = (full - restricted).applyfunc(sp.simplify)
    assert diff != sp.zeros(32)


def test_su3_c_cartan_extraction_returns_two_generators() -> None:
    indices = su3_c_cartan_indices()
    coords = su3_c_cartan_coords()
    assert len(indices) == 2
    assert len(coords) == 2


def test_g_sm_8_cartan_basis_is_three_dimensional() -> None:
    basis = g_sm_8_cartan_basis_coords()
    assert len(basis) == 3
    span = sp.Matrix.hstack(*basis)
    assert span.rank() == 3


def test_k1_fails_under_pinned_embedding() -> None:
    # The expected kill-test outcome under the {0..7} embedding.
    # If this test ever starts passing K1, the embedding choice may have
    # changed or a deeper algebraic structure has been discovered.
    assert not k1_passes()


def test_k1_failure_witnesses_cover_every_sm_cartan_generator() -> None:
    # All three SM Cartan generators are mapped outside the SM Cartan span
    # by triality under the {0..7} embedding.
    witnesses = k1_failure_witnesses()
    assert len(witnesses) == 3


def test_y_prime_observable_is_real_symmetric() -> None:
    observable = y_prime_observable()
    assert observable.shape == (32, 32)
    assert (observable - observable.T).applyfunc(sp.simplify) == sp.zeros(32)


def test_y_prime_spectrum_has_total_complex_multiplicity_sixteen() -> None:
    spectrum = y_prime_complex_spectrum()
    assert sum(spectrum.values()) == 16


def test_y_prime_real_spectrum_multiplicities_are_even() -> None:
    spectrum = y_prime_real_spectrum()
    assert all(multiplicity % 2 == 0 for multiplicity in spectrum.values())


def test_kill_test_audit_payload_records_k1_failure() -> None:
    payload = kill_test_audit_payload()
    assert payload.k1_passes is False
    assert payload.k1_failure_witness_count == 3
    assert payload.g_sm_8_cartan_dimension == 3
    assert payload.su3_c_cartan_dimension == 2
    assert payload.y_prime_total_complex_multiplicity == 16
    assert "K1 FAIL" in payload.verdict
    assert "Program dies cleanly at K1" in payload.interpretation
