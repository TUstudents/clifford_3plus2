"""Tests for ``color_center_z3.py`` — Phase D-2."""

from __future__ import annotations

from clifford_3plus2_d5.topology.color_center_z3 import (
    PER_GENERATION_FIELD_CONTENT,
    color_z3_decomposition_payload,
    multiplicities_are_symmetric,
    per_generation_total_fermion_count,
    three_generations_required_multiplicity,
    z3_decomposition_per_generation,
)


def test_per_generation_total_is_sixteen() -> None:
    assert per_generation_total_fermion_count() == 16


def test_field_content_has_six_species() -> None:
    assert len(PER_GENERATION_FIELD_CONTENT) == 6


def test_z3_decomposition_is_eight_four_four() -> None:
    trivial, omega, omega2 = z3_decomposition_per_generation()
    assert (trivial, omega, omega2) == (8, 4, 4)


def test_decomposition_sums_to_sixteen() -> None:
    trivial, omega, omega2 = z3_decomposition_per_generation()
    assert trivial + omega + omega2 == 16


def test_multiplicities_are_not_symmetric() -> None:
    assert not multiplicities_are_symmetric()


def test_symmetric_multiplicity_would_be_16_over_3() -> None:
    assert three_generations_required_multiplicity() == 16 / 3


def test_audit_payload_confirms_kill() -> None:
    payload = color_z3_decomposition_payload()
    assert payload.chiral16_total_count == 16
    assert payload.trivial_multiplicity == 8
    assert payload.omega_multiplicity == 4
    assert payload.omega2_multiplicity == 4
    assert not payload.is_symmetric
    assert "COLOR Z_3 KILL" in payload.verdict
    assert "8 + 4 + 4" in payload.verdict
    assert "ASYMMETRIC" in payload.interpretation
