"""Tests for ``bcc_centrosymmetry.py``."""

from __future__ import annotations

from clifford_3plus2_d5.strongcp.bcc_centrosymmetry import (
    bb_walk_is_centrosymmetric_per_direction,
    bcc_centrosymmetry_payload,
    bcc_lattice_site_set_is_centrosymmetric,
    bloch_symbol_is_centrosymmetric,
)


def test_lattice_sites_centrosymmetric() -> None:
    assert bcc_lattice_site_set_is_centrosymmetric()


def test_walk_hops_centrosymmetric() -> None:
    assert bb_walk_is_centrosymmetric_per_direction()


def test_bloch_symbol_centrosymmetric() -> None:
    assert bloch_symbol_is_centrosymmetric()


def test_centrosymmetry_payload_all_consistent() -> None:
    payload = bcc_centrosymmetry_payload()
    assert payload.lattice_sites_centrosymmetric
    assert payload.walk_hops_centrosymmetric
    assert payload.bloch_symbol_centrosymmetric
    assert payload.all_centrosymmetric
    assert "BCC CENTROSYMMETRIC" in payload.verdict


def test_centrosymmetry_interpretation_mentions_selection_rule() -> None:
    payload = bcc_centrosymmetry_payload()
    text = payload.interpretation.lower()
    assert "g-irrep" in text or "u-irrep" in text
    assert "centrosymm" in text
