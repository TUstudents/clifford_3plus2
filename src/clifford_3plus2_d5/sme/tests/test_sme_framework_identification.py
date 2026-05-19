"""Tests for ``sme_framework_identification.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.sme.sme_framework_identification import (
    dim5_sme_sector_label,
    framework_identification_payload,
    h1_cp_odd_fraction,
    h1_is_chirality_preserving,
    h1_is_hermitian,
    h1_is_momentum_bilinear,
    h1_lives_entirely_in_cp_odd_t2g,
    h1_symmetry_class,
    h1_t2g_fraction,
)


def test_h1_is_hermitian() -> None:
    assert h1_is_hermitian()


def test_h1_is_chirality_preserving() -> None:
    assert h1_is_chirality_preserving()


def test_h1_is_momentum_bilinear() -> None:
    assert h1_is_momentum_bilinear()


def test_h1_cp_odd_fraction_is_one() -> None:
    assert sp.simplify(h1_cp_odd_fraction() - 1) == 0


def test_h1_t2g_fraction_is_one() -> None:
    assert sp.simplify(h1_t2g_fraction() - 1) == 0


def test_h1_lives_entirely_in_cp_odd_t2g() -> None:
    assert h1_lives_entirely_in_cp_odd_t2g()


def test_dim5_sme_sector_label_mentions_non_minimal() -> None:
    label = dim5_sme_sector_label()
    assert "dim-5" in label
    assert "non-minimal" in label
    assert "fermion" in label
    assert "CP-odd" in label


def test_symmetry_class_has_all_fields_consistent() -> None:
    sym = h1_symmetry_class()
    assert sym.hermitian
    assert sym.chirality_preserving
    assert sym.momentum_bilinear
    assert sym.lives_entirely_in_cp_odd_t2g
    assert sp.simplify(sym.cp_odd_fraction - 1) == 0
    assert sp.simplify(sym.t2g_fraction - 1) == 0


def test_framework_identification_payload_consistent() -> None:
    payload = framework_identification_payload()
    assert payload.all_classes_consistent
    assert "FRAMEWORK IDENTIFIED" in payload.verdict
    assert "dim-5" in payload.sme_sector_label
