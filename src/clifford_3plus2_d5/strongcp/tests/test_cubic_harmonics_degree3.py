"""Tests for ``cubic_harmonics_degree3.py``."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.strongcp.cubic_harmonics_degree3 import (
    DEGREE3_IRREP_NAMES,
    coefficient_vector_to_polynomial_deg3,
    decompose_degree3_polynomial,
    deg3_projectors_satisfy_idempotent_orthogonal_complete,
    degree3_monomial_basis,
    polynomial_to_coefficient_vector_deg3,
    projector_A1g_deg3,
    projector_A1u_deg3,
    projector_A2g_deg3,
    projector_A2u_deg3,
    projector_Eg_deg3,
    projector_Eu_deg3,
    projector_T1g_deg3,
    projector_T1u_deg3,
    projector_T2g_deg3,
    projector_T2u_deg3,
)


def _ks() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    return sp.symbols("kx ky kz", real=True)


def test_monomial_basis_is_ten_terms_in_fixed_order() -> None:
    ks = _ks()
    basis = degree3_monomial_basis(ks)
    assert len(basis) == 10
    kx, ky, kz = ks
    assert basis[0] == kx**3
    assert basis[9] == kx * ky * kz


def test_projector_A2u_is_rank_one() -> None:
    p = projector_A2u_deg3()
    assert p.shape == (10, 10)
    assert sp.trace(p) == 1


def test_projector_T2u_is_rank_three() -> None:
    p = projector_T2u_deg3()
    assert p.shape == (10, 10)
    assert sp.trace(p) == 3


def test_projector_T1u_is_rank_six() -> None:
    p = projector_T1u_deg3()
    assert p.shape == (10, 10)
    assert sp.trace(p) == 6


def test_empty_irreps_at_degree3_are_zero() -> None:
    for proj_fn in (
        projector_A1g_deg3,
        projector_A2g_deg3,
        projector_Eg_deg3,
        projector_T1g_deg3,
        projector_T2g_deg3,
        projector_A1u_deg3,
        projector_Eu_deg3,
    ):
        assert proj_fn() == sp.zeros(10, 10)


def test_projectors_idempotent_orthogonal_complete() -> None:
    assert deg3_projectors_satisfy_idempotent_orthogonal_complete()


def test_kx_ky_kz_projects_entirely_into_A2u() -> None:
    ks = _ks()
    kx, ky, kz = ks
    decomp = decompose_degree3_polynomial(kx * ky * kz, ks)
    assert decomp["A2u"] == kx * ky * kz
    assert decomp["T2u"] == 0
    assert decomp["T1u"] == 0


def test_kx_y2_minus_z2_projects_entirely_into_T2u() -> None:
    ks = _ks()
    kx, ky, kz = ks
    poly = sp.expand(kx * (ky**2 - kz**2))
    decomp = decompose_degree3_polynomial(poly, ks)
    assert decomp["A2u"] == 0
    assert sp.expand(decomp["T2u"] - poly) == 0
    assert decomp["T1u"] == 0


def test_kx_r2_projects_entirely_into_T1u() -> None:
    ks = _ks()
    kx, ky, kz = ks
    poly = sp.expand(kx * (kx**2 + ky**2 + kz**2))
    decomp = decompose_degree3_polynomial(poly, ks)
    assert decomp["A2u"] == 0
    assert decomp["T2u"] == 0
    assert sp.expand(decomp["T1u"] - poly) == 0


def test_decomposition_sum_equals_original() -> None:
    ks = _ks()
    kx, ky, kz = ks
    poly = sp.expand(
        2 * kx**3 + 3 * ky**2 * kz - kx * ky * kz + 5 * kz**2 * ky
    )
    decomp = decompose_degree3_polynomial(poly, ks)
    total = sum(decomp.values(), start=sp.Integer(0))
    assert sp.expand(total - poly) == 0


def test_polynomial_coefficient_round_trip() -> None:
    ks = _ks()
    kx, ky, kz = ks
    original = sp.expand(
        kx**3 + 2 * ky**3 + 3 * kz**3
        + 4 * kx**2 * ky - 5 * kx**2 * kz
        + 6 * ky**2 * kx + 7 * ky**2 * kz
        - 8 * kz**2 * kx + 9 * kz**2 * ky
        + 11 * kx * ky * kz
    )
    vec = polynomial_to_coefficient_vector_deg3(original, ks)
    assert vec.shape == (10, 1)
    reconstructed = coefficient_vector_to_polynomial_deg3(vec, ks)
    assert sp.expand(reconstructed - original) == 0


def test_irrep_names_are_a2u_t2u_t1u() -> None:
    assert DEGREE3_IRREP_NAMES == ("A2u", "T2u", "T1u")
