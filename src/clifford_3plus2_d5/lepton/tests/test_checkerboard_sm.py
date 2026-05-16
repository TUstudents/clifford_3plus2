"""Session 19a checkerboard tests for extracted SM background generators."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard_sm import (
    patisalam_edge_gauge_covariance_holds,
    patisalam_expected_massless_generator,
    patisalam_expected_massless_hamiltonian,
    patisalam_finite_spin04_transform,
    patisalam_floquet_eigenvalues_at,
    patisalam_gauge_transform_link,
    patisalam_has_gapless_eigenvalue_at,
    patisalam_massless_effective_generator,
    patisalam_massless_effective_hamiltonian,
    patisalam_sample_gapless_momenta,
    sm_background_generator,
    sm_background_generator_is_valid,
    sm_checkerboard_audit_payload,
    sm_checkerboard_gauge_shift_bloch,
    sm_expected_gauge_generator,
    sm_expected_gauge_hamiltonian,
    sm_gauge_effective_generator,
    sm_gauge_effective_hamiltonian,
)
from clifford_3plus2_d5.lepton.checkerboard_patisalam import (
    patisalam_checkerboard_massless_floquet,
)
from clifford_3plus2_d5.lepton.continuum import matrix_zero


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def test_sm_massless_checkerboard_is_chiral16_checkerboard() -> None:
    epsilon, k = sp.symbols("epsilon k")
    assert patisalam_checkerboard_massless_floquet(epsilon, k).shape == (64, 64)
    assert _same(
        patisalam_massless_effective_generator(epsilon, k),
        patisalam_expected_massless_generator(k),
    )
    assert _same(
        patisalam_massless_effective_hamiltonian(epsilon, k),
        patisalam_expected_massless_hamiltonian(k),
    )


def test_sm_background_generators_are_valid_and_j_compatible() -> None:
    assert sm_background_generator_is_valid("su3_c")
    assert sm_background_generator_is_valid("su2_l")
    assert sm_background_generator_is_valid("u1_y")


def test_sm_gauge_shift_generator_for_each_sector() -> None:
    epsilon, k = sp.symbols("epsilon k")
    for sector in ("su3_c", "su2_l", "u1_y"):
        gauge_generator = sm_background_generator(sector)
        floquet = sm_checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)
        assert floquet.shape == (64, 64)
        assert _same(
            sm_gauge_effective_generator(epsilon, k, gauge_generator),
            sm_expected_gauge_generator(k, gauge_generator),
        )
        assert _same(
            sm_gauge_effective_hamiltonian(epsilon, k, gauge_generator),
            sm_expected_gauge_hamiltonian(k, gauge_generator),
        )


def test_sm_finite_gauge_covariance_identity() -> None:
    gauge_left = patisalam_finite_spin04_transform()
    gauge_right = patisalam_finite_spin04_transform().T
    link = patisalam_finite_spin04_transform()
    transformed = patisalam_gauge_transform_link(link, gauge_left, gauge_right)
    assert transformed.shape == (32, 32)
    assert patisalam_edge_gauge_covariance_holds(link, gauge_left, gauge_right)


def test_sm_exact_floquet_gapless_samples() -> None:
    epsilon = sp.symbols("epsilon", positive=True)
    assert patisalam_has_gapless_eigenvalue_at(epsilon, 0)
    assert not patisalam_has_gapless_eigenvalue_at(epsilon, sp.pi / (2 * epsilon))
    assert not patisalam_has_gapless_eigenvalue_at(epsilon, sp.pi / epsilon)
    assert not patisalam_has_gapless_eigenvalue_at(
        epsilon,
        -sp.pi / (2 * epsilon),
    )
    assert patisalam_sample_gapless_momenta(epsilon) == (0,)
    assert set(patisalam_floquet_eigenvalues_at(epsilon, sp.pi / epsilon)) == {-1}


def test_sm_checkerboard_audit_payload_is_stable() -> None:
    payload = sm_checkerboard_audit_payload()
    assert payload["internal_real_dimension"] == 32
    assert payload["massless_floquet_shape"] == (64, 64)
    assert payload["massless_hamiltonian_eigenvalues"] == {sp.symbols("k"): 32, -sp.symbols("k"): 32}
    assert payload["su3_c_background_valid"] is True
    assert payload["su2_l_background_valid"] is True
    assert payload["u1_y_background_valid"] is True
    assert payload["gapless_sample_momenta"] == (0,)
