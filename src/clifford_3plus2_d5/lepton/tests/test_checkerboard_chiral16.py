"""Session 17 continuum tests for the Cl(0,10) chiral-16 checkerboard."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard_chiral16 import (
    chiral16_background_su3_is_valid,
    chiral16_checkerboard_audit_payload,
    chiral16_checkerboard_gauge_shift_bloch,
    chiral16_checkerboard_massless_floquet,
    chiral16_edge_gauge_covariance_holds,
    chiral16_expected_gauge_generator,
    chiral16_expected_gauge_hamiltonian,
    chiral16_expected_massless_generator,
    chiral16_expected_massless_hamiltonian,
    chiral16_floquet_eigenvalues_at,
    chiral16_gauge_effective_generator,
    chiral16_gauge_effective_hamiltonian,
    chiral16_gauge_transform_link,
    chiral16_has_gapless_eigenvalue_at,
    chiral16_massless_effective_generator,
    chiral16_massless_effective_hamiltonian,
    chiral16_rigid_su3_gauge_transform,
    chiral16_sample_gapless_momenta,
    chiral16_su3_background_generator,
)
from clifford_3plus2_d5.lepton.continuum import matrix_zero


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def test_chiral16_massless_checkerboard_shape_and_generator() -> None:
    epsilon, k = sp.symbols("epsilon k")
    assert chiral16_checkerboard_massless_floquet(epsilon, k).shape == (64, 64)
    assert _same(
        chiral16_massless_effective_generator(epsilon, k),
        chiral16_expected_massless_generator(k),
    )
    assert _same(
        chiral16_massless_effective_hamiltonian(epsilon, k),
        chiral16_expected_massless_hamiltonian(k),
    )


def test_chiral16_massless_dispersion_eigenvalues_are_exactly_pm_k() -> None:
    k = sp.symbols("k")
    assert chiral16_expected_massless_hamiltonian(k).eigenvals() == {k: 32, -k: 32}


def test_lifted_su3_background_generator_is_valid() -> None:
    assert chiral16_background_su3_is_valid()


def test_chiral16_gauge_shift_generator_contains_background_connection() -> None:
    epsilon, k = sp.symbols("epsilon k")
    gauge_generator = chiral16_su3_background_generator()
    floquet = chiral16_checkerboard_gauge_shift_bloch(
        epsilon,
        k,
        gauge_generator,
    )
    assert floquet.shape == (64, 64)
    assert _same(
        chiral16_gauge_effective_generator(epsilon, k, gauge_generator),
        chiral16_expected_gauge_generator(k, gauge_generator),
    )
    assert _same(
        chiral16_gauge_effective_hamiltonian(epsilon, k, gauge_generator),
        chiral16_expected_gauge_hamiltonian(k, gauge_generator),
    )


def test_chiral16_finite_su3_gauge_covariance_identity() -> None:
    gauge_left = chiral16_rigid_su3_gauge_transform()
    gauge_right = chiral16_rigid_su3_gauge_transform().T
    link = chiral16_rigid_su3_gauge_transform()
    transformed = chiral16_gauge_transform_link(link, gauge_left, gauge_right)
    assert transformed.shape == (32, 32)
    assert chiral16_edge_gauge_covariance_holds(link, gauge_left, gauge_right)


def test_chiral16_exact_floquet_gapless_samples() -> None:
    epsilon = sp.symbols("epsilon", positive=True)
    assert chiral16_has_gapless_eigenvalue_at(epsilon, 0)
    assert not chiral16_has_gapless_eigenvalue_at(epsilon, sp.pi / (2 * epsilon))
    assert not chiral16_has_gapless_eigenvalue_at(epsilon, sp.pi / epsilon)
    assert not chiral16_has_gapless_eigenvalue_at(
        epsilon,
        -sp.pi / (2 * epsilon),
    )
    assert chiral16_sample_gapless_momenta(epsilon) == (0,)
    assert set(chiral16_floquet_eigenvalues_at(epsilon, sp.pi / epsilon)) == {-1}


def test_chiral16_checkerboard_audit_payload_is_stable() -> None:
    payload = chiral16_checkerboard_audit_payload()
    assert payload["internal_real_dimension"] == 32
    assert payload["massless_floquet_shape"] == (64, 64)
    assert payload["massless_hamiltonian_eigenvalues"] == {sp.symbols("k"): 32, -sp.symbols("k"): 32}
    assert payload["su3_background_valid"] is True
    assert payload["su3_commutes_with_chosen_j"] is True
    assert payload["gapless_sample_momenta"] == (0,)
    assert "SU(2)_L is not supplied" in str(payload["ew_note"])
