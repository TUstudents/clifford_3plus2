"""Session 16 continuum-limit tests for the Cl(8) checkerboard walk."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.lepton.checkerboard import (
    checkerboard_gauge_shift_bloch,
    checkerboard_massless_floquet,
    edge_gauge_covariance_holds,
    expected_gauge_generator,
    expected_gauge_hamiltonian,
    expected_massless_generator,
    expected_massless_hamiltonian,
    floquet_eigenvalues_at,
    gauge_effective_generator,
    gauge_effective_hamiltonian,
    gauge_transform_link,
    has_gapless_eigenvalue_at,
    massless_effective_generator,
    massless_effective_hamiltonian,
    rigid_su3_gauge_transform,
    sample_gapless_momenta,
    su3_background_generator,
    su3_generator_is_valid,
)
from clifford_3plus2_d5.lepton.continuum import (
    effective_generator_from_floquet,
    first_order_in_epsilon,
    hamiltonian_from_real_skew_generator,
    matrix_zero,
)


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return matrix_zero(left - right)


def test_effective_generator_complex_convention() -> None:
    epsilon = sp.symbols("epsilon")
    hamiltonian = sp.diag(2, -3)
    floquet = sp.eye(2) - sp.I * epsilon * hamiltonian
    assert _same(
        effective_generator_from_floquet(
            floquet,
            epsilon=epsilon,
            convention="complex_hamiltonian",
        ),
        hamiltonian,
    )


def test_effective_generator_real_skew_convention() -> None:
    epsilon = sp.symbols("epsilon")
    generator = sp.Matrix([[0, 1], [-1, 0]])
    floquet = sp.eye(2) + epsilon * generator
    assert _same(
        effective_generator_from_floquet(
            floquet,
            epsilon=epsilon,
            convention="real_skew",
        ),
        generator,
    )
    assert _same(hamiltonian_from_real_skew_generator(generator), sp.I * generator)


def test_first_order_expansion_keeps_dimensionful_k_convention() -> None:
    epsilon, k = sp.symbols("epsilon k")
    phase = sp.Matrix([[sp.exp(-sp.I * k * epsilon)]])
    assert first_order_in_epsilon(phase, epsilon) == sp.Matrix([[-sp.I * k]])


def test_massless_checkerboard_generator_and_hamiltonian() -> None:
    epsilon, k = sp.symbols("epsilon k")
    assert _same(
        massless_effective_generator(epsilon, k),
        expected_massless_generator(k),
    )
    assert _same(
        massless_effective_hamiltonian(epsilon, k),
        expected_massless_hamiltonian(k),
    )


def test_massless_dispersion_eigenvalues_are_exactly_pm_k() -> None:
    k = sp.symbols("k")
    eigenvalues = expected_massless_hamiltonian(k).eigenvals()
    assert eigenvalues == {k: 8, -k: 8}


def test_su3_background_generator_is_skew_and_in_su3() -> None:
    generator = su3_background_generator(0)
    assert su3_generator_is_valid(generator)


def test_gauge_shift_generator_contains_background_connection() -> None:
    epsilon, k = sp.symbols("epsilon k")
    gauge_generator = su3_background_generator(0)
    floquet = checkerboard_gauge_shift_bloch(epsilon, k, gauge_generator)
    assert _same(
        gauge_effective_generator(epsilon, k, gauge_generator),
        expected_gauge_generator(k, gauge_generator),
    )
    assert _same(
        gauge_effective_hamiltonian(epsilon, k, gauge_generator),
        expected_gauge_hamiltonian(k, gauge_generator),
    )
    assert floquet.shape == (16, 16)


def test_finite_gauge_covariance_identity() -> None:
    gauge_left = rigid_su3_gauge_transform()
    gauge_right = rigid_su3_gauge_transform().T
    link = rigid_su3_gauge_transform()
    transformed = gauge_transform_link(link, gauge_left, gauge_right)
    assert transformed.shape == (8, 8)
    assert edge_gauge_covariance_holds(link, gauge_left, gauge_right)


def test_exact_floquet_gapless_samples() -> None:
    epsilon = sp.symbols("epsilon", positive=True)
    assert has_gapless_eigenvalue_at(epsilon, 0)
    assert not has_gapless_eigenvalue_at(epsilon, sp.pi / (2 * epsilon))
    assert not has_gapless_eigenvalue_at(epsilon, sp.pi / epsilon)
    assert not has_gapless_eigenvalue_at(epsilon, -sp.pi / (2 * epsilon))
    assert sample_gapless_momenta(epsilon) == (0,)
    assert set(floquet_eigenvalues_at(epsilon, sp.pi / epsilon)) == {-1}


def test_massless_floquet_shape() -> None:
    epsilon, k = sp.symbols("epsilon k")
    assert checkerboard_massless_floquet(epsilon, k).shape == (16, 16)
