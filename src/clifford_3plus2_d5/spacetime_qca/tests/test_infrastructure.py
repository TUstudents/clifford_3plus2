"""Infrastructure tests for the spacetime QCA package."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    alpha_matrices,
    background_gauge_hamiltonian,
    bcc_body_diagonal_directions,
    dirac_hamiltonian,
    effective_generator_from_floquet,
    first_order_in_epsilon,
    gamma0,
    gamma5,
    gamma_matrices,
    hypercube_bz_corners,
    hypercube_continuum_target,
    hypercube_corner_eigenvalues,
    hypercube_gapless_corners,
    infrastructure_audit_payload,
    is_hermitian,
    is_unitary,
    matrix_zero,
    naive_hypercube_hamiltonian,
    pauli_matrices,
    same_matrix,
    sigma_x,
    squared_norm,
    validate_weyl_hops,
    weyl_bloch_symbol_from_hops,
)


def test_pauli_matrices_are_hermitian_unitary_and_anticommute() -> None:
    sx, sy, sz = pauli_matrices()
    for sigma in (sx, sy, sz):
        assert is_hermitian(sigma)
        assert is_unitary(sigma)
        assert same_matrix(sigma * sigma, sp.eye(2))

    assert matrix_zero(sx * sy + sy * sx)
    assert matrix_zero(sx * sz + sz * sx)
    assert matrix_zero(sy * sz + sz * sy)


def test_bcc_body_diagonal_geometry() -> None:
    directions = bcc_body_diagonal_directions()
    assert len(directions) == 8
    assert len(set(directions)) == 8
    assert all(sp.simplify(squared_norm(direction) - 1) == 0 for direction in directions)

    unnormalized = bcc_body_diagonal_directions(normalized=False)
    assert all(squared_norm(direction) == 3 for direction in unnormalized)


def test_chiral_dirac_matrices_have_expected_clifford_relations() -> None:
    gammas = gamma_matrices()
    metric = (1, -1, -1, -1)

    for index, gamma in enumerate(gammas):
        assert same_matrix(gamma * gamma, metric[index] * sp.eye(4))

    for i in range(4):
        for j in range(i + 1, 4):
            assert matrix_zero(gammas[i] * gammas[j] + gammas[j] * gammas[i])

    assert same_matrix(gamma0() * gamma0(), sp.eye(4))
    assert same_matrix(gamma5(), sp.diag(1, 1, -1, -1))


def test_alpha_matrices_are_chiral_block_diagonal() -> None:
    sx = sigma_x()
    ax, ay, az = alpha_matrices()

    assert same_matrix(ax[:2, :2], sx)
    assert same_matrix(ax[2:, 2:], -sx)
    for alpha in (ax, ay, az):
        assert is_hermitian(alpha)
        assert same_matrix(alpha * alpha, sp.eye(4))


def test_dirac_hamiltonian_has_pm_abs_k_axis_spectrum() -> None:
    k = sp.symbols("k")
    hamiltonian = dirac_hamiltonian(k, 0, 0)
    assert hamiltonian.eigenvals() == {k: 2, -k: 2}


def test_first_order_and_effective_hamiltonian_convention() -> None:
    epsilon, k = sp.symbols("epsilon k")
    phase = sp.Matrix([[sp.exp(-sp.I * k * epsilon)]])
    assert first_order_in_epsilon(phase, epsilon) == sp.Matrix([[-sp.I * k]])

    hamiltonian = sp.diag(2, -3)
    floquet = sp.eye(2) - sp.I * epsilon * hamiltonian
    assert same_matrix(
        effective_generator_from_floquet(
            floquet,
            epsilon=epsilon,
            convention="complex_hamiltonian",
        ),
        hamiltonian,
    )


def test_hypercube_control_has_all_eight_corner_doublers() -> None:
    epsilon = sp.symbols("epsilon", positive=True)
    corners = hypercube_bz_corners(epsilon)
    assert len(corners) == 8
    assert set(hypercube_gapless_corners(epsilon)) == set(corners)
    assert all(hypercube_corner_eigenvalues(epsilon, corner) == {0: 4} for corner in corners)

    kx, ky, kz = sp.symbols("kx ky kz")
    continuum_difference = naive_hypercube_hamiltonian(epsilon, kx, ky, kz).applyfunc(
        lambda value: sp.series(value, epsilon, 0, 2).removeO(),
    ) - hypercube_continuum_target(kx, ky, kz)
    assert matrix_zero(continuum_difference)


def test_background_gauge_lift_adds_internal_hamiltonian_term() -> None:
    k = sp.symbols("k")
    h_space = dirac_hamiltonian(k, 0, 0)
    generator = sp.Matrix([[0, 1], [-1, 0]])

    lifted = background_gauge_hamiltonian(h_space, generator)
    expected = sp.kronecker_product(h_space, sp.eye(2)) + sp.kronecker_product(
        sp.eye(4),
        sp.I * generator,
    )
    assert same_matrix(lifted, expected)


def test_bcc_weyl_symbol_accepts_supplied_hops_only() -> None:
    epsilon, kx, ky, kz = sp.symbols("epsilon kx ky kz")
    hops = tuple(sp.zeros(2) for _ in range(8))
    symbol = weyl_bloch_symbol_from_hops(epsilon, (kx, ky, kz), hops)
    assert symbol == sp.zeros(2)

    with pytest.raises(ValueError, match="exactly 8"):
        validate_weyl_hops(tuple(sp.zeros(2) for _ in range(7)))


def test_infrastructure_audit_payload_records_intentional_gap() -> None:
    payload = infrastructure_audit_payload()
    assert payload.bcc_direction_count == 8
    assert payload.dirac_dimension == 4
    assert payload.hypercube_corner_count == 8
    assert any("not guessed" in note for note in payload.notes)
