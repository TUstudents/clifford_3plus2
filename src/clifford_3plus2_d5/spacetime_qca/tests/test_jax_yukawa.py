"""JAX parity tests for the Session 38 Hermitian Yukawa layer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca import (
    hermitian_yukawa_hamiltonian,
    hermitian_yukawa_internal_control,
)
from clifford_3plus2_d5.spacetime_qca.jax_yukawa import (
    jax_hermitian_yukawa_hamiltonian,
    jax_hermitian_yukawa_internal_control,
    jax_neutral_yukawa_hamiltonian,
    jax_neutral_yukawa_internal_control,
)

pytestmark = pytest.mark.slow


def test_jax_hermitian_yukawa_shapes_and_zero_phi() -> None:
    internal = jax_hermitian_yukawa_internal_control(0.0 + 0.0j, 0.0 + 0.0j)
    hamiltonian = jax_hermitian_yukawa_hamiltonian(0.0 + 0.0j, 0.0 + 0.0j)
    assert internal.shape == (32, 32)
    assert hamiltonian.shape == (128, 128)
    np.testing.assert_allclose(np.asarray(internal), np.zeros((32, 32), dtype=np.complex64))
    np.testing.assert_allclose(np.asarray(hamiltonian), np.zeros((128, 128), dtype=np.complex64))


def test_jax_hermitian_yukawa_matches_sympy_selected_phi() -> None:
    phi_plus = 2.0 - 3.0j
    phi_zero = 2.5 + (7.0 / 3.0) * 1j
    jax_internal = np.asarray(jax_hermitian_yukawa_internal_control(phi_plus, phi_zero))
    sympy_internal = sympy_matrix_to_numpy(
        hermitian_yukawa_internal_control(
            phi_plus=(sp.Integer(2), sp.Integer(-3)),
            phi_zero=(sp.Rational(5, 2), sp.Rational(7, 3)),
        ),
        dtype=np.complex64,
    )
    np.testing.assert_allclose(jax_internal, sympy_internal, atol=1e-6)


def test_jax_hermitian_yukawa_hamiltonian_matches_sympy() -> None:
    phi_plus = -1.0 + 0.5j
    phi_zero = 3.0 - 2.0j
    jax_hamiltonian = np.asarray(jax_hermitian_yukawa_hamiltonian(phi_plus, phi_zero))
    sympy_hamiltonian = sympy_matrix_to_numpy(
        hermitian_yukawa_hamiltonian(
            phi_plus=(sp.Integer(-1), sp.Rational(1, 2)),
            phi_zero=(sp.Integer(3), sp.Integer(-2)),
        ),
        dtype=np.complex64,
    )
    np.testing.assert_allclose(jax_hamiltonian, sympy_hamiltonian, atol=1e-6)


def test_jax_neutral_yukawa_is_hermitian() -> None:
    internal = jax_neutral_yukawa_internal_control(1.25)
    hamiltonian = jax_neutral_yukawa_hamiltonian(1.25)
    assert bool(jnp.all(jnp.isfinite(internal)))
    assert bool(jnp.all(jnp.isfinite(hamiltonian)))
    np.testing.assert_allclose(np.asarray(internal), np.asarray(internal.conj().T), atol=1e-6)
    np.testing.assert_allclose(np.asarray(hamiltonian), np.asarray(hamiltonian.conj().T), atol=1e-6)
