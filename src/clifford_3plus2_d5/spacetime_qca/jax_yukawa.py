"""JAX adapters for the static Hermitian Yukawa ``Y(Phi)`` layer.

The exact Session 38 construction lives in :mod:`spacetime_qca.yukawa`.
This module mirrors the same deterministic two-complex Higgs slice for
numerical kernels: ``phi_plus`` and ``phi_zero`` are native complex scalars,
split into real and imaginary coordinates against the selected real-form
basis maps.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.mass import beta_matrix
from clifford_3plus2_d5.spacetime_qca.yukawa import selected_higgs_phi_basis

__all__ = [
    "jax_hermitian_yukawa_hamiltonian",
    "jax_hermitian_yukawa_internal_control",
    "jax_neutral_yukawa_hamiltonian",
    "jax_neutral_yukawa_internal_control",
]


@lru_cache(maxsize=8)
def _selected_higgs_phi_basis_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return np.stack(
        [sympy_matrix_to_numpy(matrix, dtype=dtype) for matrix in selected_higgs_phi_basis()],
        axis=0,
    )


@lru_cache(maxsize=8)
def _beta_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return sympy_matrix_to_numpy(beta_matrix(), dtype=dtype)


def jax_hermitian_yukawa_internal_control(
    phi_plus: Any = 0.0 + 0.0j,
    phi_zero: Any = 1.0 + 0.0j,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return the static internal ``Y(Phi)`` matrix as a JAX array."""

    dtype = np.dtype(dtype)
    basis = jnp.asarray(_selected_higgs_phi_basis_numpy(dtype.name), dtype=dtype)
    plus = jnp.asarray(phi_plus, dtype=dtype)
    zero = jnp.asarray(phi_zero, dtype=dtype)
    coefficients = jnp.stack((jnp.real(plus), jnp.imag(plus), jnp.real(zero), jnp.imag(zero))).astype(dtype)
    raising = jnp.tensordot(coefficients, basis, axes=(0, 0))
    return raising + jnp.swapaxes(raising, -1, -2)


def jax_hermitian_yukawa_hamiltonian(
    phi_plus: Any = 0.0 + 0.0j,
    phi_zero: Any = 1.0 + 0.0j,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return ``beta x Y(Phi)`` as a JAX array."""

    dtype = np.dtype(dtype)
    beta = jnp.asarray(_beta_numpy(dtype.name), dtype=dtype)
    internal = jax_hermitian_yukawa_internal_control(phi_plus, phi_zero, dtype=dtype)
    return jnp.kron(beta, internal)


def jax_neutral_yukawa_internal_control(
    vev: Any = 1.0,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return the neutral static Higgs direction ``Phi = (0, vev)``."""

    return jax_hermitian_yukawa_internal_control(0.0 + 0.0j, vev + 0.0j, dtype=dtype)


def jax_neutral_yukawa_hamiltonian(
    vev: Any = 1.0,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return ``beta x Y(Phi)`` for the neutral static Higgs direction."""

    return jax_hermitian_yukawa_hamiltonian(0.0 + 0.0j, vev + 0.0j, dtype=dtype)
