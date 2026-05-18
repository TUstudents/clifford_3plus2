"""Generic state-layout helpers for JAX simulation kernels."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy as sp

from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE


def sympy_scalar_to_complex(value: sp.Expr) -> complex:
    """Convert an exact SymPy scalar to a Python complex number."""

    return complex(sp.N(value))


def sympy_matrix_to_numpy(matrix: sp.Matrix, *, dtype: Any = np.complex64) -> np.ndarray:
    """Convert a SymPy matrix to a dense NumPy complex array."""

    return np.array(
        [
            [sympy_scalar_to_complex(matrix[row, col]) for col in range(matrix.cols)]
            for row in range(matrix.rows)
        ],
        dtype=dtype,
    )


def flatten_dirac_internal_state(state: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``(..., 4, internal_dim)`` to ``(..., 4 * internal_dim)``."""

    if state.ndim < 2:
        raise ValueError("state must have at least two axes")
    if state.shape[-2] != 4:
        raise ValueError("dirac-internal state must have a 4-dimensional Dirac axis")
    return state.reshape((*state.shape[:-2], state.shape[-2] * state.shape[-1]))


def unflatten_dirac_internal_state(state: jnp.ndarray, *, internal_dim: int) -> jnp.ndarray:
    """Unflatten ``(..., 4 * internal_dim)`` to ``(..., 4, internal_dim)``."""

    if state.ndim < 1:
        raise ValueError("state must have at least one axis")
    if state.shape[-1] != 4 * internal_dim:
        raise ValueError(f"last axis must have dimension {4 * internal_dim}")
    return state.reshape((*state.shape[:-1], 4, internal_dim))


def zero_jax_dirac_state(
    lattice_shape: tuple[int, int, int],
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a zero Dirac state with shape ``(nx, ny, nz, 4)``."""

    return jnp.zeros((*lattice_shape, 4), dtype=dtype)


def zero_jax_dirac_internal_state(
    lattice_shape: tuple[int, int, int],
    internal_dim: int,
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a zero tensor state with shape ``(nx, ny, nz, 4, internal_dim)``."""

    return jnp.zeros((*lattice_shape, 4, internal_dim), dtype=dtype)


def state_norm_squared(state: jnp.ndarray) -> jnp.ndarray:
    """Return the squared complex norm of a JAX state array."""

    return jnp.real(jnp.vdot(state, state))


def norm_drift(before: jnp.ndarray, after: jnp.ndarray) -> jnp.ndarray:
    """Return ``||after||^2 - ||before||^2``."""

    return state_norm_squared(after) - state_norm_squared(before)
