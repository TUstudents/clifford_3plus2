"""JAX state helpers for numerical spacetime-QCA simulations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import sympy as sp

from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D
from clifford_3plus2_d5.spacetime_qca.state import State


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


def sympy_state_to_jax_flat(
    state: State,
    lattice: PeriodicLattice3D,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a JAX array with shape ``(nx, ny, nz, spinor_dim)``."""

    sites = lattice.sites()
    if set(state) != set(sites):
        raise ValueError("state support must match lattice sites")
    spinor_dims = {spinor.shape for spinor in state.values()}
    if len(spinor_dims) != 1:
        raise ValueError("all state spinors must have the same shape")
    ((spinor_dim, cols),) = spinor_dims
    if cols != 1:
        raise ValueError("state spinors must be column vectors")

    array = np.zeros((*lattice.shape, spinor_dim), dtype=np.complex64)
    for site in sites:
        spinor = state[site]
        array[site] = [sympy_scalar_to_complex(spinor[index, 0]) for index in range(spinor_dim)]
    return jnp.asarray(array, dtype=dtype)


def sympy_state_to_jax_dirac_internal(
    state: State,
    lattice: PeriodicLattice3D,
    *,
    internal_dim: int,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a tensor state with shape ``(nx, ny, nz, 4, internal_dim)``."""

    flat = sympy_state_to_jax_flat(state, lattice, dtype=dtype)
    if flat.shape[-1] != 4 * internal_dim:
        raise ValueError(f"state spinors must have dimension {4 * internal_dim}")
    return flat.reshape((*lattice.shape, 4, internal_dim))


def flatten_dirac_internal_state(state: jnp.ndarray) -> jnp.ndarray:
    """Flatten ``(..., 4, internal_dim)`` to ``(..., 4 * internal_dim)``."""

    if state.ndim < 2:
        raise ValueError("state must have at least two axes")
    if state.shape[-2] != 4:
        raise ValueError("dirac-internal state must have a 4-dimensional Dirac axis")
    return state.reshape((*state.shape[:-2], state.shape[-2] * state.shape[-1]))


def zero_jax_dirac_state(
    lattice_shape: tuple[int, int, int],
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a zero Dirac state with shape ``(nx, ny, nz, 4)``."""

    return jnp.zeros((*lattice_shape, 4), dtype=dtype)


def zero_jax_dirac_internal_state(
    lattice_shape: tuple[int, int, int],
    internal_dim: int,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a zero tensor state with shape ``(nx, ny, nz, 4, internal_dim)``."""

    return jnp.zeros((*lattice_shape, 4, internal_dim), dtype=dtype)
