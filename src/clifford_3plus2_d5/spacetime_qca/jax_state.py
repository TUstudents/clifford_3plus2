"""JAX state helpers for numerical spacetime-QCA simulations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.state import (
    flatten_dirac_internal_state,
    sympy_matrix_to_numpy,
    sympy_scalar_to_complex,
    zero_jax_dirac_internal_state,
    zero_jax_dirac_state,
)
from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D
from clifford_3plus2_d5.spacetime_qca.state import State

__all__ = [
    "flatten_dirac_internal_state",
    "sympy_matrix_to_numpy",
    "sympy_scalar_to_complex",
    "sympy_state_to_jax_dirac_internal",
    "sympy_state_to_jax_flat",
    "zero_jax_dirac_internal_state",
    "zero_jax_dirac_state",
]


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
