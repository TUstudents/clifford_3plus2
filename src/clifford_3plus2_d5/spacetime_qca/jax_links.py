"""JAX link-field helpers for numerical BCC steps."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca.jax_state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D
from clifford_3plus2_d5.spacetime_qca.links import LinkField, bcc_link_displacements, validate_link_field


def jax_identity_link_field(
    lattice_shape: tuple[int, int, int],
    internal_dim: int,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return identity links with shape ``(nx, ny, nz, 8, d, d)``."""

    identity = jnp.eye(internal_dim, dtype=dtype)
    return jnp.broadcast_to(identity, (*lattice_shape, 8, internal_dim, internal_dim))


def jax_constant_link_field(
    lattice_shape: tuple[int, int, int],
    link: jnp.ndarray,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Return a constant link field with shape ``(nx, ny, nz, 8, d, d)``."""

    link = jnp.asarray(link, dtype=dtype)
    if link.ndim != 2 or link.shape[0] != link.shape[1]:
        raise ValueError("link must be a square matrix")
    return jnp.broadcast_to(link, (*lattice_shape, 8, link.shape[0], link.shape[1]))


def sympy_link_field_to_jax(
    links: LinkField,
    lattice: PeriodicLattice3D,
    *,
    dtype: Any = jnp.complex64,
) -> jnp.ndarray:
    """Convert an exact link field to ``(nx, ny, nz, 8, d, d)`` JAX layout."""

    internal_dim = validate_link_field(links, lattice)
    displacements = bcc_link_displacements()
    array = np.zeros((*lattice.shape, len(displacements), internal_dim, internal_dim), dtype=np.complex64)
    for site in lattice.sites():
        for index, displacement in enumerate(displacements):
            array[(*site, index, slice(None), slice(None))] = sympy_matrix_to_numpy(
                links[(site, displacement)],
            )
    return jnp.asarray(array, dtype=dtype)
