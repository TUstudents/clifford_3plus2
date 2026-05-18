"""JAX link-field helpers for numerical BCC steps."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.links import jax_constant_link_field, jax_identity_link_field
from clifford_3plus2_d5.spacetime_qca.jax_state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D
from clifford_3plus2_d5.spacetime_qca.links import LinkField, bcc_link_displacements, validate_link_field

__all__ = [
    "jax_constant_link_field",
    "jax_identity_link_field",
    "sympy_link_field_to_jax",
]


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
