"""JAX numerical kernels for the BCC Dirac spacetime QCA."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.links import jax_constant_link_field
from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    bialynicki_birula_directions,
    bialynicki_birula_hops,
    opposite_helicity_hops,
)
from clifford_3plus2_d5.spacetime_qca.dirac import block_diag


def jax_bcc_displacements() -> tuple[tuple[int, int, int], ...]:
    """Return BCC pull-step displacements in the SymPy backend order."""

    return tuple(
        tuple(int(component) for component in direction)  # type: ignore[misc]
        for direction in bialynicki_birula_directions()
    )


def jax_dirac_hops(*, dtype: Any = jnp.complex64) -> jnp.ndarray:
    """Return Dirac BCC hop matrices with shape ``(8, 4, 4)``."""

    matrices = tuple(
        block_diag(right, left)
        for right, left in zip(
            bialynicki_birula_hops(),
            opposite_helicity_hops(),
            strict=True,
        )
    )
    return jnp.asarray(
        [sympy_matrix_to_numpy(matrix) for matrix in matrices],
        dtype=dtype,
    )


def jax_dirac_step(state: jnp.ndarray) -> jnp.ndarray:
    """Apply the ungauged BCC Dirac step to ``(nx, ny, nz, 4)`` state arrays."""

    if state.ndim != 4 or state.shape[-1] != 4:
        raise ValueError("Dirac state must have shape (nx, ny, nz, 4)")
    hops = jax_dirac_hops(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(jax_bcc_displacements()):
        source = source_roll(state, displacement)
        out = out + jnp.einsum("ab,...b->...a", hops[index], source)
    return out


def jax_dirac_step_with_links(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Apply ``sum_h (W_h x U[x <- x+h]) psi[x+h]`` numerically.

    ``state`` has shape ``(nx, ny, nz, 4, internal_dim)``.  ``links`` has shape
    ``(nx, ny, nz, 8, internal_dim, internal_dim)`` in the same pull convention
    as the exact SymPy backend.
    """

    if state.ndim != 5 or state.shape[-2] != 4:
        raise ValueError("linked Dirac state must have shape (nx, ny, nz, 4, internal_dim)")
    if links.ndim != 6 or links.shape[:3] != state.shape[:3] or links.shape[3] != 8:
        raise ValueError("links must have shape (nx, ny, nz, 8, internal_dim, internal_dim)")
    if links.shape[-2:] != (state.shape[-1], state.shape[-1]):
        raise ValueError("link internal dimension must match state")

    hops = jax_dirac_hops(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(jax_bcc_displacements()):
        source = source_roll(state, displacement)
        linked = jnp.einsum("...ab,...sb->...sa", links[..., index, :, :], source)
        out = out + jnp.einsum("rs,...sd->...rd", hops[index], linked)
    return out


def jax_dirac_step_with_constant_link(state: jnp.ndarray, link: jnp.ndarray) -> jnp.ndarray:
    """Apply the linked step with the same internal link on every BCC edge."""

    return jax_dirac_step_with_links(
        state,
        jax_constant_link_field(state.shape[:3], link, dtype=state.dtype),
    )
