"""Generic JAX link-field helpers for pull-form lattice simulations."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE
from clifford_3plus2_d5.sim.lattice import Displacement3D, source_roll, validate_displacements


def validate_link_field(links: jnp.ndarray, *, edge_count: int | None = None) -> int:
    """Validate ``(..., edge, d, d)`` link-field layout and return ``d``."""

    if links.ndim < 3:
        raise ValueError("links must have at least edge and matrix axes")
    if edge_count is not None and links.shape[-3] != edge_count:
        raise ValueError(f"links must have {edge_count} edge directions")
    if links.shape[-1] != links.shape[-2]:
        raise ValueError("link matrices must be square")
    return int(links.shape[-1])


def jax_identity_link_field(
    lattice_shape: tuple[int, int, int],
    internal_dim: int,
    *,
    edge_count: int = 8,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return identity links with shape ``(nx, ny, nz, edge_count, d, d)``."""

    identity = jnp.eye(internal_dim, dtype=dtype)
    return jnp.broadcast_to(identity, (*lattice_shape, edge_count, internal_dim, internal_dim))


def jax_constant_link_field(
    lattice_shape: tuple[int, int, int],
    link: jnp.ndarray,
    *,
    edge_count: int = 8,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a constant link field with shape ``(nx, ny, nz, edge_count, d, d)``."""

    link = jnp.asarray(link, dtype=dtype)
    if link.ndim != 2 or link.shape[0] != link.shape[1]:
        raise ValueError("link must be a square matrix")
    return jnp.broadcast_to(link, (*lattice_shape, edge_count, link.shape[0], link.shape[1]))


def jax_transform_link_field(
    links: jnp.ndarray,
    site_gauge: jnp.ndarray,
    displacements: tuple[Displacement3D, ...],
) -> jnp.ndarray:
    """Apply finite site-local gauge transforms to pull-convention links.

    ``links[..., h, :, :]`` stores the link from source ``x+h`` to target
    ``x``.  The transform is ``U[x,h] -> G[x] U[x,h] G[x+h]^dagger``.
    """

    validate_displacements(displacements, expected_count=int(links.shape[-3]))
    if links.ndim != 6:
        raise ValueError("links must have shape (nx, ny, nz, edge_count, internal_dim, internal_dim)")
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != links.shape[:3]:
        raise ValueError("site_gauge must have shape (nx, ny, nz, internal_dim, internal_dim)")
    if links.shape[-2:] != site_gauge.shape[-2:]:
        raise ValueError("site gauges and link matrices must have matching square dimensions")

    transformed = []
    for index, displacement in enumerate(displacements):
        source_gauge = source_roll(site_gauge, displacement)
        source_dagger = jnp.swapaxes(jnp.conj(source_gauge), -1, -2)
        transformed.append(
            jnp.einsum(
                "...ab,...bc,...cd->...ad",
                site_gauge,
                links[..., index, :, :],
                source_dagger,
            ),
        )
    return jnp.stack(transformed, axis=3)
