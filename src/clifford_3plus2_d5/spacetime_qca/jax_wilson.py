"""JAX Wilson plaquette observables for BCC link fields."""

from __future__ import annotations

import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_step import jax_bcc_displacements
from clifford_3plus2_d5.spacetime_qca.lattice import Site
from clifford_3plus2_d5.spacetime_qca.plaquette import (
    PlaquetteShape,
    canonical_bcc_plaquette_shapes,
    is_elementary_bcc_plaquette_shape,
)


def _validate_jax_links(links: jnp.ndarray) -> int:
    if links.ndim != 6 or links.shape[3] != 8:
        raise ValueError("links must have shape (nx, ny, nz, 8, internal_dim, internal_dim)")
    if links.shape[-1] != links.shape[-2]:
        raise ValueError("link matrices must be square")
    return int(links.shape[-1])


def _wrap_site(site: Site, lattice_shape: tuple[int, int, int]) -> Site:
    return tuple(coord % size for coord, size in zip(site, lattice_shape, strict=True))  # type: ignore[return-value]


def _translate(site: Site, displacement: tuple[int, int, int], lattice_shape: tuple[int, int, int]) -> Site:
    return _wrap_site(
        tuple(coord + step for coord, step in zip(site, displacement, strict=True)),  # type: ignore[arg-type]
        lattice_shape,
    )


def _displacement_index(displacement: tuple[int, int, int]) -> int:
    return jax_bcc_displacements().index(displacement)


def jax_plaquette_holonomy(
    links: jnp.ndarray,
    base_site: Site,
    shape: PlaquetteShape,
) -> jnp.ndarray:
    """Return the ordered BCC plaquette holonomy from JAX link layout."""

    internal_dim = _validate_jax_links(links)
    if not is_elementary_bcc_plaquette_shape(shape):
        raise ValueError("shape must be an elementary BCC plaquette")

    lattice_shape = tuple(int(size) for size in links.shape[:3])  # type: ignore[assignment]
    current = _wrap_site(base_site, lattice_shape)
    holonomy = jnp.eye(internal_dim, dtype=links.dtype)
    for displacement in shape:
        index = _displacement_index(displacement)
        holonomy = holonomy @ links[(*current, index, slice(None), slice(None))]
        current = _translate(current, displacement, lattice_shape)
    if current != _wrap_site(base_site, lattice_shape):
        raise ValueError("plaquette path is not closed on this lattice")
    return holonomy


def jax_wilson_loop_trace(
    links: jnp.ndarray,
    base_site: Site,
    shape: PlaquetteShape,
) -> jnp.ndarray:
    """Return ``Tr(H[x0, shape])`` for one JAX BCC plaquette."""

    return jnp.trace(jax_plaquette_holonomy(links, base_site, shape))


def jax_normalized_wilson_loop(
    links: jnp.ndarray,
    base_site: Site,
    shape: PlaquetteShape,
) -> jnp.ndarray:
    """Return ``Tr(H) / internal_dim`` for one JAX BCC plaquette."""

    internal_dim = _validate_jax_links(links)
    return jax_wilson_loop_trace(links, base_site, shape) / internal_dim


def jax_average_normalized_wilson_loop(
    links: jnp.ndarray,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> jnp.ndarray:
    """Return the uniform site/shape average of the normalized Wilson loop."""

    _validate_jax_links(links)
    selected_shapes = shapes or canonical_bcc_plaquette_shapes()
    if not selected_shapes:
        raise ValueError("at least one plaquette shape is required")

    total = jnp.asarray(0, dtype=links.dtype)
    count = 0
    nx, ny, nz = (int(size) for size in links.shape[:3])
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for shape in selected_shapes:
                    total = total + jax_normalized_wilson_loop(links, (x, y, z), shape)
                    count += 1
    return total / count
