"""BCC plaquette geometry and background-link holonomies.

Session 24b keeps the Session 24 pull-link convention.  A plaquette path is a
closed sequence of BCC displacements ``(h_0, ..., h_3)``.  The corresponding
holonomy based at ``x_0`` is

``U[x_0 <- x_1] U[x_1 <- x_2] U[x_2 <- x_3] U[x_3 <- x_0]``.

Under ``U[x <- y] -> G[x] U[x <- y] G[y]^-1``, this transforms by conjugation
at the base site.
"""

from __future__ import annotations

from typing import TypeAlias

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.lattice import Displacement, PeriodicLattice3D, Site
from clifford_3plus2_d5.spacetime_qca.links import (
    GaugeTransform,
    LinkField,
    bcc_link_displacements,
    transform_link_field,
    validate_gauge_transform,
    validate_link_field,
)

PlaquetteShape: TypeAlias = tuple[Displacement, Displacement, Displacement, Displacement]
PlaquetteKey: TypeAlias = tuple[Site, PlaquetteShape]


def negate_displacement(displacement: Displacement) -> Displacement:
    return tuple(-component for component in displacement)  # type: ignore[return-value]


def add_displacements(left: Displacement, right: Displacement) -> Displacement:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def plaquette_vertices_unwrapped(shape: PlaquetteShape) -> tuple[Displacement, ...]:
    vertices: list[Displacement] = [(0, 0, 0)]
    current = (0, 0, 0)
    for displacement in shape:
        current = add_displacements(current, displacement)
        vertices.append(current)
    return tuple(vertices)


def is_elementary_bcc_plaquette_shape(shape: PlaquetteShape) -> bool:
    """Return whether ``shape`` is a non-backtracking four-hop BCC loop."""

    if len(shape) != 4:
        return False
    allowed = set(bcc_link_displacements())
    if any(displacement not in allowed for displacement in shape):
        return False
    if any(shape[(index + 1) % 4] == negate_displacement(shape[index]) for index in range(4)):
        return False
    vertices = plaquette_vertices_unwrapped(shape)
    return vertices[-1] == (0, 0, 0) and len(set(vertices[:-1])) == 4


def _rotations(shape: PlaquetteShape) -> tuple[PlaquetteShape, ...]:
    return tuple(shape[index:] + shape[:index] for index in range(4))  # type: ignore[return-value]


def _reverse_shape(shape: PlaquetteShape) -> PlaquetteShape:
    return tuple(negate_displacement(displacement) for displacement in reversed(shape))  # type: ignore[return-value]


def _canonical_shape(shape: PlaquetteShape) -> PlaquetteShape:
    candidates = _rotations(shape) + _rotations(_reverse_shape(shape))
    return min(candidates)


def canonical_bcc_plaquette_shapes() -> tuple[PlaquetteShape, ...]:
    """Return the six unoriented elementary BCC four-hop plaquette shapes."""

    shapes: set[PlaquetteShape] = set()
    directions = bcc_link_displacements()
    for first in directions:
        for second in directions:
            shape: PlaquetteShape = (
                first,
                second,
                negate_displacement(first),
                negate_displacement(second),
            )
            if is_elementary_bcc_plaquette_shape(shape):
                shapes.add(_canonical_shape(shape))
    return tuple(sorted(shapes))


def plaquette_path_sites(
    lattice: PeriodicLattice3D,
    base_site: Site,
    shape: PlaquetteShape,
) -> tuple[Site, Site, Site, Site, Site]:
    """Return wrapped path sites ``(x_0, ..., x_4)`` for a closed plaquette."""

    if not is_elementary_bcc_plaquette_shape(shape):
        raise ValueError("shape must be an elementary BCC plaquette")
    sites: list[Site] = [lattice.wrap(base_site)]
    current = lattice.wrap(base_site)
    for displacement in shape:
        current = lattice.translate(current, displacement)
        sites.append(current)
    if sites[-1] != sites[0]:
        raise ValueError("plaquette path is not closed on this lattice")
    return tuple(sites)  # type: ignore[return-value]


def plaquette_holonomy(
    links: LinkField,
    lattice: PeriodicLattice3D,
    base_site: Site,
    shape: PlaquetteShape,
) -> sp.Matrix:
    """Return the ordered product around one BCC plaquette."""

    internal_dim = validate_link_field(links, lattice)
    sites = plaquette_path_sites(lattice, base_site, shape)
    holonomy = sp.eye(internal_dim)
    for target, displacement in zip(sites[:-1], shape, strict=True):
        holonomy *= links[(target, displacement)]
    return holonomy.applyfunc(sp.simplify)


def plaquette_holonomy_covariance_residual(
    links: LinkField,
    lattice: PeriodicLattice3D,
    gauge: GaugeTransform,
    base_site: Site,
    shape: PlaquetteShape,
) -> sp.Matrix:
    """Return ``H[GUG^-1] - G[base] H[U] G[base]^-1``."""

    internal_dim = validate_gauge_transform(gauge, lattice)
    validate_link_field(links, lattice, internal_dim=internal_dim)
    base = lattice.wrap(base_site)
    transformed_links = transform_link_field(links, lattice, gauge)
    transformed_holonomy = plaquette_holonomy(transformed_links, lattice, base, shape)
    expected = gauge[base] * plaquette_holonomy(links, lattice, base, shape) * gauge[base].inv()
    return (transformed_holonomy - expected).applyfunc(sp.simplify)
