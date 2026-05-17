"""Position-dependent internal link fields for finite BCC steps.

The finite BCC step uses pull form:

``out[x] = sum_h (W_h x U[x <- x+h]) psi[x+h]``.

A link-field key is therefore ``(target_site, displacement)``.  The source
site is ``target_site + displacement`` with periodic wrapping.  Under a
site-local internal gauge transform ``G[x]``:

``psi[x] -> (I_space x G[x]) psi[x]``
``U[x <- y] -> G[x] U[x <- y] G[y]^-1``.
"""

from __future__ import annotations

from typing import TypeAlias

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_weyl import bialynicki_birula_directions
from clifford_3plus2_d5.spacetime_qca.lattice import Displacement, PeriodicLattice3D, Site
from clifford_3plus2_d5.spacetime_qca.state import State

LinkKey: TypeAlias = tuple[Site, Displacement]
LinkField: TypeAlias = dict[LinkKey, sp.Matrix]
GaugeTransform: TypeAlias = dict[Site, sp.Matrix]


def bcc_link_displacements() -> tuple[Displacement, ...]:
    return tuple(
        tuple(int(component) for component in direction)  # type: ignore[misc]
        for direction in bialynicki_birula_directions()
    )


def expected_link_keys(
    lattice: PeriodicLattice3D,
    displacements: tuple[Displacement, ...] | None = None,
) -> set[LinkKey]:
    directions = displacements or bcc_link_displacements()
    return {(site, direction) for site in lattice.sites() for direction in directions}


def validate_gauge_transform(
    gauge: GaugeTransform,
    lattice: PeriodicLattice3D,
) -> int:
    if set(gauge) != set(lattice.sites()):
        raise ValueError("gauge transform support must match lattice sites")
    dimensions = {matrix.shape for matrix in gauge.values()}
    if len(dimensions) != 1:
        raise ValueError("all gauge matrices must have the same shape")
    ((rows, cols),) = dimensions
    if rows != cols:
        raise ValueError("gauge matrices must be square")
    return rows


def validate_link_field(
    links: LinkField,
    lattice: PeriodicLattice3D,
    *,
    internal_dim: int | None = None,
    displacements: tuple[Displacement, ...] | None = None,
) -> int:
    if set(links) != expected_link_keys(lattice, displacements):
        raise ValueError("link field keys must match every lattice site and BCC displacement")
    dimensions = {matrix.shape for matrix in links.values()}
    if len(dimensions) != 1:
        raise ValueError("all link matrices must have the same shape")
    ((rows, cols),) = dimensions
    if rows != cols:
        raise ValueError("link matrices must be square")
    if internal_dim is not None and rows != internal_dim:
        raise ValueError(f"link matrices must be {internal_dim}x{internal_dim}")
    return rows


def identity_link_field(
    lattice: PeriodicLattice3D,
    internal_dim: int,
    *,
    displacements: tuple[Displacement, ...] | None = None,
) -> LinkField:
    return {key: sp.eye(internal_dim) for key in expected_link_keys(lattice, displacements)}


def constant_link_field(
    lattice: PeriodicLattice3D,
    link: sp.Matrix,
    *,
    displacements: tuple[Displacement, ...] | None = None,
) -> LinkField:
    if link.rows != link.cols:
        raise ValueError("link must be square")
    return {key: link for key in expected_link_keys(lattice, displacements)}


def transform_internal_state(
    state: State,
    lattice: PeriodicLattice3D,
    gauge: GaugeTransform,
    *,
    spacetime_dim: int = 4,
) -> State:
    internal_dim = validate_gauge_transform(gauge, lattice)
    expected_shape = (spacetime_dim * internal_dim, 1)
    if set(state) != set(lattice.sites()):
        raise ValueError("state support must match lattice sites")
    transformed: State = {}
    for site, spinor in state.items():
        if spinor.shape != expected_shape:
            raise ValueError(f"state spinors must be {expected_shape[0]}x1")
        operator = sp.kronecker_product(sp.eye(spacetime_dim), gauge[site])
        transformed[site] = (operator * spinor).applyfunc(sp.simplify)
    return transformed


def transform_link_field(
    links: LinkField,
    lattice: PeriodicLattice3D,
    gauge: GaugeTransform,
    *,
    displacements: tuple[Displacement, ...] | None = None,
) -> LinkField:
    internal_dim = validate_gauge_transform(gauge, lattice)
    validate_link_field(links, lattice, internal_dim=internal_dim, displacements=displacements)
    transformed: LinkField = {}
    for target, displacement in expected_link_keys(lattice, displacements):
        source = lattice.translate(target, displacement)
        transformed[(target, displacement)] = (
            gauge[target] * links[(target, displacement)] * gauge[source].inv()
        ).applyfunc(sp.simplify)
    return transformed


def pure_gauge_link_field(
    lattice: PeriodicLattice3D,
    gauge: GaugeTransform,
    *,
    displacements: tuple[Displacement, ...] | None = None,
) -> LinkField:
    identity = identity_link_field(lattice, validate_gauge_transform(gauge, lattice), displacements=displacements)
    return transform_link_field(identity, lattice, gauge, displacements=displacements)
