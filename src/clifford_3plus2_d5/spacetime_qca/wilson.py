"""Wilson plaquette observables for BCC background link fields."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.lattice import PeriodicLattice3D, Site
from clifford_3plus2_d5.spacetime_qca.links import LinkField, validate_link_field
from clifford_3plus2_d5.spacetime_qca.plaquette import (
    PlaquetteShape,
    canonical_bcc_plaquette_shapes,
    plaquette_holonomy,
)


def wilson_loop_trace(
    links: LinkField,
    lattice: PeriodicLattice3D,
    base_site: Site,
    shape: PlaquetteShape,
) -> sp.Expr:
    """Return ``Tr(H[x0, shape])`` for one BCC plaquette."""

    return sp.simplify(sp.trace(plaquette_holonomy(links, lattice, base_site, shape)))


def normalized_wilson_loop(
    links: LinkField,
    lattice: PeriodicLattice3D,
    base_site: Site,
    shape: PlaquetteShape,
) -> sp.Expr:
    """Return ``Tr(H) / internal_dim`` for one BCC plaquette."""

    internal_dim = validate_link_field(links, lattice)
    return sp.simplify(wilson_loop_trace(links, lattice, base_site, shape) / internal_dim)


def average_normalized_wilson_loop(
    links: LinkField,
    lattice: PeriodicLattice3D,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> sp.Expr:
    """Return the uniform site/shape average of the normalized Wilson loop."""

    selected_shapes = shapes or canonical_bcc_plaquette_shapes()
    if not selected_shapes:
        raise ValueError("at least one plaquette shape is required")
    validate_link_field(links, lattice)

    total = sp.Integer(0)
    count = 0
    for site in lattice.sites():
        for shape in selected_shapes:
            total += normalized_wilson_loop(links, lattice, site, shape)
            count += 1
    return sp.simplify(total / count)


def wilson_plaquette_energy(
    links: LinkField,
    lattice: PeriodicLattice3D,
    base_site: Site,
    shape: PlaquetteShape,
) -> sp.Expr:
    """Return ``1 - Re(Tr(H) / internal_dim)`` for one BCC plaquette."""

    return sp.simplify(1 - sp.re(normalized_wilson_loop(links, lattice, base_site, shape)))


def average_wilson_action_density(
    links: LinkField,
    lattice: PeriodicLattice3D,
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> sp.Expr:
    """Return the uniform site/shape average Wilson plaquette energy."""

    selected_shapes = shapes or canonical_bcc_plaquette_shapes()
    if not selected_shapes:
        raise ValueError("at least one plaquette shape is required")
    validate_link_field(links, lattice)

    total = sp.Integer(0)
    count = 0
    for site in lattice.sites():
        for shape in selected_shapes:
            total += wilson_plaquette_energy(links, lattice, site, shape)
            count += 1
    return sp.simplify(total / count)


def total_wilson_action(
    links: LinkField,
    lattice: PeriodicLattice3D,
    *,
    beta: sp.Expr = sp.Integer(1),
    shapes: tuple[PlaquetteShape, ...] | None = None,
) -> sp.Expr:
    """Return ``beta * sum_p (1 - Re(Tr(H_p) / internal_dim))``."""

    selected_shapes = shapes or canonical_bcc_plaquette_shapes()
    if not selected_shapes:
        raise ValueError("at least one plaquette shape is required")
    density = average_wilson_action_density(links, lattice, selected_shapes)
    return sp.simplify(beta * density * lattice.volume * len(selected_shapes))
