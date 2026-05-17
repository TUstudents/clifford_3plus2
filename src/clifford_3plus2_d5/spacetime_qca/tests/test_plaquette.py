"""Session 24b BCC plaquette and holonomy tests."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    GaugeTransform,
    LinkField,
    PeriodicLattice3D,
    canonical_bcc_plaquette_shapes,
    identity_link_field,
    is_elementary_bcc_plaquette_shape,
    plaquette_holonomy,
    plaquette_holonomy_covariance_residual,
    plaquette_path_sites,
    plaquette_vertices_unwrapped,
    pure_gauge_link_field,
    same_matrix,
)
from clifford_3plus2_d5.spacetime_qca.links import bcc_link_displacements


def _site_gauge(lattice: PeriodicLattice3D) -> GaugeTransform:
    swap = sp.Matrix([[0, 1], [1, 0]])
    sign = sp.diag(1, -1)
    return {
        site: sign if sum(site) % 2 == 0 else swap
        for site in lattice.sites()
    }


def _varied_links(lattice: PeriodicLattice3D) -> LinkField:
    identity = sp.eye(2)
    swap = sp.Matrix([[0, 1], [1, 0]])
    sign = sp.diag(1, -1)
    links: LinkField = {}
    for site in lattice.sites():
        for index, direction in enumerate(bcc_link_displacements()):
            links[(site, direction)] = (identity if index % 3 == 0 else swap if sum(site) % 2 else sign)
    return links


def test_canonical_bcc_plaquette_shapes_are_six_nonbacktracking_loops() -> None:
    shapes = canonical_bcc_plaquette_shapes()
    assert len(shapes) == 6
    for shape in shapes:
        assert is_elementary_bcc_plaquette_shape(shape)
        vertices = plaquette_vertices_unwrapped(shape)
        assert vertices[0] == (0, 0, 0)
        assert vertices[-1] == (0, 0, 0)
        assert len(set(vertices[:-1])) == 4


def test_plaquette_path_sites_close_on_periodic_lattice() -> None:
    lattice = PeriodicLattice3D((5, 5, 5))
    shape = canonical_bcc_plaquette_shapes()[0]
    sites = plaquette_path_sites(lattice, (2, 1, 3), shape)
    assert len(sites) == 5
    assert sites[0] == sites[-1]


def test_identity_link_plaquette_holonomy_is_identity() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    links = identity_link_field(lattice, internal_dim=2)
    for shape in canonical_bcc_plaquette_shapes():
        assert same_matrix(plaquette_holonomy(links, lattice, (0, 0, 0), shape), sp.eye(2))


def test_pure_gauge_plaquette_holonomy_is_identity() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = pure_gauge_link_field(lattice, _site_gauge(lattice))
    shape = canonical_bcc_plaquette_shapes()[0]
    assert same_matrix(plaquette_holonomy(links, lattice, (1, 2, 3), shape), sp.eye(2))


def test_plaquette_holonomy_transforms_by_base_site_conjugation() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    gauge = _site_gauge(lattice)
    shape = canonical_bcc_plaquette_shapes()[1]
    residual = plaquette_holonomy_covariance_residual(links, lattice, gauge, (1, 0, 2), shape)
    assert same_matrix(residual, sp.zeros(2))


def test_degenerate_shape_is_not_elementary_plaquette() -> None:
    degenerate = ((1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1))
    assert not is_elementary_bcc_plaquette_shape(degenerate)
