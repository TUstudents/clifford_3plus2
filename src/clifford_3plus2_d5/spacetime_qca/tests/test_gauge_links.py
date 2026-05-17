"""Session 24 finite-lattice background gauge-link covariance tests."""

from __future__ import annotations

import pytest
import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    GaugeTransform,
    LinkField,
    PeriodicLattice3D,
    bcc_link_displacements,
    constant_link_field,
    delta_state,
    dirac_step_with_constant_link,
    dirac_step_with_link_field,
    identity_link_field,
    plane_wave_state,
    pure_gauge_link_field,
    same_matrix,
    state_norm_squared,
    states_close_exact,
    transform_internal_state,
    transform_link_field,
)


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


def test_identity_link_field_matches_ungauged_constant_link_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    spinor = sp.Matrix([1, 0, 0, 1, 0, 1, 1, 0])
    state = delta_state(lattice, (0, 0, 0), spinor)

    identity_links = identity_link_field(lattice, internal_dim=2)
    linked = dirac_step_with_link_field(state, lattice, identity_links)
    constant = dirac_step_with_constant_link(state, lattice, sp.eye(2))

    assert states_close_exact(linked, constant)


def test_constant_link_field_matches_constant_link_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    link = sp.Matrix([[0, 1], [1, 0]])
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 1, 0, 1, 1, 0, 1, 0]))

    local_links = constant_link_field(lattice, link)
    linked = dirac_step_with_link_field(state, lattice, local_links)
    constant = dirac_step_with_constant_link(state, lattice, link)

    assert states_close_exact(linked, constant)


def test_site_local_gauge_covariance_for_position_dependent_links() -> None:
    lattice = PeriodicLattice3D((2, 2, 2))
    state = {
        site: sp.Matrix([1 + site[0], site[1], site[2], 1, site[2], 1 + site[1], site[0], 1])
        for site in lattice.sites()
    }
    links = _varied_links(lattice)
    gauge = _site_gauge(lattice)

    transformed_state = transform_internal_state(state, lattice, gauge, spacetime_dim=4)
    transformed_links = transform_link_field(links, lattice, gauge)

    left = dirac_step_with_link_field(transformed_state, lattice, transformed_links)
    right = transform_internal_state(
        dirac_step_with_link_field(state, lattice, links),
        lattice,
        gauge,
        spacetime_dim=4,
    )

    assert states_close_exact(left, right)


def test_pure_gauge_links_preserve_plane_wave_norm() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    momentum = (sp.pi / 2, 0, 0)
    base_state = plane_wave_state(lattice, momentum, sp.Matrix([1, 0, 0, 1, 0, 1, 1, 0]))
    gauge = _site_gauge(lattice)

    gauged_state = transform_internal_state(base_state, lattice, gauge, spacetime_dim=4)
    pure_links = pure_gauge_link_field(lattice, gauge)
    stepped = dirac_step_with_link_field(gauged_state, lattice, pure_links)

    assert sp.simplify(state_norm_squared(stepped) - state_norm_squared(gauged_state)) == 0


def test_link_field_validation_rejects_missing_link() -> None:
    lattice = PeriodicLattice3D((2, 2, 2))
    links = identity_link_field(lattice, internal_dim=2)
    links.pop(next(iter(links)))
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 0, 0, 1, 0, 1, 1, 0]))

    with pytest.raises(ValueError, match="link field keys"):
        dirac_step_with_link_field(state, lattice, links)


def test_transform_link_field_uses_pull_convention() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    links = identity_link_field(lattice, internal_dim=2)
    gauge = _site_gauge(lattice)
    transformed = transform_link_field(links, lattice, gauge)
    target = (0, 0, 0)
    displacement = (1, 1, 1)
    source = lattice.translate(target, displacement)

    expected = (gauge[target] * gauge[source].inv()).applyfunc(sp.simplify)
    assert same_matrix(transformed[(target, displacement)], expected)
