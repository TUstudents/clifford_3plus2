"""Session 27 BCC Wilson observable tests."""

from __future__ import annotations

import numpy as np
import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    GaugeTransform,
    LinkField,
    PeriodicLattice3D,
    average_normalized_wilson_loop,
    average_wilson_action_density,
    canonical_bcc_plaquette_shapes,
    identity_link_field,
    jax_average_normalized_wilson_loop,
    jax_average_wilson_action_density,
    jax_normalized_wilson_loop,
    jax_plaquette_holonomy,
    jax_total_wilson_action,
    jax_wilson_plaquette_energy,
    jax_wilson_loop_trace,
    normalized_wilson_loop,
    plaquette_holonomy,
    pure_gauge_link_field,
    same_matrix,
    sympy_link_field_to_jax,
    total_wilson_action,
    wilson_plaquette_energy,
    wilson_loop_trace,
)
from clifford_3plus2_d5.spacetime_qca.links import bcc_link_displacements, transform_link_field


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
            links[(site, direction)] = identity if index % 3 == 0 else swap if sum(site) % 2 else sign
    return links


def test_exact_identity_wilson_loop_is_dimension_normalized_to_one() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    links = identity_link_field(lattice, internal_dim=2)
    shape = canonical_bcc_plaquette_shapes()[0]

    assert wilson_loop_trace(links, lattice, (0, 0, 0), shape) == 2
    assert normalized_wilson_loop(links, lattice, (0, 0, 0), shape) == 1
    assert average_normalized_wilson_loop(links, lattice) == 1


def test_exact_pure_gauge_average_wilson_loop_is_one() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = pure_gauge_link_field(lattice, _site_gauge(lattice))

    assert average_normalized_wilson_loop(links, lattice) == 1


def test_exact_wilson_trace_is_gauge_invariant() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    transformed = transform_link_field(links, lattice, _site_gauge(lattice))
    shape = canonical_bcc_plaquette_shapes()[1]

    assert (
        wilson_loop_trace(links, lattice, (1, 0, 2), shape)
        == wilson_loop_trace(transformed, lattice, (1, 0, 2), shape)
    )


def test_jax_plaquette_holonomy_matches_exact_sympy_holonomy() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    shape = canonical_bcc_plaquette_shapes()[2]
    base_site = (1, 2, 0)

    expected = plaquette_holonomy(links, lattice, base_site, shape)
    actual = jax_plaquette_holonomy(sympy_link_field_to_jax(links, lattice), base_site, shape)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected.tolist(), dtype=np.complex64))


def test_jax_normalized_wilson_loop_matches_exact_sympy_value() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    shape = canonical_bcc_plaquette_shapes()[3]
    base_site = (2, 1, 3)

    expected = complex(sp.N(normalized_wilson_loop(links, lattice, base_site, shape)))
    actual = jax_normalized_wilson_loop(sympy_link_field_to_jax(links, lattice), base_site, shape)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected, dtype=np.complex64))


def test_jax_identity_and_pure_gauge_average_wilson_loop_are_one() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    identity = identity_link_field(lattice, internal_dim=2)
    pure_gauge = pure_gauge_link_field(lattice, _site_gauge(lattice))

    np.testing.assert_allclose(
        np.asarray(jax_average_normalized_wilson_loop(sympy_link_field_to_jax(identity, lattice))),
        np.asarray(1 + 0j, dtype=np.complex64),
    )
    np.testing.assert_allclose(
        np.asarray(jax_average_normalized_wilson_loop(sympy_link_field_to_jax(pure_gauge, lattice))),
        np.asarray(1 + 0j, dtype=np.complex64),
    )


def test_jax_wilson_trace_is_gauge_invariant_against_transformed_links() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    transformed = transform_link_field(links, lattice, _site_gauge(lattice))
    shape = canonical_bcc_plaquette_shapes()[4]
    base_site = (3, 1, 2)

    original_trace = jax_wilson_loop_trace(sympy_link_field_to_jax(links, lattice), base_site, shape)
    transformed_trace = jax_wilson_loop_trace(
        sympy_link_field_to_jax(transformed, lattice),
        base_site,
        shape,
    )

    np.testing.assert_allclose(np.asarray(original_trace), np.asarray(transformed_trace))
    assert same_matrix(
        plaquette_holonomy(transformed, lattice, base_site, shape),
        _site_gauge(lattice)[base_site]
        * plaquette_holonomy(links, lattice, base_site, shape)
        * _site_gauge(lattice)[base_site].inv(),
    )


def test_exact_identity_and_pure_gauge_wilson_action_density_are_zero() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    identity = identity_link_field(lattice, internal_dim=2)
    pure_gauge = pure_gauge_link_field(lattice, _site_gauge(lattice))
    shape = canonical_bcc_plaquette_shapes()[0]

    assert wilson_plaquette_energy(identity, lattice, (0, 0, 0), shape) == 0
    assert average_wilson_action_density(identity, lattice) == 0
    assert total_wilson_action(identity, lattice) == 0
    assert average_wilson_action_density(pure_gauge, lattice) == 0


def test_exact_wilson_action_is_gauge_invariant_and_nonnegative_for_nontrivial_links() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    transformed = transform_link_field(links, lattice, _site_gauge(lattice))

    density = average_wilson_action_density(links, lattice)
    assert density >= 0
    assert density != 0
    assert density == average_wilson_action_density(transformed, lattice)
    assert total_wilson_action(links, lattice, beta=sp.Integer(3)) == (
        3 * density * lattice.volume * len(canonical_bcc_plaquette_shapes())
    )


def test_jax_wilson_action_density_matches_exact_sympy_value() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    shape = canonical_bcc_plaquette_shapes()[2]
    base_site = (1, 2, 0)
    jax_links = sympy_link_field_to_jax(links, lattice)

    expected_energy = float(sp.N(wilson_plaquette_energy(links, lattice, base_site, shape)))
    actual_energy = jax_wilson_plaquette_energy(jax_links, base_site, shape)
    np.testing.assert_allclose(np.asarray(actual_energy), np.asarray(expected_energy, dtype=np.float32))

    expected_density = float(sp.N(average_wilson_action_density(links, lattice)))
    actual_density = jax_average_wilson_action_density(jax_links)
    np.testing.assert_allclose(np.asarray(actual_density), np.asarray(expected_density, dtype=np.float32))


def test_jax_identity_and_pure_gauge_wilson_action_density_are_zero() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    identity = identity_link_field(lattice, internal_dim=2)
    pure_gauge = pure_gauge_link_field(lattice, _site_gauge(lattice))

    np.testing.assert_allclose(
        np.asarray(jax_average_wilson_action_density(sympy_link_field_to_jax(identity, lattice))),
        np.asarray(0, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(jax_average_wilson_action_density(sympy_link_field_to_jax(pure_gauge, lattice))),
        np.asarray(0, dtype=np.float32),
    )


def test_jax_total_wilson_action_matches_density_formula_and_is_gauge_invariant() -> None:
    lattice = PeriodicLattice3D((4, 4, 4))
    links = _varied_links(lattice)
    transformed = transform_link_field(links, lattice, _site_gauge(lattice))
    jax_links = sympy_link_field_to_jax(links, lattice)
    jax_transformed = sympy_link_field_to_jax(transformed, lattice)
    beta = 2.5

    density = jax_average_wilson_action_density(jax_links)
    expected_total = beta * density * lattice.volume * len(canonical_bcc_plaquette_shapes())
    total = jax_total_wilson_action(jax_links, beta=beta)
    transformed_total = jax_total_wilson_action(jax_transformed, beta=beta)

    np.testing.assert_allclose(np.asarray(total), np.asarray(expected_total))
    np.testing.assert_allclose(np.asarray(total), np.asarray(transformed_total))
