"""Session 26 JAX numerical backend tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from clifford_3plus2_d5.spacetime_qca import (
    PeriodicLattice3D,
    bcc_link_displacements,
    delta_state,
    dirac_step,
    dirac_step_with_constant_link,
    dirac_step_with_link_field,
    jax_dirac_step,
    jax_dirac_step_with_constant_link,
    jax_dirac_step_with_links,
    jax_identity_link_field,
    sympy_link_field_to_jax,
    sympy_state_to_jax_dirac_internal,
    sympy_state_to_jax_flat,
    zero_jax_dirac_internal_state,
)
from clifford_3plus2_d5.spacetime_qca.links import LinkField


def test_jax_runtime_available() -> None:
    devices = jax.devices()
    assert devices
    assert jax.default_backend() in {"cpu", "gpu", "tpu"}


def test_jax_dirac_delta_step_matches_exact_sympy_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 2, sp.I, -1]))
    expected = sympy_state_to_jax_flat(dirac_step(state, lattice), lattice)

    stepped = jax_dirac_step(sympy_state_to_jax_flat(state, lattice))

    np.testing.assert_allclose(np.asarray(stepped), np.asarray(expected), atol=1e-6)


def test_jax_constant_identity_link_matches_exact_tensor_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    internal_dim = 2
    state = delta_state(lattice, (0, 0, 0), sp.Matrix([1, 0, 0, 1, 0, 1, 1, 0]))
    expected = sympy_state_to_jax_dirac_internal(
        dirac_step_with_constant_link(state, lattice, sp.eye(internal_dim)),
        lattice,
        internal_dim=internal_dim,
    )

    stepped = jax_dirac_step_with_links(
        sympy_state_to_jax_dirac_internal(state, lattice, internal_dim=internal_dim),
        jax_identity_link_field(lattice.shape, internal_dim),
    )

    np.testing.assert_allclose(np.asarray(stepped), np.asarray(expected), atol=1e-6)


def test_jax_position_dependent_links_match_exact_sympy_step() -> None:
    lattice = PeriodicLattice3D((3, 3, 3))
    internal_dim = 2
    state = delta_state(lattice, (1, 0, 2), sp.Matrix([1, sp.I, 0, 1, -1, 0, 2, 1]))
    links: LinkField = {}
    for site in lattice.sites():
        for displacement in bcc_link_displacements():
            parity = sum(site) + displacement[0] - displacement[1] + displacement[2]
            sign = sp.Integer(-1) if parity % 2 else sp.Integer(1)
            links[(site, displacement)] = sp.diag(sign, sp.Integer(1))

    expected = sympy_state_to_jax_dirac_internal(
        dirac_step_with_link_field(state, lattice, links),
        lattice,
        internal_dim=internal_dim,
    )
    stepped = jax_dirac_step_with_links(
        sympy_state_to_jax_dirac_internal(state, lattice, internal_dim=internal_dim),
        sympy_link_field_to_jax(links, lattice),
    )

    np.testing.assert_allclose(np.asarray(stepped), np.asarray(expected), atol=1e-6)


def test_jax_constant_link_shortcut_matches_explicit_link_field() -> None:
    lattice_shape = (3, 3, 3)
    state = zero_jax_dirac_internal_state(lattice_shape, 2)
    state = state.at[0, 0, 0, 0, 0].set(1 + 0j)
    link = jnp.diag(jnp.asarray([1.0 + 0j, -1.0 + 0j], dtype=jnp.complex64))

    shortcut = jax_dirac_step_with_constant_link(state, link)
    explicit = jax_dirac_step_with_links(
        state,
        jnp.broadcast_to(link, (*lattice_shape, 8, 2, 2)),
    )

    np.testing.assert_allclose(np.asarray(shortcut), np.asarray(explicit), atol=1e-6)


def test_jax_dirac_step_is_jittable_and_runs_on_default_device() -> None:
    state = zero_jax_dirac_internal_state((4, 4, 4), 2)
    state = state.at[0, 0, 0, 0, 0].set(1 + 0j)
    links = jax_identity_link_field((4, 4, 4), 2)

    stepped = jax.jit(jax_dirac_step_with_links)(state, links).block_until_ready()

    assert stepped.shape == state.shape
    assert next(iter(stepped.devices())).platform == jax.default_backend()
