"""Shared simulator infrastructure tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5 import sim
from clifford_3plus2_d5.spacetime_qca import (
    jax_constant_link_field as spacetime_constant_link_field,
    jax_identity_link_field as spacetime_identity_link_field,
)


def test_identity_and_constant_link_fields_have_expected_layout() -> None:
    identity = sim.jax_identity_link_field((2, 3, 4), 2, edge_count=6)
    link = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64)
    constant = sim.jax_constant_link_field((2, 3, 4), link, edge_count=6)

    assert identity.shape == (2, 3, 4, 6, 2, 2)
    assert constant.shape == (2, 3, 4, 6, 2, 2)
    np.testing.assert_allclose(np.asarray(identity[1, 2, 3, 5]), np.eye(2))
    np.testing.assert_allclose(np.asarray(constant[1, 2, 3, 5]), np.asarray(link))


def test_spacetime_public_link_helpers_match_shared_sim_defaults() -> None:
    link = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)

    np.testing.assert_allclose(
        np.asarray(spacetime_identity_link_field((2, 2, 2), 2)),
        np.asarray(sim.jax_identity_link_field((2, 2, 2), 2)),
    )
    np.testing.assert_allclose(
        np.asarray(spacetime_constant_link_field((2, 2, 2), link)),
        np.asarray(sim.jax_constant_link_field((2, 2, 2), link)),
    )


def test_source_roll_matches_manual_jnp_roll() -> None:
    state = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    displacement = (1, -1, 1)

    rolled = sim.source_roll(state, displacement)
    expected = jnp.roll(state, shift=(-1, 1, -1), axis=(0, 1, 2))

    np.testing.assert_array_equal(np.asarray(rolled), np.asarray(expected))


def test_generic_gauge_transform_uses_pull_link_convention() -> None:
    links = sim.jax_identity_link_field((2, 2, 2), 2, edge_count=2)
    site_gauge = jnp.zeros((2, 2, 2, 2, 2), dtype=jnp.complex64)
    site_gauge = site_gauge.at[..., 0, 0].set(1)
    site_gauge = site_gauge.at[..., 1, 1].set(1)
    site_gauge = site_gauge.at[0, 0, 0].set(jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64))
    displacements = ((1, 0, 0), (-1, 0, 0))

    transformed = sim.jax_transform_link_field(links, site_gauge, displacements)
    source = sim.source_roll(site_gauge, displacements[0])
    expected = site_gauge @ jnp.swapaxes(jnp.conj(source), -1, -2)

    np.testing.assert_allclose(np.asarray(transformed[..., 0, :, :]), np.asarray(expected))


def test_dirac_internal_flatten_roundtrip_and_norm_metrics() -> None:
    state = jnp.arange(2 * 2 * 2 * 4 * 3, dtype=jnp.float32).reshape((2, 2, 2, 4, 3))

    flat = sim.flatten_dirac_internal_state(state)
    roundtrip = sim.unflatten_dirac_internal_state(flat, internal_dim=3)
    metrics = sim.state_transition_metrics(state, roundtrip)

    assert flat.shape == (2, 2, 2, 12)
    np.testing.assert_array_equal(np.asarray(roundtrip), np.asarray(state))
    assert metrics.all_finite
    assert metrics.norm_drift == 0


def test_benchmark_helper_returns_nonnegative_timing() -> None:
    def add_one(value: jnp.ndarray) -> jnp.ndarray:
        return value + 1

    timing = sim.benchmark_jitted_kernel(add_one, jnp.ones((4,), dtype=jnp.float32))

    assert timing.compile_seconds >= 0
    assert timing.run_seconds >= 0
