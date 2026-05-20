"""Session 41 JAX charge-normalization diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    PATISALAM_INTERNAL_DIM,
    jax_patisalam_fermion_charge_density,
    jax_patisalam_generators_chiral16,
)


def test_u1_charge_density_uses_physical_hypercharge_generator() -> None:
    state = jnp.zeros((1, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    amplitude = jnp.asarray(0.75 + 0.5j, dtype=jnp.complex64)
    state = state.at[0, 0, 0, 1, 0].set(amplitude)
    generator = jax_patisalam_generators_chiral16("u1_y")[0]

    charge = jax_patisalam_fermion_charge_density(state, sector="u1_y", coordinate_mode="raw")
    expected = jnp.real(jnp.conj(amplitude) * (1j * generator[0, 0]) * amplitude)

    np.testing.assert_allclose(np.asarray(charge[0, 0, 0, 0]), np.asarray(expected), atol=2e-6)


def test_raw_u1_alias_remains_distinct_for_charge_density_regressions() -> None:
    state = jnp.zeros((1, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.0j)
    state = state.at[0, 0, 0, 0, 2].set(0.0 + 1.0j)

    physical = jax_patisalam_fermion_charge_density(state, sector="u1_y", coordinate_mode="raw")
    raw = jax_patisalam_fermion_charge_density(state, sector="u1_y_raw", coordinate_mode="raw")

    assert not np.allclose(np.asarray(physical), np.asarray(raw))
