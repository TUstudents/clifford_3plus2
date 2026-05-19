"""Session 36 no-backreaction fermion/gauge coupling tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca import (
    PATISALAM_INTERNAL_DIM,
    canonical_bcc_plaquette_shapes,
    jax_identity_link_field,
    jax_patisalam_dirac_step,
    jax_patisalam_fermion_gauge_energy_density,
    jax_patisalam_fermion_gauge_step,
    jax_patisalam_leapfrog_step,
    jax_patisalam_site_field_from_algebra,
    jax_patisalam_transform_momentum_field,
    jax_transform_link_field,
    jax_transform_patisalam_dirac_state,
)
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import PatiSalamGaugeSector

SECTOR_DIMS: dict[PatiSalamGaugeSector, int] = {
    "su4": 15,
    "su2_l": 3,
    "su2_r": 3,
    "pati_salam": 21,
    "su3_c": 8,
    "u1_y": 1,
    "sm": 12,
}

pytestmark = pytest.mark.slow


def _shapes():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _state(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    state = jnp.zeros((*lattice_shape, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[0, 0, 0, 1, 7].set(-0.5 + 0.125j)
    state = state.at[1, 0, 0, 2, 11].set(0.375 - 0.25j)
    state = state.at[1, 0, 0, 3, 19].set(-0.125 + 0.5j)
    return state


def _identity_links(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    return jax_identity_link_field(lattice_shape, PATISALAM_INTERNAL_DIM, dtype=jnp.complex64)


def _momenta(
    sector: PatiSalamGaugeSector = "u1_y",
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
) -> jnp.ndarray:
    dim = SECTOR_DIMS[sector]
    _, _, _, hop, algebra = jnp.indices((*lattice_shape, 8, dim), dtype=jnp.float32)
    return 0.004 * (hop + 1) - 0.001 * algebra


def _site_theta(
    sector: PatiSalamGaugeSector = "u1_y",
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
) -> jnp.ndarray:
    theta = jnp.zeros((*lattice_shape, SECTOR_DIMS[sector]), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0].set(0.03)
    theta = theta.at[1, 0, 0, 0].set(-0.04)
    return theta


def test_patisalam_fermion_gauge_step_reduces_to_dirac_step_for_static_identity_gauge() -> None:
    state = _state()
    links = _identity_links()
    momenta = jnp.zeros((2, 1, 1, 8, SECTOR_DIMS["u1_y"]), dtype=jnp.float32)

    stepped_state, stepped_links, stepped_momenta = jax_patisalam_fermion_gauge_step(
        state,
        links,
        momenta,
        sector="u1_y",
        step_size=0.01,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    np.testing.assert_allclose(np.asarray(stepped_state), np.asarray(jax_patisalam_dirac_step(state, links)), atol=2e-6)
    np.testing.assert_allclose(np.asarray(stepped_links), np.asarray(links), atol=2e-6)
    np.testing.assert_allclose(np.asarray(stepped_momenta), np.asarray(momenta), atol=2e-6)


def test_zero_fermion_state_does_not_backreact_on_gauge_leapfrog() -> None:
    state = jnp.zeros((2, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    links = _identity_links()
    momenta = _momenta("u1_y")

    stepped_state, stepped_links, stepped_momenta = jax_patisalam_fermion_gauge_step(
        state,
        links,
        momenta,
        sector="u1_y",
        step_size=0.025,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )
    expected_links, expected_momenta = jax_patisalam_leapfrog_step(
        links,
        momenta,
        sector="u1_y",
        step_size=0.025,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    np.testing.assert_allclose(np.asarray(stepped_state), np.asarray(state), atol=1e-7)
    np.testing.assert_allclose(np.asarray(stepped_links), np.asarray(expected_links), atol=2e-6)
    np.testing.assert_allclose(np.asarray(stepped_momenta), np.asarray(expected_momenta), atol=2e-6)


def test_patisalam_dirac_step_is_covariant_under_site_local_internal_gauge() -> None:
    state = _state()
    links = _identity_links()
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta("u1_y"), sector="u1_y")
    transformed_state = jax_transform_patisalam_dirac_state(state, site_gauge)
    transformed_links = jax_transform_link_field(links, site_gauge)

    stepped_transformed = jax_patisalam_dirac_step(transformed_state, transformed_links)
    transformed_stepped = jax_transform_patisalam_dirac_state(
        jax_patisalam_dirac_step(state, links),
        site_gauge,
    )

    np.testing.assert_allclose(np.asarray(stepped_transformed), np.asarray(transformed_stepped), atol=2e-6)


def test_zero_timestep_coupled_step_is_covariant_under_site_local_gauge() -> None:
    state = _state()
    links = _identity_links()
    momenta = _momenta("u1_y")
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta("u1_y"), sector="u1_y")

    stepped_state, stepped_links, stepped_momenta = jax_patisalam_fermion_gauge_step(
        state,
        links,
        momenta,
        sector="u1_y",
        step_size=0.0,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )
    transformed_state, transformed_links, transformed_momenta = jax_patisalam_fermion_gauge_step(
        jax_transform_patisalam_dirac_state(state, site_gauge),
        jax_transform_link_field(links, site_gauge),
        jax_patisalam_transform_momentum_field(momenta, site_gauge, sector="u1_y"),
        sector="u1_y",
        step_size=0.0,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    np.testing.assert_allclose(
        np.asarray(transformed_state),
        np.asarray(jax_transform_patisalam_dirac_state(stepped_state, site_gauge)),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(transformed_links),
        np.asarray(jax_transform_link_field(stepped_links, site_gauge)),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(transformed_momenta),
        np.asarray(jax_patisalam_transform_momentum_field(stepped_momenta, site_gauge, sector="u1_y")),
        atol=2e-6,
    )


def test_fermion_gauge_diagnostics_report_norm_and_gauge_hamiltonian() -> None:
    diagnostics = jax_patisalam_fermion_gauge_energy_density(
        _state(),
        _identity_links(),
        _momenta("u1_y"),
        sector="u1_y",
        shapes=_shapes(),
    )

    assert set(diagnostics) == {"fermion_norm", "gauge_hamiltonian_density"}
    assert float(diagnostics["fermion_norm"]) > 0
    assert bool(jnp.isfinite(diagnostics["gauge_hamiltonian_density"]))
