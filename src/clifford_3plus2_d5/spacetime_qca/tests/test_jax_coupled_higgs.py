"""Session 40 coupled fermion/gauge/Higgs prototype tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.spacetime_qca import (
    PATISALAM_INTERNAL_DIM,
    canonical_bcc_plaquette_shapes,
    jax_higgs_link_field_from_algebra,
    jax_higgs_site_gauge_from_algebra,
    jax_identity_link_field,
    jax_patisalam_fermion_gauge_step_with_backreaction,
    jax_patisalam_link_field_from_algebra,
    jax_transform_higgs_field,
    jax_transform_higgs_links,
)
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    jax_apply_site_local_yukawa_kick,
    jax_higgs_coordinates_from_patisalam_sector,
    jax_higgs_force,
    jax_higgs_leapfrog_step,
    jax_higgs_link_field_from_patisalam_sector,
    jax_higgs_momentum_energy_density,
    jax_higgs_site_gauge_from_patisalam_sector,
    jax_higgs_total_energy,
    jax_patisalam_fermion_gauge_higgs_diagnostics,
    jax_patisalam_fermion_gauge_higgs_step,
    jax_transform_higgs_momentum,
)


SECTOR_DIMS = {
    "u1_y": 1,
    "su2_l": 3,
    "sm": 12,
}


def _shapes():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _state(lattice_shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    state = jnp.zeros((*lattice_shape, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[0, 0, 0, 1, 7].set(-0.5 + 0.125j)
    return state


def _patisalam_links(lattice_shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    return jax_identity_link_field(lattice_shape, PATISALAM_INTERNAL_DIM, dtype=jnp.complex64)


def _momenta(sector: str = "u1_y", lattice_shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    dim = SECTOR_DIMS[sector]
    _, _, _, hop, algebra = jnp.indices((*lattice_shape, 8, dim), dtype=jnp.float32)
    return 0.001 * (hop + 1) - 0.0001 * algebra


def _phi(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    x = jnp.arange(np.prod(lattice_shape), dtype=jnp.float32).reshape(lattice_shape)
    return jnp.stack((0.1 * x + 0.05j, 1.0 + 0.03 * x - 0.02j * x), axis=-1)


def _higgs_links(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    values = jnp.arange(np.prod(lattice_shape) * 8 * 4, dtype=jnp.float32).reshape((*lattice_shape, 8, 4))
    return jax_higgs_link_field_from_algebra(0.002 * values)


def _higgs_site_theta(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    values = jnp.arange(np.prod(lattice_shape) * 4, dtype=jnp.float32).reshape((*lattice_shape, 4))
    return 0.02 * values


def test_higgs_momentum_transform_preserves_norm_and_energy() -> None:
    momentum = 0.25 * _phi()
    site_gauge = jax_higgs_site_gauge_from_algebra(_higgs_site_theta())

    transformed = jax_transform_higgs_momentum(momentum, site_gauge)

    np.testing.assert_allclose(
        np.asarray(jnp.vdot(transformed, transformed)),
        np.asarray(jnp.vdot(momentum, momentum)),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax_higgs_momentum_energy_density(transformed)),
        np.asarray(jax_higgs_momentum_energy_density(momentum)),
        atol=2e-5,
    )


def test_higgs_force_vanishes_at_neutral_vev_with_identity_links() -> None:
    phi = jnp.zeros((2, 1, 1, 2), dtype=jnp.complex64).at[..., 1].set(1.0 + 0.0j)
    links = jax_higgs_link_field_from_algebra(jnp.zeros((2, 1, 1, 8, 4), dtype=jnp.float32))

    force = jax_higgs_force(phi, links, vev_squared=1.0, quartic=1.0)

    np.testing.assert_allclose(np.asarray(force), np.zeros((2, 1, 1, 2), dtype=np.complex64), atol=2e-5)


def test_higgs_force_is_gauge_covariant() -> None:
    phi = _phi()
    links = _higgs_links()
    site_gauge = jax_higgs_site_gauge_from_algebra(_higgs_site_theta())

    force = jax_higgs_force(phi, links, vev_squared=0.8, quartic=1.2)
    transformed_force = jax_higgs_force(
        jax_transform_higgs_field(phi, site_gauge),
        jax_transform_higgs_links(links, site_gauge),
        vev_squared=0.8,
        quartic=1.2,
    )
    expected = jax_transform_higgs_field(force, site_gauge)

    np.testing.assert_allclose(np.asarray(transformed_force), np.asarray(expected), atol=2e-4)


def test_higgs_leapfrog_zero_step_and_neutral_fixed_point() -> None:
    phi = _phi()
    momentum = 0.2 * _phi()
    links = _higgs_links()
    zero_phi, zero_momentum = jax_higgs_leapfrog_step(phi, momentum, links, step_size=0.0)
    np.testing.assert_allclose(np.asarray(zero_phi), np.asarray(phi), atol=1e-7)
    np.testing.assert_allclose(np.asarray(zero_momentum), np.asarray(momentum), atol=1e-7)

    neutral = jnp.zeros((2, 1, 1, 2), dtype=jnp.complex64).at[..., 1].set(1.0 + 0.0j)
    zero = jnp.zeros_like(neutral)
    identity_links = jax_higgs_link_field_from_algebra(jnp.zeros((2, 1, 1, 8, 4), dtype=jnp.float32))
    updated_phi, updated_momentum = jax_higgs_leapfrog_step(neutral, zero, identity_links, step_size=0.05)
    np.testing.assert_allclose(np.asarray(updated_phi), np.asarray(neutral), atol=2e-5)
    np.testing.assert_allclose(np.asarray(updated_momentum), np.asarray(zero), atol=2e-5)


def test_higgs_total_energy_is_finite_and_includes_momentum() -> None:
    phi = _phi()
    momentum = 0.25 * _phi()
    links = _higgs_links()

    energy = jax_higgs_total_energy(phi, momentum, links, vev_squared=0.7, quartic=1.1)

    assert bool(jnp.isfinite(energy))
    assert float(energy) > 0


def test_higgs_total_energy_zero_for_neutral_vev_control() -> None:
    phi = jnp.zeros((2, 1, 1, 2), dtype=jnp.complex64).at[..., 1].set(1.0 + 0.0j)
    momentum = jnp.zeros_like(phi)
    links = jax_higgs_link_field_from_algebra(jnp.zeros((2, 1, 1, 8, 4), dtype=jnp.float32))

    energy = jax_higgs_total_energy(phi, momentum, links, vev_squared=1.0, quartic=1.0)

    np.testing.assert_allclose(np.asarray(energy), 0.0, atol=2e-5)


def test_sector_to_higgs_coordinate_adapters() -> None:
    u1 = jnp.asarray([[[[0.25]]]], dtype=jnp.float32)
    su2 = jnp.asarray([[[[0.1, 0.2, 0.3]]]], dtype=jnp.float32)
    sm = jnp.zeros((1, 1, 1, 12), dtype=jnp.float32).at[..., 8:12].set(jnp.asarray((1.0, 2.0, 3.0, 4.0)))

    np.testing.assert_allclose(
        np.asarray(jax_higgs_coordinates_from_patisalam_sector(u1, sector="u1_y")),
        np.asarray([[[[0.0, 0.0, 0.0, 0.25]]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(jax_higgs_coordinates_from_patisalam_sector(su2, sector="su2_l")),
        np.asarray([[[[0.1, 0.2, 0.3, 0.0]]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(jax_higgs_coordinates_from_patisalam_sector(sm, sector="sm")),
        np.asarray([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32),
    )

    assert jax_higgs_site_gauge_from_patisalam_sector(u1, sector="u1_y").shape == (1, 1, 1, 2, 2)
    assert jax_higgs_link_field_from_patisalam_sector(
        jnp.zeros((1, 1, 1, 8, 12), dtype=jnp.float32),
        sector="sm",
    ).shape == (1, 1, 1, 8, 2, 2)

    with pytest.raises(ValueError, match="does not act"):
        jax_higgs_coordinates_from_patisalam_sector(jnp.zeros((1, 1, 1, 8), dtype=jnp.float32), sector="su3_c")


def test_coupled_diagnostics_report_expected_keys_and_finite_values() -> None:
    lattice_shape = (1, 1, 1)
    diagnostics = jax_patisalam_fermion_gauge_higgs_diagnostics(
        _state(lattice_shape),
        _patisalam_links(lattice_shape),
        _momenta("u1_y", lattice_shape),
        jnp.zeros((*lattice_shape, 2), dtype=jnp.complex64).at[..., 1].set(1.0 + 0.0j),
        jnp.zeros((*lattice_shape, 2), dtype=jnp.complex64),
        jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32)),
        sector="u1_y",
        shapes=_shapes(),
    )

    assert set(diagnostics) == {
        "fermion_norm",
        "gauge_hamiltonian_density",
        "higgs_norm",
        "higgs_momentum_energy_density",
        "higgs_kinetic_energy_density",
        "higgs_potential_density",
        "higgs_energy_density",
        "gauss_residual_norm",
        "yukawa_norm_drift",
    }
    for value in diagnostics.values():
        assert bool(jnp.all(jnp.isfinite(value)))


def test_coupled_wrapper_rejects_color_only_sector() -> None:
    lattice_shape = (1, 1, 1)
    state = _state(lattice_shape)
    links = jax_identity_link_field(lattice_shape, PATISALAM_INTERNAL_DIM, dtype=jnp.complex64)
    momenta = jnp.zeros((*lattice_shape, 8, 8), dtype=jnp.float32)
    phi = jnp.zeros((*lattice_shape, 2), dtype=jnp.complex64)
    higgs_links = jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32))

    with pytest.raises(ValueError, match="not supported"):
        jax_patisalam_fermion_gauge_higgs_step(
            state,
            links,
            momenta,
            phi,
            phi,
            higgs_links,
            sector="su3_c",  # type: ignore[arg-type]
            step_size=0.0,
        )


@pytest.mark.slow
def test_site_local_yukawa_kick_zero_phi_and_zero_step_are_noops() -> None:
    state = _state()
    zero_phi = jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64)
    neutral_phi = zero_phi.at[..., 1].set(1.0 + 0.0j)

    np.testing.assert_allclose(
        np.asarray(jax_apply_site_local_yukawa_kick(state, zero_phi, step_size=0.1)),
        np.asarray(state),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jax_apply_site_local_yukawa_kick(state, neutral_phi, step_size=0.0)),
        np.asarray(state),
        atol=1e-7,
    )


@pytest.mark.slow
def test_site_local_yukawa_kick_matches_local_hamiltonian_action() -> None:
    state = _state()
    phi = jnp.zeros((1, 1, 1, 2), dtype=jnp.complex64).at[..., 1].set(0.75 + 0.0j)

    kicked = jax_apply_site_local_yukawa_kick(state, phi, step_size=0.05)

    assert kicked.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(kicked)))
    assert float(jnp.abs(jnp.vdot(kicked - state, kicked - state))) > 0


@pytest.mark.slow
def test_coupled_step_reduces_to_session37_when_higgs_and_yukawa_are_disabled() -> None:
    lattice_shape = (1, 1, 1)
    state = _state(lattice_shape)
    links = _patisalam_links(lattice_shape)
    momenta = _momenta("u1_y", lattice_shape)
    phi = jnp.zeros((*lattice_shape, 2), dtype=jnp.complex64)
    higgs_momentum = jnp.zeros_like(phi)
    higgs_links = jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32))
    kwargs = {
        "sector": "u1_y",
        "step_size": 0.01,
        "matter_coupling": 0.0,
        "shapes": _shapes(),
        "force_epsilon": 5e-3,
    }

    expected_state, expected_links, expected_momenta = jax_patisalam_fermion_gauge_step_with_backreaction(
        state,
        links,
        momenta,
        **kwargs,
    )
    actual = jax_patisalam_fermion_gauge_higgs_step(
        state,
        links,
        momenta,
        phi,
        higgs_momentum,
        higgs_links,
        yukawa_coupling=0.0,
        **kwargs,
    )

    np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(expected_state), atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual[1]), np.asarray(expected_links), atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual[2]), np.asarray(expected_momenta), atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual[3]), np.asarray(phi), atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual[4]), np.asarray(higgs_momentum), atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual[5]), np.asarray(higgs_links), atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("sector", ("u1_y", "su2_l", "sm"))
def test_coupled_step_smoke_shapes_for_supported_sectors(sector: str) -> None:
    lattice_shape = (1, 1, 1)
    state = _state(lattice_shape)
    link_theta = jnp.zeros((*lattice_shape, 8, SECTOR_DIMS[sector]), dtype=jnp.float32)
    links = jax_patisalam_link_field_from_algebra(link_theta, sector=sector)
    momenta = jnp.zeros((*lattice_shape, 8, SECTOR_DIMS[sector]), dtype=jnp.float32)
    phi = jnp.zeros((*lattice_shape, 2), dtype=jnp.complex64).at[..., 1].set(1.0 + 0.0j)
    higgs_momentum = jnp.zeros_like(phi)
    higgs_links = jax_higgs_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, 4), dtype=jnp.float32))

    step_size = 0.0 if sector == "sm" else 0.005
    actual = jax_patisalam_fermion_gauge_higgs_step(
        state,
        links,
        momenta,
        phi,
        higgs_momentum,
        higgs_links,
        sector=sector,
        step_size=step_size,
        matter_coupling=0.0,
        yukawa_coupling=0.0,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    assert actual[0].shape == state.shape
    assert actual[1].shape == links.shape
    assert actual[2].shape == momenta.shape
    assert actual[3].shape == phi.shape
    assert actual[4].shape == higgs_momentum.shape
    assert actual[5].shape == higgs_links.shape
    for item in actual:
        assert bool(jnp.all(jnp.isfinite(item)))
