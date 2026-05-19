"""Session 37 Gauss-law and backreaction prototype tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.spacetime_qca import (
    PATISALAM_INTERNAL_DIM,
    canonical_bcc_plaquette_shapes,
    jax_identity_link_field,
    jax_patisalam_algebra_matrix,
    jax_patisalam_apply_fermion_backreaction,
    jax_patisalam_electric_divergence,
    jax_patisalam_fermion_charge_density,
    jax_patisalam_fermion_gauge_step,
    jax_patisalam_fermion_gauge_step_with_backreaction,
    jax_patisalam_fermion_link_current,
    jax_patisalam_gauss_residual,
    jax_patisalam_generators_chiral16,
    jax_patisalam_momentum_algebra,
    jax_patisalam_project_to_coordinates,
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
    x, _, _, hop, algebra = jnp.indices((*lattice_shape, 8, dim), dtype=jnp.float32)
    return 0.003 * (x + 1) + 0.001 * (hop + 1) - 0.0002 * algebra


def _site_theta(
    sector: PatiSalamGaugeSector = "u1_y",
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
) -> jnp.ndarray:
    theta = jnp.zeros((*lattice_shape, SECTOR_DIMS[sector]), dtype=jnp.float32)
    theta = theta.at[0, 0, 0, 0].set(0.03)
    theta = theta.at[1, 0, 0, 0].set(-0.04)
    if SECTOR_DIMS[sector] > 1:
        theta = theta.at[0, 0, 0, 1].set(0.015)
        theta = theta.at[1, 0, 0, 1].set(-0.01)
    return theta


def _transform_site_coordinates(
    coordinates: jnp.ndarray,
    site_gauge: jnp.ndarray,
    *,
    sector: PatiSalamGaugeSector,
) -> jnp.ndarray:
    matrix = jax_patisalam_algebra_matrix(coordinates, sector=sector)
    transformed = site_gauge @ matrix @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2)
    return jax_patisalam_project_to_coordinates(transformed, sector=sector)


def test_momentum_algebra_has_shape_and_is_antihermitian() -> None:
    algebra = jax_patisalam_momentum_algebra(_momenta("su2_l"), sector="su2_l")

    assert algebra.shape == (2, 1, 1, 8, PATISALAM_INTERNAL_DIM, PATISALAM_INTERNAL_DIM)
    for matrix in np.asarray(algebra).reshape(-1, PATISALAM_INTERNAL_DIM, PATISALAM_INTERNAL_DIM):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=2e-7)


def test_zero_momenta_give_zero_electric_divergence() -> None:
    links = _identity_links()
    momenta = jnp.zeros((2, 1, 1, 8, SECTOR_DIMS["su2_l"]), dtype=jnp.float32)

    divergence = jax_patisalam_electric_divergence(links, momenta, sector="su2_l")

    assert divergence.shape == (2, 1, 1, SECTOR_DIMS["su2_l"])
    np.testing.assert_allclose(np.asarray(divergence), 0, atol=1e-7)


def test_electric_divergence_is_covariant_under_site_gauge_transform() -> None:
    sector: PatiSalamGaugeSector = "su2_l"
    links = _identity_links()
    momenta = _momenta(sector)
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector), sector=sector)

    transformed_divergence = jax_patisalam_electric_divergence(
        jax_transform_link_field(links, site_gauge),
        jax_patisalam_transform_momentum_field(momenta, site_gauge, sector=sector),
        sector=sector,
    )
    expected = _transform_site_coordinates(
        jax_patisalam_electric_divergence(links, momenta, sector=sector),
        site_gauge,
        sector=sector,
    )

    np.testing.assert_allclose(np.asarray(transformed_divergence), np.asarray(expected), atol=2e-5)


def test_zero_fermion_state_gives_zero_charge_density() -> None:
    state = jnp.zeros((2, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)

    charge = jax_patisalam_fermion_charge_density(state, sector="su3_c")

    assert charge.shape == (2, 1, 1, SECTOR_DIMS["su3_c"])
    np.testing.assert_allclose(np.asarray(charge), 0, atol=1e-7)


def test_raw_u1_charge_density_matches_charge_observable_expectation() -> None:
    state = jnp.zeros((1, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    amplitude = jnp.asarray(1.25 - 0.5j, dtype=jnp.complex64)
    state = state.at[0, 0, 0, 2, 0].set(amplitude)
    generator = jax_patisalam_generators_chiral16("u1_y")[0]
    expected = jnp.real(jnp.conj(amplitude) * (1j * generator[0, 0]) * amplitude)

    raw = jax_patisalam_fermion_charge_density(state, sector="u1_y", coordinate_mode="raw")

    np.testing.assert_allclose(np.asarray(raw[0, 0, 0, 0]), np.asarray(expected), atol=2e-6)


def test_charge_density_is_covariant_under_site_gauge_transform() -> None:
    sector: PatiSalamGaugeSector = "su2_l"
    state = _state()
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector), sector=sector)

    transformed_charge = jax_patisalam_fermion_charge_density(
        jax_transform_patisalam_dirac_state(state, site_gauge),
        sector=sector,
    )
    expected = _transform_site_coordinates(
        jax_patisalam_fermion_charge_density(state, sector=sector),
        site_gauge,
        sector=sector,
    )

    np.testing.assert_allclose(np.asarray(transformed_charge), np.asarray(expected), atol=3e-5)


def test_gauss_residual_reduces_to_divergence_for_zero_fermions() -> None:
    state = jnp.zeros((2, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)
    links = _identity_links()
    momenta = _momenta("u1_y")

    residual = jax_patisalam_gauss_residual(state, links, momenta, sector="u1_y", matter_coupling=1.7)

    np.testing.assert_allclose(
        np.asarray(residual),
        np.asarray(jax_patisalam_electric_divergence(links, momenta, sector="u1_y")),
        atol=2e-6,
    )


def test_gauss_residual_reduces_to_minus_charge_for_zero_momenta() -> None:
    state = _state()
    links = _identity_links()
    momenta = jnp.zeros((2, 1, 1, 8, SECTOR_DIMS["u1_y"]), dtype=jnp.float32)
    coupling = 0.4

    residual = jax_patisalam_gauss_residual(state, links, momenta, sector="u1_y", matter_coupling=coupling)

    np.testing.assert_allclose(
        np.asarray(residual),
        np.asarray(-coupling * jax_patisalam_fermion_charge_density(state, sector="u1_y")),
        atol=2e-6,
    )


def test_gauss_residual_is_covariant_under_site_gauge_transform() -> None:
    sector: PatiSalamGaugeSector = "u1_y"
    state = _state()
    links = _identity_links()
    momenta = _momenta(sector)
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector), sector=sector)

    transformed_residual = jax_patisalam_gauss_residual(
        jax_transform_patisalam_dirac_state(state, site_gauge),
        jax_transform_link_field(links, site_gauge),
        jax_patisalam_transform_momentum_field(momenta, site_gauge, sector=sector),
        sector=sector,
    )
    expected = _transform_site_coordinates(
        jax_patisalam_gauss_residual(state, links, momenta, sector=sector),
        site_gauge,
        sector=sector,
    )

    np.testing.assert_allclose(np.asarray(transformed_residual), np.asarray(expected), atol=3e-5)


def test_zero_fermion_state_gives_zero_link_current() -> None:
    state = jnp.zeros((2, 1, 1, 4, PATISALAM_INTERNAL_DIM), dtype=jnp.complex64)

    current = jax_patisalam_fermion_link_current(state, _identity_links(), sector="u1_y", epsilon=2e-3)

    assert current.shape == (2, 1, 1, 8, SECTOR_DIMS["u1_y"])
    np.testing.assert_allclose(np.asarray(current), 0, atol=2e-6)


def test_link_current_is_finite_and_u1_gauge_invariant() -> None:
    sector: PatiSalamGaugeSector = "u1_y"
    state = _state()
    links = _identity_links()
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector), sector=sector)

    current = jax_patisalam_fermion_link_current(state, links, sector=sector, epsilon=2e-3)
    transformed_current = jax_patisalam_fermion_link_current(
        jax_transform_patisalam_dirac_state(state, site_gauge),
        jax_transform_link_field(links, site_gauge),
        sector=sector,
        epsilon=2e-3,
    )

    assert bool(jnp.all(jnp.isfinite(current)))
    np.testing.assert_allclose(np.asarray(transformed_current), np.asarray(current), atol=4e-4)


def test_link_current_is_covariant_under_nonabelian_site_gauge_transform() -> None:
    sector: PatiSalamGaugeSector = "su2_l"
    state = _state((1, 1, 1))
    links = _identity_links((1, 1, 1))
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector, (1, 1, 1)), sector=sector)

    current = jax_patisalam_fermion_link_current(state, links, sector=sector, epsilon=2e-3)
    transformed_current = jax_patisalam_fermion_link_current(
        jax_transform_patisalam_dirac_state(state, site_gauge),
        jax_transform_link_field(links, site_gauge),
        sector=sector,
        epsilon=2e-3,
    )
    expected = jax_patisalam_transform_momentum_field(current, site_gauge, sector=sector)

    np.testing.assert_allclose(np.asarray(transformed_current), np.asarray(expected), atol=7e-4)


def test_backreaction_kick_is_linear_and_zero_current_is_noop() -> None:
    momenta = _momenta("su2_l")
    zero_current = jnp.zeros_like(momenta)
    current = jnp.ones_like(momenta) * 0.25

    np.testing.assert_allclose(
        np.asarray(jax_patisalam_apply_fermion_backreaction(momenta, zero_current, step_size=0.2)),
        np.asarray(momenta),
        atol=1e-7,
    )
    kicked = jax_patisalam_apply_fermion_backreaction(momenta, current, step_size=0.2, matter_coupling=0.5)
    np.testing.assert_allclose(np.asarray(kicked - momenta), np.asarray(0.025 * jnp.ones_like(momenta)), atol=1e-7)


def test_zero_coupling_backreaction_step_matches_session36_wrapper() -> None:
    state = _state()
    links = _identity_links()
    momenta = _momenta("u1_y")
    kwargs = {
        "sector": "u1_y",
        "step_size": 0.01,
        "shapes": _shapes(),
        "force_epsilon": 5e-3,
    }

    expected = jax_patisalam_fermion_gauge_step(state, links, momenta, **kwargs)
    actual = jax_patisalam_fermion_gauge_step_with_backreaction(
        state,
        links,
        momenta,
        matter_coupling=0.0,
        current_epsilon=2e-3,
        **kwargs,
    )

    for actual_item, expected_item in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(actual_item), np.asarray(expected_item), atol=2e-6)
