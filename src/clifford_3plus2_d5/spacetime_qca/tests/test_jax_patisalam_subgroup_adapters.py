"""Session 35 Pati-Salam and SM subgroup adapter tests."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from clifford_3plus2_d5.lepton.patisalam_sm import hypercharge_generator, sm_gauge_generators
from clifford_3plus2_d5.lepton.sm_hypercharge import physical_hypercharge_generator
from clifford_3plus2_d5.sim.state import sympy_matrix_to_numpy
from clifford_3plus2_d5.spacetime_qca import (
    canonical_bcc_plaquette_shapes,
    jax_average_wilson_action_density,
    jax_patisalam_action_descent_step,
    jax_patisalam_algebra_matrix,
    jax_patisalam_apply_momentum_update,
    jax_patisalam_gauge_hamiltonian_density,
    jax_patisalam_generators_chiral16,
    jax_patisalam_leapfrog_step,
    jax_patisalam_left_force,
    jax_patisalam_link_field_from_algebra,
    jax_patisalam_link_from_algebra,
    jax_patisalam_momentum_kinetic_energy_density,
    jax_patisalam_project_to_coordinates,
    jax_patisalam_pure_gauge_links_from_site_algebra,
    jax_patisalam_site_field_from_algebra,
    jax_patisalam_transform_momentum_field,
)
from clifford_3plus2_d5.spacetime_qca.jax_patisalam import PatiSalamGaugeSector

SECTOR_DIMS: dict[PatiSalamGaugeSector, int] = {
    "su4": 15,
    "su2_l": 3,
    "su2_r": 3,
    "pati_salam": 21,
    "su3_c": 8,
    "u1_y": 1,
    "u1_y_raw": 1,
    "sm": 12,
    "sm_raw": 12,
}


def _shapes():
    return (tuple(canonical_bcc_plaquette_shapes())[0],)


def _coordinates(dim: int) -> jnp.ndarray:
    return jnp.linspace(-0.04, 0.05, dim, dtype=jnp.float32)


def _theta(sector: PatiSalamGaugeSector, shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    dim = SECTOR_DIMS[sector]
    theta = jnp.zeros((*shape, 8, dim), dtype=jnp.float32)
    for hop, generator_index in enumerate(range(min(dim, 4))):
        theta = theta.at[0, 0, 0, hop, generator_index].set(0.035 - 0.006 * hop)
    return theta


def _site_theta(sector: PatiSalamGaugeSector, shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    dim = SECTOR_DIMS[sector]
    values = jnp.zeros((*shape, dim), dtype=jnp.float32)
    for generator_index in range(min(dim, 4)):
        values = values.at[..., generator_index].set(0.02 - 0.004 * generator_index)
    return values


def _momenta(sector: PatiSalamGaugeSector, shape: tuple[int, int, int] = (1, 1, 1)) -> jnp.ndarray:
    dim = SECTOR_DIMS[sector]
    _, _, _, hop, algebra = jnp.indices((*shape, 8, dim), dtype=jnp.float32)
    return 0.0015 * (hop + 1) - 0.0004 * algebra


@pytest.mark.parametrize("sector, dimension", SECTOR_DIMS.items())
def test_patisalam_sector_generators_are_valid_and_non_degenerate(
    sector: PatiSalamGaugeSector,
    dimension: int,
) -> None:
    generators = jax_patisalam_generators_chiral16(sector)

    assert generators.shape == (dimension, 32, 32)
    for matrix in np.asarray(generators):
        np.testing.assert_allclose(matrix.conj().T, -matrix, atol=1e-7)

    gram = jnp.real(jnp.einsum("aij,bji->ab", jnp.swapaxes(jnp.conj(generators), -1, -2), generators))
    assert np.linalg.matrix_rank(np.asarray(gram)) == dimension


@pytest.mark.parametrize("sector, dimension", SECTOR_DIMS.items())
def test_patisalam_sector_projection_recovers_coordinates(
    sector: PatiSalamGaugeSector,
    dimension: int,
) -> None:
    coordinates = _coordinates(dimension)
    algebra = jax_patisalam_algebra_matrix(coordinates, sector=sector)

    recovered = jax_patisalam_project_to_coordinates(algebra, sector=sector)

    np.testing.assert_allclose(np.asarray(recovered), np.asarray(coordinates), atol=4e-6)


@pytest.mark.parametrize("sector, dimension", SECTOR_DIMS.items())
def test_patisalam_sector_link_exponential_preserves_unitarity(
    sector: PatiSalamGaugeSector,
    dimension: int,
) -> None:
    theta = _coordinates(dimension)[None, :]

    links = jax_patisalam_link_from_algebra(theta, sector=sector)

    matrix = np.asarray(links[0])
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(32), atol=8e-6)


def test_sm_factor_links_commute_between_independent_subsectors() -> None:
    su3 = jax_patisalam_link_from_algebra(_coordinates(8)[None, :], sector="su3_c")[0]
    su2_l = jax_patisalam_link_from_algebra(_coordinates(3)[None, :], sector="su2_l")[0]
    u1_y = jax_patisalam_link_from_algebra(_coordinates(1)[None, :], sector="u1_y")[0]

    for left, right in itertools.combinations((su3, su2_l, u1_y), 2):
        np.testing.assert_allclose(
            np.asarray(left @ right),
            np.asarray(right @ left),
            atol=8e-6,
        )


def test_u1_y_sector_uses_physical_hypercharge_with_raw_alias() -> None:
    physical = sympy_matrix_to_numpy(physical_hypercharge_generator(), dtype=np.complex64)
    raw = sympy_matrix_to_numpy(hypercharge_generator(), dtype=np.complex64)

    np.testing.assert_allclose(
        np.asarray(jax_patisalam_generators_chiral16("u1_y")[0]),
        physical,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(jax_patisalam_generators_chiral16("u1_y_raw")[0]),
        raw,
        atol=1e-7,
    )
    assert not np.allclose(physical, raw)


def test_sm_sector_uses_physical_hypercharge_with_raw_alias() -> None:
    physical = sympy_matrix_to_numpy(physical_hypercharge_generator(), dtype=np.complex64)
    raw_sm = np.stack(
        [sympy_matrix_to_numpy(generator, dtype=np.complex64) for generator in sm_gauge_generators()],
        axis=0,
    )

    sm = np.asarray(jax_patisalam_generators_chiral16("sm"))
    sm_raw = np.asarray(jax_patisalam_generators_chiral16("sm_raw"))

    np.testing.assert_allclose(sm[-1], physical, atol=1e-7)
    np.testing.assert_allclose(sm_raw, raw_sm, atol=1e-7)
    np.testing.assert_allclose(sm[:-1], sm_raw[:-1], atol=1e-7)
    assert not np.allclose(sm[-1], sm_raw[-1])


@pytest.mark.parametrize("sector", ["su2_l", "su2_r", "su3_c", "u1_y", "sm"])
def test_patisalam_sector_pure_gauge_has_zero_action(sector: PatiSalamGaugeSector) -> None:
    links = jax_patisalam_pure_gauge_links_from_site_algebra(_site_theta(sector), sector=sector)

    action = jax_average_wilson_action_density(links, _shapes())

    np.testing.assert_allclose(np.asarray(action), np.asarray(0, dtype=np.float32), atol=4e-6)


@pytest.mark.slow
@pytest.mark.parametrize("sector", ["su2_l", "su3_c", "u1_y"])
def test_patisalam_sector_finite_difference_force_lowers_action(sector: PatiSalamGaugeSector) -> None:
    links = jax_patisalam_link_field_from_algebra(_theta(sector), sector=sector)
    action_before = jax_average_wilson_action_density(links, _shapes())

    updated, force = jax_patisalam_action_descent_step(
        links,
        sector=sector,
        step_size=0.12,
        epsilon=5e-3,
        shapes=_shapes(),
    )
    action_after = jax_average_wilson_action_density(updated, _shapes())

    assert force.shape == (1, 1, 1, 8, SECTOR_DIMS[sector])
    assert float(jnp.linalg.norm(force)) > 0
    assert float(action_after) < float(action_before)


def test_patisalam_u1_batched_finite_difference_matches_scalar() -> None:
    links = jax_patisalam_link_field_from_algebra(_theta("u1_y"), sector="u1_y")

    scalar = jax_patisalam_left_force(
        links,
        sector="u1_y",
        epsilon=5e-3,
        shapes=_shapes(),
        method="finite_difference",
    )
    batched = jax_patisalam_left_force(
        links,
        sector="u1_y",
        epsilon=5e-3,
        shapes=_shapes(),
        method="finite_difference_batched",
        chunk_size=3,
    )

    np.testing.assert_allclose(np.asarray(batched), np.asarray(scalar), atol=1e-5)


def test_patisalam_u1_analytic_staple_matches_scalar() -> None:
    links = jax_patisalam_link_field_from_algebra(_theta("u1_y"), sector="u1_y")

    scalar = jax_patisalam_left_force(
        links,
        sector="u1_y",
        epsilon=2e-3,
        shapes=_shapes(),
        method="finite_difference",
    )
    analytic = jax_patisalam_left_force(
        links,
        sector="u1_y",
        shapes=_shapes(),
        method="analytic_staple",
    )

    np.testing.assert_allclose(np.asarray(analytic), np.asarray(scalar), atol=2e-5)


@pytest.mark.parametrize("sector", ["su2_l", "su3_c", "u1_y", "sm"])
def test_patisalam_sector_momentum_transform_preserves_kinetic_energy(
    sector: PatiSalamGaugeSector,
) -> None:
    momenta = _momenta(sector)
    site_gauge = jax_patisalam_site_field_from_algebra(_site_theta(sector), sector=sector)

    transformed = jax_patisalam_transform_momentum_field(momenta, site_gauge, sector=sector)

    np.testing.assert_allclose(
        np.asarray(jax_patisalam_momentum_kinetic_energy_density(transformed, sector=sector)),
        np.asarray(jax_patisalam_momentum_kinetic_energy_density(momenta, sector=sector)),
        atol=4e-5,
    )


@pytest.mark.slow
@pytest.mark.parametrize("sector", ["su2_l", "su3_c", "u1_y"])
def test_patisalam_sector_identity_links_with_zero_momenta_are_fixed_by_leapfrog(
    sector: PatiSalamGaugeSector,
) -> None:
    links = jax_patisalam_link_field_from_algebra(
        jnp.zeros((1, 1, 1, 8, SECTOR_DIMS[sector]), dtype=jnp.float32),
        sector=sector,
    )
    momenta = jnp.zeros((1, 1, 1, 8, SECTOR_DIMS[sector]), dtype=jnp.float32)

    updated_links, updated_momenta = jax_patisalam_leapfrog_step(
        links,
        momenta,
        sector=sector,
        step_size=0.01,
        shapes=_shapes(),
        force_epsilon=5e-3,
    )

    np.testing.assert_allclose(np.asarray(updated_links), np.asarray(links), atol=1e-6)
    np.testing.assert_allclose(np.asarray(updated_momenta), np.asarray(momenta), atol=1e-6)


def test_patisalam_sector_momentum_update_preserves_compact_links() -> None:
    links = jax_patisalam_link_field_from_algebra(_theta("sm"), sector="sm")
    momenta = _momenta("sm")

    updated = jax_patisalam_apply_momentum_update(links, momenta, sector="sm", step_size=0.01)
    hamiltonian = jax_patisalam_gauge_hamiltonian_density(links, momenta, sector="sm", shapes=_shapes())

    assert bool(jnp.isfinite(hamiltonian))
    for matrix in np.asarray(updated).reshape(-1, 32, 32):
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(32), atol=8e-6)
