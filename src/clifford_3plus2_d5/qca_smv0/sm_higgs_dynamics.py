"""Dynamic Higgs-field evolution for QCA_SMv0.

Stage 8 adds a classical Higgs field/momentum update around the already
implemented Higgs/Yukawa collision.  The Higgs evolves as an electroweak
doublet with gauge-covariant BCC gradient energy and a local quartic potential.
Fermion backreaction, boundary rules, and quantized scalar registers remain out
of scope for this stage.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS
from clifford_3plus2_d5.qca_smv0.sm_higgs import SM_HIGGS_DIM, sm_constant_higgs
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.links import jax_identity_link_field, jax_transform_link_field, validate_link_field

SM_HIGGS_GENERATOR_COUNT = 4


class HiggsDynamicsParameters(NamedTuple):
    """Parameters for the Stage 8 Higgs field update."""

    vev: float = 1.0
    quartic: float = 0.25
    gradient_coupling: float = 1.0


class HiggsDynamicsDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 8 dynamic Higgs fields."""

    generator_antihermitian_residual: jnp.ndarray
    link_unitarity_residual: jnp.ndarray
    vacuum_force_norm: jnp.ndarray
    pure_gauge_gradient_energy: jnp.ndarray
    force_covariance_residual: jnp.ndarray
    energy_gauge_invariance_residual: jnp.ndarray
    hamiltonian_drift: jnp.ndarray
    reversibility_residual: jnp.ndarray
    jit_delta_higgs: jnp.ndarray
    jit_delta_momenta: jnp.ndarray


DEFAULT_HIGGS_DYNAMICS_PARAMETERS = HiggsDynamicsParameters()


def sm_higgs_generators() -> jnp.ndarray:
    """Return anti-Hermitian `SU(2)_L x U(1)_Y` generators on the Higgs doublet."""

    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex64)
    identity = jnp.eye(SM_HIGGS_DIM, dtype=jnp.complex64)
    return jnp.stack((0.5j * sigma_x, 0.5j * sigma_y, 0.5j * sigma_z, 0.5j * identity), axis=0)


def sm_higgs_generator_antihermitian_residual() -> jnp.ndarray:
    """Return `max_abs(T+T^dag)` for Higgs electroweak generators."""

    generators = sm_higgs_generators()
    return jnp.max(jnp.abs(generators + jnp.swapaxes(jnp.conj(generators), -1, -2)))


def _validate_higgs_theta(theta: jnp.ndarray, *, with_edges: bool) -> tuple[int, int, int]:
    expected_tail = (8, SM_HIGGS_GENERATOR_COUNT) if with_edges else (SM_HIGGS_GENERATOR_COUNT,)
    if theta.ndim != (5 if with_edges else 4) or theta.shape[-len(expected_tail) :] != expected_tail:
        shape = "(nx, ny, nz, 8, 4)" if with_edges else "(nx, ny, nz, 4)"
        raise ValueError(f"Higgs electroweak algebra coordinates must have shape {shape}")
    return int(theta.shape[0]), int(theta.shape[1]), int(theta.shape[2])


def _validate_higgs_field(field: jnp.ndarray) -> tuple[int, int, int]:
    if field.ndim != 4 or field.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    return int(field.shape[0]), int(field.shape[1]), int(field.shape[2])


def _validate_higgs_momenta(momenta: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    if momenta.ndim != 4 or momenta.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs momenta must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and momenta.shape[:3] != lattice_shape:
        raise ValueError("Higgs momenta must match the Higgs lattice")


def _validate_higgs_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    validate_link_field(links, edge_count=8)
    if links.ndim != 6 or links.shape[3:] != (8, SM_HIGGS_DIM, SM_HIGGS_DIM):
        raise ValueError("Higgs links must have shape (nx, ny, nz, 8, 2, 2)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("Higgs links must match the Higgs lattice")


def sm_higgs_algebra_matrix_field(theta: jnp.ndarray) -> jnp.ndarray:
    """Return Higgs electroweak algebra matrices from site or link coordinates."""

    if theta.ndim == 4:
        _validate_higgs_theta(theta, with_edges=False)
    elif theta.ndim == 5:
        _validate_higgs_theta(theta, with_edges=True)
    else:
        raise ValueError("Higgs electroweak coordinates must be site or link fields")
    generators = sm_higgs_generators().astype(jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def _matrix_exponential_field(algebra: jnp.ndarray) -> jnp.ndarray:
    flat = algebra.reshape((-1, algebra.shape[-2], algebra.shape[-1]))
    links = jax.vmap(jsp_linalg.expm)(flat)
    return links.reshape(algebra.shape)


def sm_higgs_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return finite Higgs electroweak BCC links from `(nx,ny,nz,8,4)` coordinates."""

    _validate_higgs_theta(theta, with_edges=True)
    return _matrix_exponential_field(sm_higgs_algebra_matrix_field(theta))


def sm_higgs_site_gauge_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return finite Higgs electroweak site gauges from `(nx,ny,nz,4)` coordinates."""

    _validate_higgs_theta(site_theta, with_edges=False)
    return _matrix_exponential_field(sm_higgs_algebra_matrix_field(site_theta))


def sm_identity_higgs_links(lattice_shape: tuple[int, int, int]) -> jnp.ndarray:
    """Return identity Higgs electroweak links."""

    return jax_identity_link_field(lattice_shape, SM_HIGGS_DIM, dtype=jnp.complex64)


def sm_transform_higgs_field(field: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply site-local electroweak gauge transforms to a Higgs field."""

    lattice_shape = _validate_higgs_field(field)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != lattice_shape or site_gauge.shape[-2:] != (
        SM_HIGGS_DIM,
        SM_HIGGS_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 2, 2)")
    return jnp.einsum("...ab,...b->...a", site_gauge, field)


def sm_transform_higgs_links(links: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply pull-convention site-local Higgs electroweak transforms to links."""

    _validate_higgs_links(links)
    return jax_transform_link_field(links, site_gauge, BCC_DISPLACEMENTS)


def sm_higgs_link_unitarity_residual(links: jnp.ndarray) -> jnp.ndarray:
    """Return `max_abs(U^dag U-I)` over a Higgs electroweak link field."""

    _validate_higgs_links(links)
    identity = jnp.eye(SM_HIGGS_DIM, dtype=links.dtype)
    gram = jnp.einsum("...ji,...jk->...ik", jnp.conj(links), links)
    return jnp.max(jnp.abs(gram - identity))


def deterministic_higgs_theta(lattice_shape: tuple[int, int, int], *, scale: float = 0.3) -> jnp.ndarray:
    """Return deterministic Higgs link algebra coordinates for tests."""

    nx, ny, nz = lattice_shape
    coords = jnp.indices((nx, ny, nz, 8, SM_HIGGS_GENERATOR_COUNT), dtype=jnp.float32)
    pattern = (
        0.17 * (coords[0] + 1.0)
        + 0.11 * (coords[1] + 1.0)
        + 0.07 * (coords[2] + 1.0)
        + 0.13 * (coords[3] + 1.0)
        + 0.19 * (coords[4] + 1.0)
    )
    return scale * jnp.sin(pattern)


def deterministic_higgs_site_theta(lattice_shape: tuple[int, int, int], *, scale: float = 0.25) -> jnp.ndarray:
    """Return deterministic site-gauge coordinates for Higgs tests."""

    nx, ny, nz = lattice_shape
    coords = jnp.indices((nx, ny, nz, SM_HIGGS_GENERATOR_COUNT), dtype=jnp.float32)
    pattern = 0.23 * (coords[0] + 1.0) + 0.17 * (coords[1] + 1.0) + 0.29 * (coords[2] + 1.0) + 0.31 * (
        coords[3] + 1.0
    )
    return scale * jnp.cos(pattern)


def deterministic_higgs_field(lattice_shape: tuple[int, int, int]) -> jnp.ndarray:
    """Return a deterministic non-vacuum Higgs doublet field."""

    vacuum = sm_constant_higgs(lattice_shape)
    nx, ny, nz = lattice_shape
    coords = jnp.indices((nx, ny, nz), dtype=jnp.float32)
    perturb_0 = 0.04 * jnp.sin(0.37 * (coords[0] + 1.0)) + 0.03j * jnp.cos(0.19 * (coords[1] + 1.0))
    perturb_1 = 0.05 * jnp.cos(0.23 * (coords[0] + coords[1] + 1.0)) + 0.02j * jnp.sin(0.41 * (coords[2] + 1.0))
    perturb = jnp.stack((perturb_0, perturb_1), axis=-1).astype(jnp.complex64)
    return vacuum + perturb


def deterministic_higgs_momenta(lattice_shape: tuple[int, int, int]) -> jnp.ndarray:
    """Return deterministic Higgs momenta."""

    nx, ny, nz = lattice_shape
    coords = jnp.indices((nx, ny, nz), dtype=jnp.float32)
    pi_0 = 0.03 * jnp.cos(0.13 * (coords[0] + 2.0)) - 0.02j * jnp.sin(0.31 * (coords[1] + 1.0))
    pi_1 = -0.025 * jnp.sin(0.17 * (coords[0] + coords[2] + 1.0)) + 0.015j * jnp.cos(0.27 * (coords[2] + 1.0))
    return jnp.stack((pi_0, pi_1), axis=-1).astype(jnp.complex64)


def sm_higgs_potential_density(
    field: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> jnp.ndarray:
    """Return mean quartic Higgs potential density."""

    _validate_higgs_field(field)
    norm_sq = jnp.real(jnp.sum(jnp.conj(field) * field, axis=-1))
    target = jnp.asarray(parameters.vev * parameters.vev / 2.0, dtype=norm_sq.dtype)
    return jnp.mean(jnp.asarray(parameters.quartic, dtype=norm_sq.dtype) * (norm_sq - target) ** 2)


def sm_higgs_kinetic_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return mean scalar momentum kinetic density."""

    _validate_higgs_momenta(momenta)
    return 0.5 * jnp.mean(jnp.real(jnp.sum(jnp.conj(momenta) * momenta, axis=-1)))


def sm_higgs_covariant_differences(field: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return `U_h(x) H(x+h)-H(x)` for BCC pull links."""

    lattice_shape = _validate_higgs_field(field)
    _validate_higgs_links(links, lattice_shape)
    differences = []
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(field, displacement)
        transported = jnp.einsum("...ab,...b->...a", links[..., index, :, :], source)
        differences.append(transported - field)
    return jnp.stack(differences, axis=3)


def sm_higgs_gradient_density(
    field: jnp.ndarray,
    links: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> jnp.ndarray:
    """Return mean gauge-covariant BCC gradient density."""

    differences = sm_higgs_covariant_differences(field, links)
    local = 0.5 * jnp.real(jnp.sum(jnp.conj(differences) * differences, axis=(-2, -1)))
    return jnp.asarray(parameters.gradient_coupling, dtype=local.dtype) * jnp.mean(local)


def sm_higgs_hamiltonian_density(
    field: jnp.ndarray,
    momenta: jnp.ndarray,
    links: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> jnp.ndarray:
    """Return mean Higgs Hamiltonian density."""

    return (
        sm_higgs_kinetic_density(momenta)
        + sm_higgs_gradient_density(field, links, parameters=parameters)
        + sm_higgs_potential_density(field, parameters=parameters)
    )


def sm_higgs_force(
    field: jnp.ndarray,
    links: jnp.ndarray,
    *,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> jnp.ndarray:
    """Return the force `-dH/dphi*` from gradient and potential energy."""

    lattice_shape = _validate_higgs_field(field)
    _validate_higgs_links(links, lattice_shape)
    site_count = jnp.asarray(field.shape[0] * field.shape[1] * field.shape[2], dtype=field.real.dtype)
    norm_sq = jnp.real(jnp.sum(jnp.conj(field) * field, axis=-1, keepdims=True))
    target = jnp.asarray(parameters.vev * parameters.vev / 2.0, dtype=norm_sq.dtype)
    potential_force = (
        -2.0
        * jnp.asarray(parameters.quartic, dtype=norm_sq.dtype)
        * (norm_sq - target)
        * field
        / site_count
    )

    target_sum = jnp.zeros_like(field)
    source_sum = jnp.zeros_like(field)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(field, displacement)
        link = links[..., index, :, :]
        transported = jnp.einsum("...ab,...b->...a", link, source)
        target_sum = target_sum + (field - transported)

        link_at_target = source_roll(link, tuple(-component for component in displacement))
        target_field = source_roll(field, tuple(-component for component in displacement))
        source_diff = jnp.einsum("...ab,...b->...a", link_at_target, field) - target_field
        source_sum = source_sum + jnp.einsum("...ba,...b->...a", jnp.conj(link_at_target), source_diff)

    grad_derivative = jnp.asarray(0.5 * parameters.gradient_coupling, dtype=field.real.dtype) * (
        target_sum + source_sum
    ) / site_count
    return potential_force - grad_derivative


def sm_higgs_leapfrog_step(
    field: jnp.ndarray,
    momenta: jnp.ndarray,
    links: jnp.ndarray,
    *,
    step_size: float,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance Higgs field and momenta with a reversible leapfrog step."""

    lattice_shape = _validate_higgs_field(field)
    _validate_higgs_momenta(momenta, lattice_shape)
    _validate_higgs_links(links, lattice_shape)
    half_momenta = momenta + 0.5 * step_size * sm_higgs_force(field, links, parameters=parameters)
    updated_field = field + step_size * half_momenta
    updated_momenta = half_momenta + 0.5 * step_size * sm_higgs_force(updated_field, links, parameters=parameters)
    return updated_field, updated_momenta


def sm_pure_gauge_higgs_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge Higgs links generated from site-local gauges."""

    lattice_shape = _validate_higgs_theta(site_theta, with_edges=False)
    site_gauge = sm_higgs_site_gauge_from_algebra(site_theta)
    return sm_transform_higgs_links(sm_identity_higgs_links(lattice_shape), site_gauge)


def sm_higgs_dynamics_diagnostics() -> HiggsDynamicsDiagnostics:
    """Return focused Stage 8 dynamic Higgs-field diagnostics."""

    lattice_shape = (2, 1, 1)
    parameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS
    field = deterministic_higgs_field(lattice_shape)
    momenta = deterministic_higgs_momenta(lattice_shape)
    links = sm_higgs_link_field_from_algebra(deterministic_higgs_theta(lattice_shape, scale=0.08))
    site_gauge = sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape, scale=0.17))
    transformed_field = sm_transform_higgs_field(field, site_gauge)
    transformed_links = sm_transform_higgs_links(links, site_gauge)
    transformed_force = sm_higgs_force(transformed_field, transformed_links, parameters=parameters)
    expected_force = sm_transform_higgs_field(sm_higgs_force(field, links, parameters=parameters), site_gauge)
    transformed_energy = sm_higgs_hamiltonian_density(
        transformed_field,
        sm_transform_higgs_field(momenta, site_gauge),
        transformed_links,
        parameters=parameters,
    )
    energy = sm_higgs_hamiltonian_density(field, momenta, links, parameters=parameters)
    vacuum = sm_constant_higgs(lattice_shape, vev=parameters.vev)
    identity_links = sm_identity_higgs_links(lattice_shape)
    pure_gauge_links = sm_pure_gauge_higgs_links_from_site_algebra(deterministic_higgs_site_theta(lattice_shape))
    pure_gauge_field = sm_transform_higgs_field(vacuum, sm_higgs_site_gauge_from_algebra(deterministic_higgs_site_theta(lattice_shape)))

    step_size = 0.01
    updated_field, updated_momenta = sm_higgs_leapfrog_step(field, momenta, links, step_size=step_size)
    reversed_field, reversed_momenta = sm_higgs_leapfrog_step(updated_field, -updated_momenta, links, step_size=step_size)
    jitted = jax.jit(sm_higgs_leapfrog_step, static_argnames=("step_size",))
    jit_field, jit_momenta = jitted(field, momenta, links, step_size=step_size)

    return HiggsDynamicsDiagnostics(
        generator_antihermitian_residual=sm_higgs_generator_antihermitian_residual(),
        link_unitarity_residual=sm_higgs_link_unitarity_residual(links),
        vacuum_force_norm=jnp.linalg.norm(sm_higgs_force(vacuum, identity_links, parameters=parameters)),
        pure_gauge_gradient_energy=sm_higgs_gradient_density(pure_gauge_field, pure_gauge_links, parameters=parameters),
        force_covariance_residual=jnp.max(jnp.abs(transformed_force - expected_force)),
        energy_gauge_invariance_residual=jnp.abs(transformed_energy - energy),
        hamiltonian_drift=jnp.abs(
            sm_higgs_hamiltonian_density(updated_field, updated_momenta, links, parameters=parameters) - energy,
        ),
        reversibility_residual=jnp.maximum(
            jnp.max(jnp.abs(reversed_field - field)),
            jnp.max(jnp.abs(reversed_momenta + momenta)),
        ),
        jit_delta_higgs=jnp.max(jnp.abs(jit_field - updated_field)),
        jit_delta_momenta=jnp.max(jnp.abs(jit_momenta - updated_momenta)),
    )
