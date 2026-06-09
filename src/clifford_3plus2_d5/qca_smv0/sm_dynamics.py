"""Pure dynamic SM gauge fields for QCA_SMv0.

Stage 3 adds a compact pure-gauge Hamiltonian layer on top of the Stage 2
static Standard-Model links.  Fermions remain spectators: they can be
transported through the evolved links, but they do not source the gauge
momenta in this stage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    BCC_PLAQUETTE_PAIRS,
    SM_GENERATOR_COUNT,
    SM_INTERNAL_DIM,
    deterministic_sm_link_theta,
    deterministic_sm_site_theta,
    deterministic_sm_state,
    sm_algebra_matrix_field,
    sm_average_wilson_action_density,
    sm_gauged_dirac_step,
    sm_generators,
    sm_link_field_from_algebra,
    sm_link_unitarity_residual,
    sm_plaquette_holonomies,
    sm_site_gauge_from_algebra,
    sm_transform_links,
)
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.state import state_norm_squared


class DynamicSMGaugeDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 3 pure dynamic gauge fields."""

    projection_roundtrip_residual: jnp.ndarray
    identity_force_norm: jnp.ndarray
    pure_gauge_force_norm: jnp.ndarray
    nonflat_force_norm: jnp.ndarray
    link_unitarity_after_leapfrog: jnp.ndarray
    hamiltonian_drift: jnp.ndarray
    reversibility_residual: jnp.ndarray
    momentum_covariance_residual: jnp.ndarray
    gauss_zero_residual: jnp.ndarray
    gauss_covariance_residual: jnp.ndarray
    gauss_pure_leapfrog_residual: jnp.ndarray
    weak_field_residual_ratio: jnp.ndarray
    yang_mills_action_ratio: jnp.ndarray
    spectator_norm_drift: jnp.ndarray
    jit_delta_links: jnp.ndarray
    jit_delta_momenta: jnp.ndarray


def _validate_sm_momenta(momenta: jnp.ndarray, links: jnp.ndarray | None = None) -> None:
    if momenta.ndim != 5 or momenta.shape[-2:] != (8, SM_GENERATOR_COUNT):
        raise ValueError("SM momenta must have shape (nx, ny, nz, 8, 12)")
    if links is not None and momenta.shape[:4] != links.shape[:4]:
        raise ValueError("SM momenta and links must share shape (nx, ny, nz, 8)")


@lru_cache(maxsize=4)
def _generator_gram_numpy(dtype_name: str) -> np.ndarray:
    generators = np.asarray(sm_generators(dtype=np.dtype(dtype_name)))
    daggers = np.swapaxes(np.conj(generators), -1, -2)
    gram = np.real(np.einsum("aij,bji->ab", daggers, generators))
    return gram.astype(np.float32, copy=False)


def sm_generator_gram(*, dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the real Hilbert-Schmidt Gram matrix of the SM generator basis."""

    return jnp.asarray(_generator_gram_numpy(np.dtype(dtype).name), dtype=jnp.float32)


def sm_project_to_coordinates(matrix: jnp.ndarray) -> jnp.ndarray:
    """Project SM algebra matrices onto the 12-coordinate generator basis."""

    if matrix.shape[-2:] != (SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("matrix must have trailing shape (32, 32)")
    generators = sm_generators(dtype=jnp.result_type(matrix, 1j))
    daggers = jnp.swapaxes(jnp.conj(generators), -1, -2)
    gram = sm_generator_gram(dtype=generators.dtype)
    rhs = jnp.real(jnp.einsum("aij,...ji->...a", daggers, matrix.astype(generators.dtype)))
    flat_rhs = rhs.reshape((-1, SM_GENERATOR_COUNT))
    flat_coordinates = jnp.linalg.solve(gram, flat_rhs.T).T
    return flat_coordinates.reshape(rhs.shape)


def sm_momentum_algebra(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return anti-Hermitian momentum matrices with shape ``(...,8,32,32)``."""

    _validate_sm_momenta(momenta)
    return sm_algebra_matrix_field(momenta)


def sm_transform_momenta(momenta: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply target-site adjoint gauge transforms to link momenta."""

    _validate_sm_momenta(momenta)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != momenta.shape[:3] or site_gauge.shape[-2:] != (
        SM_INTERNAL_DIM,
        SM_INTERNAL_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 32, 32)")
    momentum_matrices = sm_momentum_algebra(momenta)
    gauge_dagger = jnp.swapaxes(jnp.conj(site_gauge), -1, -2)
    transformed = jnp.einsum("...ab,...hbc,...cd->...had", site_gauge, momentum_matrices, gauge_dagger)
    return sm_project_to_coordinates(transformed)


def sm_momentum_kinetic_energy_density(momenta: jnp.ndarray) -> jnp.ndarray:
    """Return ``0.5 mean(Re Tr(P^dagger P)/32)`` for SM link momenta."""

    momentum_matrices = sm_momentum_algebra(momenta)
    densities = jnp.real(jnp.einsum("...ij,...ji->...", jnp.swapaxes(jnp.conj(momentum_matrices), -1, -2), momentum_matrices))
    return 0.5 * jnp.mean(densities / jnp.asarray(SM_INTERNAL_DIM, dtype=densities.dtype))


def sm_gauge_hamiltonian_density(links: jnp.ndarray, momenta: jnp.ndarray, *, beta: float = 1.0) -> jnp.ndarray:
    """Return pure-gauge kinetic plus Wilson-action density."""

    _validate_sm_momenta(momenta, links)
    return sm_momentum_kinetic_energy_density(momenta) + jnp.asarray(beta, dtype=jnp.real(links).dtype) * (
        sm_average_wilson_action_density(links)
    )


def _left_updated_links(links: jnp.ndarray, update: jnp.ndarray, link_index: tuple[int, int, int, int]) -> jnp.ndarray:
    updated_link = update @ links[link_index]
    return links.at[link_index].set(updated_link)


def _generator_left_updates(links: jnp.ndarray, epsilon: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    generators = sm_generators(dtype=links.dtype)
    plus = jax.vmap(jsp_linalg.expm)(step * generators)
    minus = jax.vmap(jsp_linalg.expm)(-step * generators)
    return plus, minus


def _force_coordinate_indices(links: jnp.ndarray) -> np.ndarray:
    return np.asarray(list(np.ndindex(*links.shape[:4], SM_GENERATOR_COUNT)), dtype=np.int32)


def sm_left_wilson_force(links: jnp.ndarray, *, epsilon: float = 1e-3) -> jnp.ndarray:
    """Return centered finite-difference left-trivialized Wilson force."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    real_dtype = jnp.real(jnp.asarray(0, dtype=links.dtype)).dtype
    step = jnp.asarray(epsilon, dtype=real_dtype)
    plus_updates, minus_updates = _generator_left_updates(links, epsilon)
    coordinate_indices = jnp.asarray(_force_coordinate_indices(links))

    def action_for_index(index: jnp.ndarray, updates: jnp.ndarray) -> jnp.ndarray:
        update = updates[index[4]]
        updated_link = update @ links[index[0], index[1], index[2], index[3]]
        perturbed_links = links.at[index[0], index[1], index[2], index[3]].set(updated_link)
        return sm_average_wilson_action_density(perturbed_links)

    batched_action = jax.vmap(action_for_index, in_axes=(0, None))
    derivatives = (batched_action(coordinate_indices, plus_updates) - batched_action(coordinate_indices, minus_updates)) / (
        2 * step
    )
    force = jnp.zeros((*links.shape[:4], SM_GENERATOR_COUNT), dtype=real_dtype)
    force = force.at[
        coordinate_indices[:, 0],
        coordinate_indices[:, 1],
        coordinate_indices[:, 2],
        coordinate_indices[:, 3],
        coordinate_indices[:, 4],
    ].set(derivatives)
    return force


def sm_linearized_plaquette_field_strength(theta: jnp.ndarray) -> jnp.ndarray:
    """Return the first-order BCC plaquette field strength matrices.

    For Stage 2/3 pull links, the selected plaquette holonomy is
    ``U_b[x] U_a[x+b] U_b[x+a]^dag U_a[x]^dag``.  If
    ``U_h = exp(epsilon A_h)``, then to first order
    ``W_ab = I + epsilon F_ab + O(epsilon^2)`` with
    ``F_ab = A_b[x] + A_a[x+b] - A_b[x+a] - A_a[x]``.
    """

    if theta.ndim != 5 or theta.shape[-2:] != (8, SM_GENERATOR_COUNT):
        raise ValueError("theta must have shape (nx, ny, nz, 8, 12)")
    algebra = sm_algebra_matrix_field(theta)
    strengths = []
    for first, second in BCC_PLAQUETTE_PAIRS:
        displacement_a = BCC_DISPLACEMENTS[first]
        displacement_b = BCC_DISPLACEMENTS[second]
        algebra_a = algebra[..., first, :, :]
        algebra_b = algebra[..., second, :, :]
        strengths.append(
            algebra_b
            + source_roll(algebra_a, displacement_b)
            - source_roll(algebra_b, displacement_a)
            - algebra_a,
        )
    return jnp.stack(strengths, axis=3)


def sm_linearized_yang_mills_action_density(theta: jnp.ndarray) -> jnp.ndarray:
    """Return the weak-field ``0.5 Tr(F^dag F)/32`` plaquette density."""

    field_strength = sm_linearized_plaquette_field_strength(theta)
    density = jnp.real(
        jnp.einsum(
            "...ij,...ji->...",
            jnp.swapaxes(jnp.conj(field_strength), -1, -2),
            field_strength,
        ),
    )
    return 0.5 * jnp.mean(density / jnp.asarray(SM_INTERNAL_DIM, dtype=density.dtype))


def sm_weak_field_holonomy_residual(theta: jnp.ndarray, *, epsilon: float) -> jnp.ndarray:
    """Return ``||W - (I + epsilon F)||`` for weak-field plaquettes."""

    links = sm_link_field_from_algebra(jnp.asarray(epsilon, dtype=theta.dtype) * theta)
    exact = sm_plaquette_holonomies(links)
    field_strength = sm_linearized_plaquette_field_strength(theta)
    identity = jnp.eye(SM_INTERNAL_DIM, dtype=exact.dtype)
    approximate = identity + jnp.asarray(epsilon, dtype=exact.dtype) * field_strength
    return jnp.linalg.norm((exact - approximate).reshape((-1,)))


def sm_weak_field_yang_mills_action_ratio(theta: jnp.ndarray, *, epsilon: float) -> jnp.ndarray:
    """Return exact Wilson action divided by the weak-field Yang-Mills action."""

    links = sm_link_field_from_algebra(jnp.asarray(epsilon, dtype=theta.dtype) * theta)
    exact_action = sm_average_wilson_action_density(links)
    linear_action = sm_linearized_yang_mills_action_density(theta)
    return exact_action / ((jnp.asarray(epsilon, dtype=theta.dtype) ** 2) * linear_action)


def sm_apply_momentum_update(links: jnp.ndarray, momenta: jnp.ndarray, *, step_size: float) -> jnp.ndarray:
    """Apply ``U -> exp(step_size P) U`` to every SM link."""

    _validate_sm_momenta(momenta, links)
    updates = sm_link_field_from_algebra(jnp.asarray(step_size, dtype=momenta.dtype) * momenta)
    return jnp.einsum("...ij,...jk->...ik", updates, links)


def sm_leapfrog_step(
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    force_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one reversible pure-gauge leapfrog update."""

    _validate_sm_momenta(momenta, links)
    dt = jnp.asarray(step_size, dtype=momenta.dtype)
    coupling = jnp.asarray(beta, dtype=momenta.dtype)
    first_force = sm_left_wilson_force(links, epsilon=force_epsilon)
    half_momenta = momenta - 0.5 * dt * coupling * first_force
    updated_links = sm_apply_momentum_update(links, half_momenta, step_size=step_size)
    second_force = sm_left_wilson_force(updated_links, epsilon=force_epsilon)
    updated_momenta = half_momenta - 0.5 * dt * coupling * second_force
    return updated_links, updated_momenta


def sm_no_backreaction_fermion_gauge_step(
    state: jnp.ndarray,
    links: jnp.ndarray,
    momenta: jnp.ndarray,
    *,
    step_size: float,
    beta: float = 1.0,
    force_epsilon: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve pure gauge fields, then transport fermions through the new links."""

    updated_links, updated_momenta = sm_leapfrog_step(
        links,
        momenta,
        step_size=step_size,
        beta=beta,
        force_epsilon=force_epsilon,
    )
    updated_state = sm_gauged_dirac_step(state, updated_links)
    return updated_state, updated_links, updated_momenta


@lru_cache(maxsize=1)
def _minus_displacement_indices() -> tuple[int, ...]:
    return tuple(
        BCC_DISPLACEMENTS.index(tuple(-component for component in displacement))
        for displacement in BCC_DISPLACEMENTS
    )


def sm_electric_divergence(links: jnp.ndarray, momenta: jnp.ndarray) -> jnp.ndarray:
    """Return target-site electric divergence in SM algebra coordinates."""

    _validate_sm_momenta(momenta, links)
    momentum_matrices = sm_momentum_algebra(momenta)
    divergence = jnp.zeros((*links.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=links.dtype)
    minus_indices = _minus_displacement_indices()
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        outgoing_index = minus_indices[index]
        incoming = momentum_matrices[..., index, :, :]
        outgoing_momentum = source_roll(momentum_matrices[..., outgoing_index, :, :], displacement)
        outgoing_link = source_roll(links[..., outgoing_index, :, :], displacement)
        outgoing_link_dagger = jnp.swapaxes(jnp.conj(outgoing_link), -1, -2)
        transported_outgoing = outgoing_link_dagger @ outgoing_momentum @ outgoing_link
        divergence = divergence + incoming - transported_outgoing
    return sm_project_to_coordinates(divergence)


def deterministic_sm_momenta(
    lattice_shape: tuple[int, int, int] = (1, 1, 1),
    *,
    scale: float = 0.025,
) -> jnp.ndarray:
    """Return deterministic real SM link momenta for Stage 3 audits."""

    x, y, z, edge, gen = jnp.indices((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)
    return scale * (
        0.03 * (x + 1.0)
        - 0.02 * (y + 1.0)
        + 0.017 * (z + 1.0)
        + 0.011 * (edge + 1.0)
        - 0.005 * (gen + 1.0)
    )


def deterministic_sm_curved_link_theta(
    lattice_shape: tuple[int, int, int] = (2, 2, 2),
    *,
    scale: float = 0.3,
) -> jnp.ndarray:
    """Return a deterministic weak-field profile with nonzero lattice curl."""

    x, y, z, edge, gen = jnp.indices((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)
    return scale * (
        0.07 * (x + 1.0) * (edge + 1.0)
        - 0.04 * (y + 1.0) * (gen + 1.0)
        + 0.03 * (z + 1.0) * (edge + gen + 1.0)
        + 0.011 * (x + 1.0) * (y + 1.0) * (gen + 1.0)
        + 0.005 * (x + 1.0) * (z + 1.0) * (edge + 1.0) * (gen + 1.0)
    )


def dynamic_sm_gauge_diagnostics() -> DynamicSMGaugeDiagnostics:
    """Return the focused Stage 3 pure dynamic gauge diagnostics."""

    lattice_shape = (1, 1, 1)
    theta = deterministic_sm_link_theta(lattice_shape, scale=0.6)
    links = sm_link_field_from_algebra(theta)
    force_links = sm_link_field_from_algebra(deterministic_sm_link_theta(lattice_shape, scale=2.0))
    momenta = deterministic_sm_momenta(lattice_shape)
    zero_momenta = jnp.zeros_like(momenta)
    site_gauge = sm_site_gauge_from_algebra(deterministic_sm_site_theta(lattice_shape))
    pure_links = sm_transform_links(
        sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)),
        site_gauge,
    )
    weak_theta = deterministic_sm_curved_link_theta()

    momentum_matrices = sm_momentum_algebra(momenta)
    projection_roundtrip = sm_project_to_coordinates(momentum_matrices)
    transformed_momenta = sm_transform_momenta(momenta, site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    transformed_momentum_matrices = sm_momentum_algebra(transformed_momenta)
    expected_transformed_matrices = jnp.einsum(
        "...ab,...hbc,...cd->...had",
        site_gauge,
        momentum_matrices,
        jnp.swapaxes(jnp.conj(site_gauge), -1, -2),
    )

    step_size = 0.02
    hamiltonian_before = sm_gauge_hamiltonian_density(links, momenta)
    updated_links, updated_momenta = sm_leapfrog_step(links, momenta, step_size=step_size)
    hamiltonian_after = sm_gauge_hamiltonian_density(updated_links, updated_momenta)
    reversed_links, reversed_momenta = sm_leapfrog_step(
        updated_links,
        -updated_momenta,
        step_size=step_size,
    )

    divergence = sm_electric_divergence(links, momenta)
    transformed_divergence = sm_electric_divergence(transformed_links, transformed_momenta)
    divergence_matrix = sm_algebra_matrix_field(divergence)
    transformed_divergence_matrix = sm_algebra_matrix_field(transformed_divergence)
    expected_divergence_matrix = site_gauge @ divergence_matrix @ jnp.swapaxes(jnp.conj(site_gauge), -1, -2)
    pure_updated_links, pure_updated_momenta = sm_leapfrog_step(
        pure_links,
        zero_momenta,
        step_size=step_size,
    )

    weak_epsilon = 0.04
    weak_residual = sm_weak_field_holonomy_residual(weak_theta, epsilon=weak_epsilon)
    weak_residual_half = sm_weak_field_holonomy_residual(weak_theta, epsilon=0.5 * weak_epsilon)

    state = deterministic_sm_state(lattice_shape)
    updated_state = sm_gauged_dirac_step(state, updated_links)
    jitted_leapfrog = jax.jit(sm_leapfrog_step, static_argnames=("step_size", "beta", "force_epsilon"))
    jit_links, jit_momenta = jitted_leapfrog(links, momenta, step_size=step_size)

    return DynamicSMGaugeDiagnostics(
        projection_roundtrip_residual=jnp.max(jnp.abs(projection_roundtrip - momenta)),
        identity_force_norm=jnp.linalg.norm(
            sm_left_wilson_force(
                sm_link_field_from_algebra(jnp.zeros((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)),
            ),
        ),
        pure_gauge_force_norm=jnp.linalg.norm(sm_left_wilson_force(pure_links)),
        nonflat_force_norm=jnp.linalg.norm(sm_left_wilson_force(force_links)),
        link_unitarity_after_leapfrog=sm_link_unitarity_residual(updated_links),
        hamiltonian_drift=jnp.abs(hamiltonian_after - hamiltonian_before),
        reversibility_residual=jnp.max(jnp.abs(reversed_links - links))
        + jnp.max(jnp.abs(reversed_momenta + momenta)),
        momentum_covariance_residual=jnp.max(jnp.abs(transformed_momentum_matrices - expected_transformed_matrices)),
        gauss_zero_residual=jnp.linalg.norm(sm_electric_divergence(links, zero_momenta)),
        gauss_covariance_residual=jnp.max(jnp.abs(transformed_divergence_matrix - expected_divergence_matrix)),
        gauss_pure_leapfrog_residual=jnp.linalg.norm(sm_electric_divergence(pure_updated_links, pure_updated_momenta)),
        weak_field_residual_ratio=weak_residual_half / weak_residual,
        yang_mills_action_ratio=sm_weak_field_yang_mills_action_ratio(weak_theta, epsilon=weak_epsilon),
        spectator_norm_drift=jnp.abs(state_norm_squared(updated_state) - state_norm_squared(state)),
        jit_delta_links=jnp.max(jnp.abs(jit_links - updated_links)),
        jit_delta_momenta=jnp.max(jnp.abs(jit_momenta - updated_momenta)),
    )
