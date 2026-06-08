"""Static Standard-Model gauge-background transport for QCA_SMv0.

This module intentionally owns its local implementation surface.  It imports
only generic simulator helpers from :mod:`clifford_3plus2_d5.sim` and the
Stage 1 BCC bulk walk from :mod:`clifford_3plus2_d5.qca_smv0`.

The internal carrier is a pragmatic simulator convention: one SM chiral-16
label set, duplicated to a 32-component internal register.  The hypercharges
use the left-handed chiral convention

``Q, u^c, d^c, L, e^c, nu^c = 1/6, -2/3, 1/3, -1/2, 1, 0``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from clifford_3plus2_d5.qca_smv0.bulk_bcc import BCC_DISPLACEMENTS, bcc_dirac_hop_matrices
from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.links import jax_identity_link_field, jax_transform_link_field, validate_link_field
from clifford_3plus2_d5.sim.state import norm_drift

SM_INTERNAL_DIM = 32
SM_CHIRAL16_DIM = 16
SM_GENERATOR_COUNT = 12
BCC_PLAQUETTE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 3),
    (0, 5),
    (0, 6),
    (1, 2),
    (1, 7),
    (2, 4),
)


class StaticSMGaugeDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 2 static SM gauge backgrounds."""

    generator_count: int
    generator_antihermitian_residual: jnp.ndarray
    link_unitarity_residual: jnp.ndarray
    identity_reduction_residual: jnp.ndarray
    norm_drift: jnp.ndarray
    gauge_covariance_residual: jnp.ndarray
    identity_wilson_action: jnp.ndarray
    pure_gauge_wilson_action: jnp.ndarray
    nontrivial_wilson_action: jnp.ndarray
    wilson_trace_covariance_residual: jnp.ndarray
    continuum_residual_ratio: jnp.ndarray
    jit_delta: jnp.ndarray


def _q_index(color: int, weak: int) -> int:
    return 2 * color + weak


def _u_c_index(color: int) -> int:
    return 6 + color


def _d_c_index(color: int) -> int:
    return 9 + color


def _l_index(weak: int) -> int:
    return 12 + weak


def _e_c_index() -> int:
    return 14


def _nu_c_index() -> int:
    return 15


def _pauli_hermitian() -> np.ndarray:
    return np.asarray(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=np.complex64,
    )


def _gell_mann_hermitian() -> np.ndarray:
    zero = np.zeros((3, 3), dtype=np.complex64)
    matrices = []
    for entries in (
        ((0, 1, 1), (1, 0, 1)),
        ((0, 1, -1j), (1, 0, 1j)),
        ((0, 0, 1), (1, 1, -1)),
        ((0, 2, 1), (2, 0, 1)),
        ((0, 2, -1j), (2, 0, 1j)),
        ((1, 2, 1), (2, 1, 1)),
        ((1, 2, -1j), (2, 1, 1j)),
    ):
        matrix = zero.copy()
        for row, col, value in entries:
            matrix[row, col] = value
        matrices.append(matrix)
    lambda8 = np.diag([1, 1, -2]).astype(np.complex64) / np.sqrt(np.float32(3.0))
    matrices.append(lambda8)
    return np.stack(matrices, axis=0)


def _embed_single_copy(generators16: np.ndarray, block16: np.ndarray) -> np.ndarray:
    generators32 = np.zeros((generators16.shape[0], SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=np.complex64)
    generators32[:, :SM_CHIRAL16_DIM, :SM_CHIRAL16_DIM] = generators16
    generators32[:, SM_CHIRAL16_DIM:, SM_CHIRAL16_DIM:] = block16
    return generators32


@lru_cache(maxsize=4)
def _sm_generators_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    generators16 = np.zeros((SM_GENERATOR_COUNT, SM_CHIRAL16_DIM, SM_CHIRAL16_DIM), dtype=np.complex64)

    su3_fund = 0.5j * _gell_mann_hermitian()
    su3_antifund = -0.5j * np.conj(_gell_mann_hermitian())
    su2_fund = 0.5j * _pauli_hermitian()

    for gen_index in range(8):
        fundamental = su3_fund[gen_index]
        antifundamental = su3_antifund[gen_index]
        for weak in range(2):
            for row in range(3):
                for col in range(3):
                    generators16[gen_index, _q_index(row, weak), _q_index(col, weak)] = fundamental[row, col]
        for row in range(3):
            for col in range(3):
                generators16[gen_index, _u_c_index(row), _u_c_index(col)] = antifundamental[row, col]
                generators16[gen_index, _d_c_index(row), _d_c_index(col)] = antifundamental[row, col]

    for gen_index in range(3):
        matrix = su2_fund[gen_index]
        target = 8 + gen_index
        for color in range(3):
            for row in range(2):
                for col in range(2):
                    generators16[target, _q_index(color, row), _q_index(color, col)] = matrix[row, col]
        for row in range(2):
            for col in range(2):
                generators16[target, _l_index(row), _l_index(col)] = matrix[row, col]

    hypercharges = np.zeros((SM_CHIRAL16_DIM,), dtype=np.float32)
    for color in range(3):
        for weak in range(2):
            hypercharges[_q_index(color, weak)] = 1.0 / 6.0
        hypercharges[_u_c_index(color)] = -2.0 / 3.0
        hypercharges[_d_c_index(color)] = 1.0 / 3.0
    hypercharges[_l_index(0)] = -0.5
    hypercharges[_l_index(1)] = -0.5
    hypercharges[_e_c_index()] = 1.0
    hypercharges[_nu_c_index()] = 0.0
    generators16[11] = 1j * np.diag(hypercharges)

    generators32 = _embed_single_copy(generators16, generators16)
    return generators32.astype(dtype, copy=False)


def sm_generators(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return local anti-Hermitian SM generators with shape ``(12,32,32)``."""

    dtype = np.dtype(dtype)
    return jnp.asarray(_sm_generators_numpy(dtype.name), dtype=dtype)


def sm_generator_antihermitian_residual(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return ``max_abs(T + T^dagger)`` over the local SM generator basis."""

    generators = sm_generators(dtype=dtype)
    return jnp.max(jnp.abs(generators + jnp.swapaxes(jnp.conj(generators), -1, -2)))


def _validate_sm_theta(theta: jnp.ndarray, *, with_edges: bool) -> tuple[int, int, int]:
    expected_tail = (8, SM_GENERATOR_COUNT) if with_edges else (SM_GENERATOR_COUNT,)
    if theta.ndim != (5 if with_edges else 4) or theta.shape[-len(expected_tail) :] != expected_tail:
        shape = "(nx, ny, nz, 8, 12)" if with_edges else "(nx, ny, nz, 12)"
        raise ValueError(f"SM algebra coordinates must have shape {shape}")
    return int(theta.shape[0]), int(theta.shape[1]), int(theta.shape[2])


def _validate_sm_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 5 or state.shape[-2:] != (4, SM_INTERNAL_DIM):
        raise ValueError("SM Dirac state must have shape (nx, ny, nz, 4, 32)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_sm_links(links: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> None:
    validate_link_field(links, edge_count=8)
    if links.ndim != 6 or links.shape[3:] != (8, SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("SM links must have shape (nx, ny, nz, 8, 32, 32)")
    if lattice_shape is not None and links.shape[:3] != lattice_shape:
        raise ValueError("SM links must match the state lattice")


def sm_algebra_matrix_field(theta: jnp.ndarray) -> jnp.ndarray:
    """Return SM algebra matrices from site or link coordinates."""

    if theta.ndim == 4:
        _validate_sm_theta(theta, with_edges=False)
    elif theta.ndim == 5:
        _validate_sm_theta(theta, with_edges=True)
    else:
        raise ValueError("SM algebra coordinates must be site or link fields")
    generators = sm_generators(dtype=jnp.result_type(theta, 1j))
    return jnp.einsum("...a,aij->...ij", theta, generators)


def _matrix_exponential_field(algebra: jnp.ndarray) -> jnp.ndarray:
    flat = algebra.reshape((-1, algebra.shape[-2], algebra.shape[-1]))
    links = jax.vmap(jsp_linalg.expm)(flat)
    return links.reshape(algebra.shape)


def sm_link_field_from_algebra(theta: jnp.ndarray) -> jnp.ndarray:
    """Return finite static SM links from coordinates ``(nx,ny,nz,8,12)``."""

    _validate_sm_theta(theta, with_edges=True)
    return _matrix_exponential_field(sm_algebra_matrix_field(theta))


def sm_site_gauge_from_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return finite site-local SM gauges from coordinates ``(nx,ny,nz,12)``."""

    _validate_sm_theta(site_theta, with_edges=False)
    return _matrix_exponential_field(sm_algebra_matrix_field(site_theta))


def sm_identity_links(
    lattice_shape: tuple[int, int, int],
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return identity SM links in qca_smv0 BCC edge order."""

    return jax_identity_link_field(lattice_shape, SM_INTERNAL_DIM, dtype=dtype)


def sm_transform_state(state: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply ``psi[x] -> G[x] psi[x]`` to an SM Dirac/internal state."""

    lattice_shape = _validate_sm_state(state)
    if site_gauge.ndim != 5 or site_gauge.shape[:3] != lattice_shape or site_gauge.shape[-2:] != (
        SM_INTERNAL_DIM,
        SM_INTERNAL_DIM,
    ):
        raise ValueError("site_gauge must have shape (nx, ny, nz, 32, 32)")
    return jnp.einsum("...ab,...sb->...sa", site_gauge, state)


def sm_transform_links(links: jnp.ndarray, site_gauge: jnp.ndarray) -> jnp.ndarray:
    """Apply pull-convention site-local SM gauge transforms to links."""

    _validate_sm_links(links)
    return jax_transform_link_field(links, site_gauge, BCC_DISPLACEMENTS)


def sm_pure_gauge_links_from_site_algebra(site_theta: jnp.ndarray) -> jnp.ndarray:
    """Return pure-gauge SM BCC links generated from site-local gauges."""

    lattice_shape = _validate_sm_theta(site_theta, with_edges=False)
    site_gauge = sm_site_gauge_from_algebra(site_theta)
    return sm_transform_links(sm_identity_links(lattice_shape, dtype=site_gauge.dtype), site_gauge)


def sm_link_unitarity_residual(links: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(U^dagger U-I)`` over an SM link field."""

    _validate_sm_links(links)
    identity = jnp.eye(SM_INTERNAL_DIM, dtype=links.dtype)
    gram = jnp.einsum("...ji,...jk->...ik", jnp.conj(links), links)
    return jnp.max(jnp.abs(gram - identity))


def sm_gauged_dirac_step(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Apply one free Dirac BCC step through static SM background links."""

    lattice_shape = _validate_sm_state(state)
    _validate_sm_links(links, lattice_shape)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        linked = jnp.einsum("...ab,...sb->...sa", links[..., index, :, :], source)
        out = out + jnp.einsum("rs,...sd->...rd", hops[index], linked)
    return out


def sm_free_dirac_internal_step(state: jnp.ndarray) -> jnp.ndarray:
    """Apply the free BCC Dirac step independently to every internal component."""

    _validate_sm_state(state)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        out = out + jnp.einsum("rs,...sd->...rd", hops[index], source)
    return out


def sm_gauged_norm_drift(state: jnp.ndarray, links: jnp.ndarray) -> jnp.ndarray:
    """Return squared-norm drift after one static SM-linked Dirac step."""

    return norm_drift(state, sm_gauged_dirac_step(state, links))


def sm_linearized_gauged_dirac_step(
    state: jnp.ndarray,
    theta: jnp.ndarray,
    *,
    epsilon: float,
) -> jnp.ndarray:
    """Return the first-order weak-link approximation to the gauged step."""

    lattice_shape = _validate_sm_state(state)
    _validate_sm_theta(theta, with_edges=True)
    if theta.shape[:3] != lattice_shape:
        raise ValueError("theta must match the state lattice")
    algebra = sm_algebra_matrix_field(theta)
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    eps = jnp.asarray(epsilon, dtype=jnp.real(state).dtype)
    out = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        insertion = jnp.einsum("...ab,...sb->...sa", algebra[..., index, :, :], source)
        linked = source + eps * insertion
        out = out + jnp.einsum("rs,...sd->...rd", hops[index], linked)
    return out


def sm_continuum_linearization_residual(state: jnp.ndarray, theta: jnp.ndarray, *, epsilon: float) -> jnp.ndarray:
    """Return norm of exact weak-link step minus first-order approximation."""

    links = sm_link_field_from_algebra(jnp.asarray(epsilon, dtype=theta.dtype) * theta)
    exact = sm_gauged_dirac_step(state, links)
    linear = sm_linearized_gauged_dirac_step(state, theta, epsilon=epsilon)
    return jnp.linalg.norm((exact - linear).reshape((-1,)))


def _link_dagger(link: jnp.ndarray) -> jnp.ndarray:
    return jnp.swapaxes(jnp.conj(link), -1, -2)


def sm_plaquette_holonomies(links: jnp.ndarray) -> jnp.ndarray:
    """Return local BCC parallelogram holonomies for selected edge pairs."""

    _validate_sm_links(links)
    holonomies = []
    for first, second in BCC_PLAQUETTE_PAIRS:
        displacement_a = BCC_DISPLACEMENTS[first]
        displacement_b = BCC_DISPLACEMENTS[second]
        link_a = links[..., first, :, :]
        link_b = links[..., second, :, :]
        link_a_at_b = source_roll(link_a, displacement_b)
        link_b_at_a = source_roll(link_b, displacement_a)
        holonomies.append(
            jnp.einsum(
                "...ab,...bc,...cd,...de->...ae",
                link_b,
                link_a_at_b,
                _link_dagger(link_b_at_a),
                _link_dagger(link_a),
            ),
        )
    return jnp.stack(holonomies, axis=3)


def sm_normalized_wilson_traces(links: jnp.ndarray) -> jnp.ndarray:
    """Return normalized plaquette traces over the selected BCC pairs."""

    holonomies = sm_plaquette_holonomies(links)
    return jnp.trace(holonomies, axis1=-2, axis2=-1) / jnp.asarray(SM_INTERNAL_DIM, dtype=holonomies.dtype)


def sm_average_normalized_wilson_loop(links: jnp.ndarray) -> jnp.ndarray:
    """Return the average normalized Wilson trace for static SM links."""

    return jnp.mean(sm_normalized_wilson_traces(links))


def sm_average_wilson_action_density(links: jnp.ndarray) -> jnp.ndarray:
    """Return ``mean(1-Re Tr(W)/32)`` over selected BCC plaquettes."""

    traces = sm_normalized_wilson_traces(links)
    return jnp.real(jnp.mean(1.0 - traces))


def deterministic_sm_state(lattice_shape: tuple[int, int, int] = (2, 1, 1)) -> jnp.ndarray:
    """Return a deterministic finite SM Dirac/internal state for audits."""

    state = jnp.zeros((*lattice_shape, 4, SM_INTERNAL_DIM), dtype=jnp.complex64)
    state = state.at[0, 0, 0, 0, 0].set(1.0 + 0.25j)
    state = state.at[0, 0, 0, 1, 7].set(-0.5 + 0.125j)
    state = state.at[-1, 0, 0, 2, 18].set(0.375 - 0.25j)
    state = state.at[-1, 0, 0, 3, 31].set(-0.125 + 0.5j)
    return state


def deterministic_sm_link_theta(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
    *,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Return deterministic non-pure SM link coordinates for audits."""

    x, y, z, edge, gen = jnp.indices((*lattice_shape, 8, SM_GENERATOR_COUNT), dtype=jnp.float32)
    return scale * (
        0.11 * (x + 1.0)
        - 0.07 * (y + 1.0)
        + 0.05 * (z + 1.0)
        + 0.013 * (edge + 1.0)
        - 0.017 * (gen + 1.0)
    )


def deterministic_sm_site_theta(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
    *,
    scale: float = 0.04,
) -> jnp.ndarray:
    """Return deterministic site-local SM gauge coordinates for audits."""

    x, y, z, gen = jnp.indices((*lattice_shape, SM_GENERATOR_COUNT), dtype=jnp.float32)
    return scale * (0.19 * (x + 1.0) - 0.03 * y + 0.02 * z + 0.007 * (gen + 1.0))


def static_sm_gauge_diagnostics() -> StaticSMGaugeDiagnostics:
    """Return the focused Stage 2 static SM gauge-background diagnostics."""

    lattice_shape = (2, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    theta = deterministic_sm_link_theta(lattice_shape)
    links = sm_link_field_from_algebra(theta)
    identity_links = sm_identity_links(lattice_shape, dtype=state.dtype)
    site_theta = deterministic_sm_site_theta(lattice_shape)
    site_gauge = sm_site_gauge_from_algebra(site_theta)
    pure_links = sm_pure_gauge_links_from_site_algebra(site_theta)

    transformed_state = sm_transform_state(state, site_gauge)
    transformed_links = sm_transform_links(links, site_gauge)
    gauge_covariance_residual = jnp.max(
        jnp.abs(
            sm_gauged_dirac_step(transformed_state, transformed_links)
            - sm_transform_state(sm_gauged_dirac_step(state, links), site_gauge),
        ),
    )

    epsilon = 0.05
    residual_epsilon = sm_continuum_linearization_residual(state, theta, epsilon=epsilon)
    residual_half = sm_continuum_linearization_residual(state, theta, epsilon=0.5 * epsilon)
    continuum_ratio = residual_half / residual_epsilon

    trace_original = sm_average_normalized_wilson_loop(links)
    trace_transformed = sm_average_normalized_wilson_loop(transformed_links)
    jitted_step = jax.jit(sm_gauged_dirac_step)

    return StaticSMGaugeDiagnostics(
        generator_count=SM_GENERATOR_COUNT,
        generator_antihermitian_residual=sm_generator_antihermitian_residual(dtype=state.dtype),
        link_unitarity_residual=sm_link_unitarity_residual(links),
        identity_reduction_residual=jnp.max(
            jnp.abs(sm_gauged_dirac_step(state, identity_links) - sm_free_dirac_internal_step(state)),
        ),
        norm_drift=sm_gauged_norm_drift(state, links),
        gauge_covariance_residual=gauge_covariance_residual,
        identity_wilson_action=sm_average_wilson_action_density(identity_links),
        pure_gauge_wilson_action=sm_average_wilson_action_density(pure_links),
        nontrivial_wilson_action=sm_average_wilson_action_density(links),
        wilson_trace_covariance_residual=jnp.abs(trace_transformed - trace_original),
        continuum_residual_ratio=continuum_ratio,
        jit_delta=jnp.max(jnp.abs(jitted_step(state, links) - sm_gauged_dirac_step(state, links))),
    )
