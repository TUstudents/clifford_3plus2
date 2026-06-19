"""Bare bulk BCC Weyl walk for the QCA_SMv0 sidecar.

This module contains only the free two-component bulk walker.  It does not
encode boundary conditions, gauge links, Higgs fields, flavor registers, or
recirculation rules.

The position-space convention follows :mod:`clifford_3plus2_d5.sim.lattice`:
``source_roll(psi, h)`` reads ``psi[x + h]``.  With that pull convention the
Bloch symbol is

``U(k) = S_x(k_x) S_y(k_y) S_z(k_z)``,

where ``S_j(k_j) = P_j^+ exp(i k_j) + P_j^- exp(-i k_j)``.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE
from clifford_3plus2_d5.sim.lattice import source_roll
from clifford_3plus2_d5.sim.state import norm_drift

BCC_WEYL_SPIN_DIM = 2
BCC_DIRAC_SPIN_DIM = 4
BCC_DISPLACEMENTS: tuple[tuple[int, int, int], ...] = tuple(product((-1, 1), repeat=3))
BCC_ANISOTROPY_DIRECTIONS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 2.0, 3.0),
)


class BareBCCWalkDiagnostics(NamedTuple):
    """Small diagnostics payload for the free bulk BCC walk."""

    hop_count: int
    hop_completeness_residual: jnp.ndarray
    symbol_unitarity_residual: jnp.ndarray
    norm_drift: jnp.ndarray
    small_k_speed_error: jnp.ndarray
    anisotropy_spreads: jnp.ndarray
    anisotropy_halving_ratios: jnp.ndarray
    dirac_hop_completeness_residual: jnp.ndarray
    dirac_symbol_unitarity_residual: jnp.ndarray
    dirac_norm_drift: jnp.ndarray
    dirac_small_k_speed_error: jnp.ndarray


def _validate_weyl_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 4 or state.shape[-1] != BCC_WEYL_SPIN_DIM:
        raise ValueError("Weyl state must have shape (nx, ny, nz, 2)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_dirac_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 4 or state.shape[-1] != BCC_DIRAC_SPIN_DIM:
        raise ValueError("Dirac state must have shape (nx, ny, nz, 4)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_dirac_spin_axis_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim < 4 or state.shape[3] != BCC_DIRAC_SPIN_DIM:
        raise ValueError("Dirac spin-axis state must have shape (nx, ny, nz, 4, ...)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_chirality(chirality: int) -> None:
    if chirality not in (-1, 1):
        raise ValueError("chirality must be +1 or -1")


@lru_cache(maxsize=8)
def _pauli_numpy(dtype_name: str) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    return np.asarray(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=dtype,
    )


def pauli_matrices(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return ``(tau_x, tau_y, tau_z)`` as a JAX array with shape ``(3,2,2)``."""

    dtype = np.dtype(dtype)
    return jnp.asarray(_pauli_numpy(dtype.name), dtype=dtype)


def bcc_axis_projectors(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return BCC axis projectors.

    The output has shape ``(3, 2, 2, 2)``.  Axis index ``0,1,2`` corresponds to
    ``x,y,z``.  Sign index ``0`` is ``P^-`` and sign index ``1`` is ``P^+``.
    """

    dtype = np.dtype(dtype)
    tau = pauli_matrices(dtype=dtype)
    identity = jnp.eye(BCC_WEYL_SPIN_DIM, dtype=dtype)
    minus = 0.5 * (identity[None, :, :] - tau)
    plus = 0.5 * (identity[None, :, :] + tau)
    return jnp.stack((minus, plus), axis=1)


def _projector_for_sign(projectors: jnp.ndarray, axis: int, sign: int) -> jnp.ndarray:
    return projectors[axis, 1 if sign > 0 else 0]


def bcc_hop_matrices(*, dtype: Any = DEFAULT_COMPLEX_DTYPE, chirality: int = 1) -> jnp.ndarray:
    """Return the eight BCC hop matrices ``A_h`` with shape ``(8,2,2)``."""

    _validate_chirality(chirality)
    dtype = np.dtype(dtype)
    projectors = bcc_axis_projectors(dtype=dtype)
    hops = []
    for sx, sy, sz in BCC_DISPLACEMENTS:
        hop = (
            _projector_for_sign(projectors, 0, chirality * sx)
            @ _projector_for_sign(projectors, 1, chirality * sy)
            @ _projector_for_sign(projectors, 2, chirality * sz)
        )
        hops.append(hop)
    return jnp.stack(hops, axis=0)


def bcc_hop_completeness_residual(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return ``max_abs(sum_h A_h^dagger A_h - I)``."""

    dtype = np.dtype(dtype)
    hops = bcc_hop_matrices(dtype=dtype)
    gram = jnp.einsum("hab,hac->bc", jnp.conj(hops), hops)
    identity = jnp.eye(BCC_WEYL_SPIN_DIM, dtype=dtype)
    return jnp.max(jnp.abs(gram - identity))


def bcc_split_symbol(k: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE, chirality: int = 1) -> jnp.ndarray:
    """Return the free BCC Weyl Bloch symbol ``S_x S_y S_z`` at momentum ``k``."""

    _validate_chirality(chirality)
    dtype = np.dtype(dtype)
    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    k_vec = jnp.asarray(k, dtype=real_dtype)
    if k_vec.shape != (3,):
        raise ValueError("k must have shape (3,)")

    projectors = bcc_axis_projectors(dtype=dtype)
    factors = []
    for axis in range(3):
        phase_plus = jnp.exp(1j * chirality * k_vec[axis]).astype(dtype)
        phase_minus = jnp.exp(-1j * chirality * k_vec[axis]).astype(dtype)
        factors.append(projectors[axis, 1] * phase_plus + projectors[axis, 0] * phase_minus)
    return factors[0] @ factors[1] @ factors[2]


def bcc_symbol_from_hops(k: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE, chirality: int = 1) -> jnp.ndarray:
    """Return the Bloch symbol by summing the eight position-space hops."""

    _validate_chirality(chirality)
    dtype = np.dtype(dtype)
    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    k_vec = jnp.asarray(k, dtype=real_dtype)
    if k_vec.shape != (3,):
        raise ValueError("k must have shape (3,)")

    hops = bcc_hop_matrices(dtype=dtype, chirality=chirality)
    symbol = jnp.zeros((BCC_WEYL_SPIN_DIM, BCC_WEYL_SPIN_DIM), dtype=dtype)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        phase = jnp.exp(1j * jnp.dot(k_vec, jnp.asarray(displacement, dtype=real_dtype))).astype(dtype)
        symbol = symbol + phase * hops[index]
    return symbol


def bcc_symbol_unitarity_residual(
    k: Any,
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
    chirality: int = 1,
) -> jnp.ndarray:
    """Return ``max_abs(U(k)^dagger U(k)-I)`` for the Bloch symbol."""

    dtype = np.dtype(dtype)
    symbol = bcc_split_symbol(k, dtype=dtype, chirality=chirality)
    identity = jnp.eye(BCC_WEYL_SPIN_DIM, dtype=dtype)
    return jnp.max(jnp.abs(jnp.conj(symbol.T) @ symbol - identity))


def bcc_weyl_step(state: jnp.ndarray, *, chirality: int = 1) -> jnp.ndarray:
    """Apply one free bulk BCC Weyl step to a periodic two-spinor field."""

    _validate_chirality(chirality)
    _validate_weyl_state(state)
    hops = bcc_hop_matrices(dtype=state.dtype, chirality=chirality)
    updated = jnp.zeros_like(state)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement)
        updated = updated + jnp.einsum("ab,...b->...a", hops[index], source)
    return updated


def bcc_norm_drift(state: jnp.ndarray) -> jnp.ndarray:
    """Return squared-norm drift after one free BCC Weyl step."""

    return norm_drift(state, bcc_weyl_step(state))


def bcc_weyl_phase_speed(k: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE, chirality: int = 1) -> jnp.ndarray:
    """Return the mean absolute eigenphase divided by ``|k|``."""

    _validate_chirality(chirality)
    real_dtype = jnp.real(jnp.asarray(0, dtype=np.dtype(dtype))).dtype
    k_vec = jnp.asarray(k, dtype=real_dtype)
    if k_vec.shape != (3,):
        raise ValueError("k must have shape (3,)")
    momentum_norm = jnp.linalg.norm(k_vec)
    symbol = bcc_split_symbol(k_vec, dtype=dtype, chirality=chirality)
    phases = jnp.angle(jnp.linalg.eigvals(symbol))
    return jnp.mean(jnp.abs(phases)) / momentum_norm


def bcc_directional_phase_speeds(
    magnitude: float,
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
    chirality: int = 1,
) -> jnp.ndarray:
    """Return phase speeds over a fixed set of directions at ``|k|=magnitude``."""

    _validate_chirality(chirality)
    real_dtype = jnp.real(jnp.asarray(0, dtype=np.dtype(dtype))).dtype
    directions = jnp.asarray(BCC_ANISOTROPY_DIRECTIONS, dtype=real_dtype)
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    speeds = [
        bcc_weyl_phase_speed(magnitude * directions[index], dtype=dtype, chirality=chirality)
        for index in range(len(BCC_ANISOTROPY_DIRECTIONS))
    ]
    return jnp.stack(speeds)


def bcc_anisotropy_spread(
    magnitude: float,
    *,
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
    chirality: int = 1,
) -> jnp.ndarray:
    """Return max-min directional phase-speed spread at fixed ``|k|``."""

    speeds = bcc_directional_phase_speeds(magnitude, dtype=dtype, chirality=chirality)
    return jnp.max(speeds) - jnp.min(speeds)


def bcc_anisotropy_scaling(
    *,
    magnitudes: tuple[float, ...] = (0.08, 0.04, 0.02),
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return directional spreads and successive ratios under halving ``|k|``."""

    spreads = jnp.stack([bcc_anisotropy_spread(magnitude, dtype=dtype) for magnitude in magnitudes])
    ratios = spreads[1:] / spreads[:-1]
    return spreads, ratios


def small_k_weyl_speed_error(
    *,
    k: tuple[float, float, float] = (1.0e-3, 2.0e-3, -1.5e-3),
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return the small-momentum phase-speed error against ``|k|``.

    The split product has eigenphases ``+-|k| + O(|k|^2)`` near the origin.
    This diagnostic reports the maximum absolute error in ``abs(phase)``.
    """

    real_dtype = jnp.real(jnp.asarray(0, dtype=np.dtype(dtype))).dtype
    k_vec = jnp.asarray(k, dtype=real_dtype)
    symbol = bcc_split_symbol(k_vec, dtype=dtype)
    phases = jnp.angle(jnp.linalg.eigvals(symbol))
    expected = jnp.linalg.norm(k_vec)
    return jnp.max(jnp.abs(jnp.abs(phases) - expected))


def _block_diag_2x2(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    zeros = jnp.zeros_like(left)
    return jnp.concatenate(
        (
            jnp.concatenate((left, zeros), axis=1),
            jnp.concatenate((zeros, right), axis=1),
        ),
        axis=0,
    )


def bcc_dirac_hop_matrices(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return block-diagonal massless Dirac hop matrices with shape ``(8,4,4)``."""

    left = bcc_hop_matrices(dtype=dtype, chirality=1)
    right = bcc_hop_matrices(dtype=dtype, chirality=-1)
    return jnp.stack([_block_diag_2x2(left[index], right[index]) for index in range(len(BCC_DISPLACEMENTS))])


def _dirac_axis_projector(projectors: jnp.ndarray, axis: int, sign: int) -> jnp.ndarray:
    left = _projector_for_sign(projectors, axis, sign)
    right = _projector_for_sign(projectors, axis, -sign)
    return _block_diag_2x2(left, right)


def bcc_dirac_axis_projectors(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return chirality-blocked Dirac axis projectors with shape ``(3,2,4,4)``."""

    dtype = np.dtype(dtype)
    projectors = bcc_axis_projectors(dtype=dtype)
    return jnp.stack(
        [
            jnp.stack(
                (
                    _dirac_axis_projector(projectors, axis, -1),
                    _dirac_axis_projector(projectors, axis, 1),
                ),
                axis=0,
            )
            for axis in range(3)
        ],
        axis=0,
    )


def bcc_dirac_hop_completeness_residual(*, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return ``max_abs(sum_h D_h^dagger D_h - I_4)`` for Dirac hops."""

    dtype = np.dtype(dtype)
    hops = bcc_dirac_hop_matrices(dtype=dtype)
    gram = jnp.einsum("hab,hac->bc", jnp.conj(hops), hops)
    identity = jnp.eye(BCC_DIRAC_SPIN_DIM, dtype=dtype)
    return jnp.max(jnp.abs(gram - identity))


def bcc_dirac_split_symbol(k: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return the massless Dirac symbol assembled from opposite Weyl chiralities."""

    left = bcc_split_symbol(k, dtype=dtype, chirality=1)
    right = bcc_split_symbol(k, dtype=dtype, chirality=-1)
    return _block_diag_2x2(left, right)


def bcc_dirac_symbol_unitarity_residual(k: Any, *, dtype: Any = DEFAULT_COMPLEX_DTYPE) -> jnp.ndarray:
    """Return ``max_abs(D(k)^dagger D(k)-I_4)`` for the massless Dirac symbol."""

    dtype = np.dtype(dtype)
    symbol = bcc_dirac_split_symbol(k, dtype=dtype)
    identity = jnp.eye(BCC_DIRAC_SPIN_DIM, dtype=dtype)
    return jnp.max(jnp.abs(jnp.conj(symbol.T) @ symbol - identity))


def bcc_dirac_spin_axis_step(state: jnp.ndarray) -> jnp.ndarray:
    """Apply the free Dirac BCC step with spin axis 3 and spectator axes."""

    lattice_shape = _validate_dirac_spin_axis_state(state)
    tail_size = int(np.prod(state.shape[4:], dtype=np.int64))
    hops = bcc_dirac_hop_matrices(dtype=state.dtype)
    out = jnp.zeros((*lattice_shape, BCC_DIRAC_SPIN_DIM, tail_size), dtype=state.dtype)
    for index, displacement in enumerate(BCC_DISPLACEMENTS):
        source = source_roll(state, displacement).reshape((*lattice_shape, BCC_DIRAC_SPIN_DIM, tail_size))
        out = out + jnp.einsum("rs,...sf->...rf", hops[index], source)
    return out.reshape(state.shape)


def _axis_displacement(axis: int, sign: int) -> tuple[int, int, int]:
    return tuple(sign if index == axis else 0 for index in range(3))  # type: ignore[return-value]


def _bcc_dirac_split_axis_step(state: jnp.ndarray, axis: int) -> jnp.ndarray:
    _validate_dirac_spin_axis_state(state)
    minus = source_roll(state, _axis_displacement(axis, -1))
    plus = source_roll(state, _axis_displacement(axis, 1))
    real_dtype = jnp.real(jnp.asarray(0, dtype=state.dtype)).dtype
    half = jnp.asarray(0.5, dtype=real_dtype)
    one_j = jnp.asarray(1j, dtype=state.dtype)

    def spin_component(local_state: jnp.ndarray, spin: int) -> jnp.ndarray:
        return jnp.take(local_state, spin, axis=3)

    if axis == 2:
        return jnp.stack(
            (
                spin_component(plus, 0),
                spin_component(minus, 1),
                spin_component(minus, 2),
                spin_component(plus, 3),
            ),
            axis=3,
        )
    if axis == 1:
        m0, m1, m2, m3 = (spin_component(minus, spin) for spin in range(BCC_DIRAC_SPIN_DIM))
        p0, p1, p2, p3 = (spin_component(plus, spin) for spin in range(BCC_DIRAC_SPIN_DIM))
        return half * jnp.stack(
            (
                m0 + one_j * m1 + p0 - one_j * p1,
                -one_j * m0 + m1 + one_j * p0 + p1,
                m2 - one_j * m3 + p2 + one_j * p3,
                one_j * m2 + m3 - one_j * p2 + p3,
            ),
            axis=3,
        )
    if axis == 0:
        m0, m1, m2, m3 = (spin_component(minus, spin) for spin in range(BCC_DIRAC_SPIN_DIM))
        p0, p1, p2, p3 = (spin_component(plus, spin) for spin in range(BCC_DIRAC_SPIN_DIM))
        return half * jnp.stack(
            (
                m0 - m1 + p0 + p1,
                -m0 + m1 + p0 + p1,
                m2 + m3 + p2 - p3,
                m2 + m3 - p2 + p3,
            ),
            axis=3,
        )
    raise ValueError("axis must be 0, 1, or 2")


def bcc_dirac_split_axis_step(state: jnp.ndarray) -> jnp.ndarray:
    """Apply the free Dirac BCC split-step product with spectator axes."""

    _validate_dirac_spin_axis_state(state)
    stepped = state
    for axis in (2, 1, 0):
        stepped = _bcc_dirac_split_axis_step(stepped, axis)
    return stepped


def bcc_dirac_step(state: jnp.ndarray) -> jnp.ndarray:
    """Apply the free massless Dirac BCC step to ``(nx,ny,nz,4)`` states."""

    _validate_dirac_state(state)
    return bcc_dirac_spin_axis_step(state)


def bcc_dirac_norm_drift(state: jnp.ndarray) -> jnp.ndarray:
    """Return squared-norm drift after one free massless Dirac step."""

    return norm_drift(state, bcc_dirac_step(state))


def small_k_dirac_speed_error(
    *,
    k: tuple[float, float, float] = (1.0e-3, 2.0e-3, -1.5e-3),
    dtype: Any = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return the small-momentum Dirac eigenphase-speed error against ``|k|``."""

    real_dtype = jnp.real(jnp.asarray(0, dtype=np.dtype(dtype))).dtype
    k_vec = jnp.asarray(k, dtype=real_dtype)
    symbol = bcc_dirac_split_symbol(k_vec, dtype=dtype)
    phases = jnp.angle(jnp.linalg.eigvals(symbol))
    expected = jnp.linalg.norm(k_vec)
    return jnp.max(jnp.abs(jnp.abs(phases) - expected))


def bare_bcc_walk_diagnostics(state: jnp.ndarray) -> BareBCCWalkDiagnostics:
    """Return focused diagnostics for one free BCC Weyl step."""

    _validate_weyl_state(state)
    dtype = state.dtype
    k = jnp.asarray((0.17, -0.11, 0.07), dtype=jnp.real(jnp.asarray(0, dtype=dtype)).dtype)
    spreads, ratios = bcc_anisotropy_scaling(dtype=dtype)
    dirac_state = jnp.concatenate((state, state), axis=-1)
    return BareBCCWalkDiagnostics(
        hop_count=len(BCC_DISPLACEMENTS),
        hop_completeness_residual=bcc_hop_completeness_residual(dtype=dtype),
        symbol_unitarity_residual=bcc_symbol_unitarity_residual(k, dtype=dtype),
        norm_drift=bcc_norm_drift(state),
        small_k_speed_error=small_k_weyl_speed_error(dtype=dtype),
        anisotropy_spreads=spreads,
        anisotropy_halving_ratios=ratios,
        dirac_hop_completeness_residual=bcc_dirac_hop_completeness_residual(dtype=dtype),
        dirac_symbol_unitarity_residual=bcc_dirac_symbol_unitarity_residual(k, dtype=dtype),
        dirac_norm_drift=bcc_dirac_norm_drift(dirac_state),
        dirac_small_k_speed_error=small_k_dirac_speed_error(dtype=dtype),
    )


def jitted_bcc_weyl_step():
    """Return a JIT-compiled free BCC Weyl step function."""

    return jax.jit(bcc_weyl_step)


def jitted_bcc_dirac_step():
    """Return a JIT-compiled free massless Dirac BCC step function."""

    return jax.jit(bcc_dirac_step)
