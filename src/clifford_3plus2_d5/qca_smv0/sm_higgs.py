"""Local Higgs/Yukawa collision for QCA_SMv0.

Stage 4 adds a site-local, exactly unitary Higgs/Yukawa collision.  The Higgs
field is a local electroweak doublet, but this stage does not yet add Higgs
field dynamics, matter backreaction, flavor, CKM/PMNS, or FN recirculation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_split_symbol
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_CHIRAL16_DIM, SM_INTERNAL_DIM, deterministic_sm_state
from clifford_3plus2_d5.sim.state import norm_drift

SM_HIGGS_DIM = 2


class SMYukawaCouplings(NamedTuple):
    """One-generation Yukawa couplings for the Stage 4 local collision."""

    up: float = 1.0
    down: float = 0.8
    electron: float = 0.6
    neutrino: float = 0.4


class HiggsYukawaDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 4 local Higgs/Yukawa collision."""

    yukawa_hermitian_residual: jnp.ndarray
    zero_step_residual: jnp.ndarray
    zero_higgs_residual: jnp.ndarray
    norm_drift: jnp.ndarray
    chirality_flip_right_norm: jnp.ndarray
    massive_dispersion_residual: jnp.ndarray
    jit_delta: jnp.ndarray


DEFAULT_YUKAWA_COUPLINGS = SMYukawaCouplings()


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


def _validate_higgs_field(higgs: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    if higgs.ndim != 4 or higgs.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and higgs.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the state lattice")
    return int(higgs.shape[0]), int(higgs.shape[1]), int(higgs.shape[2])


def _validate_sm_dirac_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 5 or state.shape[-2:] != (4, SM_INTERNAL_DIM):
        raise ValueError("SM Dirac state must have shape (nx, ny, nz, 4, 32)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def sm_constant_higgs(lattice_shape: tuple[int, int, int], *, vev: float = 1.0) -> jnp.ndarray:
    """Return a constant unitary-gauge Higgs field ``H=(0,vev/sqrt(2))``."""

    higgs = jnp.zeros((*lattice_shape, SM_HIGGS_DIM), dtype=jnp.complex64)
    return higgs.at[..., 1].set(jnp.asarray(vev / jnp.sqrt(2.0), dtype=jnp.complex64))


def sm_higgs_tilde(higgs: jnp.ndarray) -> jnp.ndarray:
    """Return ``i sigma_2 H^*`` for Higgs doublet field ``H``."""

    _validate_higgs_field(higgs)
    return jnp.stack((jnp.conj(higgs[..., 1]), -jnp.conj(higgs[..., 0])), axis=-1)


def _add_hermitian_pair(matrix: jnp.ndarray, left: int, right: int, amplitude: jnp.ndarray) -> jnp.ndarray:
    matrix = matrix.at[..., left, right].add(amplitude)
    return matrix.at[..., right, left].add(jnp.conj(amplitude))


def _chiral16_yukawa_matrix(
    higgs: jnp.ndarray,
    couplings: SMYukawaCouplings,
) -> jnp.ndarray:
    _validate_higgs_field(higgs)
    dtype = jnp.result_type(higgs, 1j)
    matrix = jnp.zeros((*higgs.shape[:3], SM_CHIRAL16_DIM, SM_CHIRAL16_DIM), dtype=dtype)
    h_tilde = sm_higgs_tilde(higgs)
    up = jnp.asarray(couplings.up, dtype=dtype)
    down = jnp.asarray(couplings.down, dtype=dtype)
    electron = jnp.asarray(couplings.electron, dtype=dtype)
    neutrino = jnp.asarray(couplings.neutrino, dtype=dtype)

    for color in range(3):
        for weak in range(2):
            matrix = _add_hermitian_pair(
                matrix,
                _q_index(color, weak),
                _u_c_index(color),
                up * h_tilde[..., weak],
            )
            matrix = _add_hermitian_pair(
                matrix,
                _q_index(color, weak),
                _d_c_index(color),
                down * higgs[..., weak],
            )

    for weak in range(2):
        matrix = _add_hermitian_pair(
            matrix,
            _l_index(weak),
            _nu_c_index(),
            neutrino * h_tilde[..., weak],
        )
        matrix = _add_hermitian_pair(
            matrix,
            _l_index(weak),
            _e_c_index(),
            electron * higgs[..., weak],
        )
    return matrix


def sm_yukawa_internal_matrix(
    higgs: jnp.ndarray,
    *,
    couplings: SMYukawaCouplings = DEFAULT_YUKAWA_COUPLINGS,
) -> jnp.ndarray:
    """Return a Hermitian local internal Yukawa matrix with shape ``(...,32,32)``."""

    chiral16 = _chiral16_yukawa_matrix(higgs, couplings)
    matrix = jnp.zeros((*higgs.shape[:3], SM_INTERNAL_DIM, SM_INTERNAL_DIM), dtype=chiral16.dtype)
    matrix = matrix.at[..., :SM_CHIRAL16_DIM, :SM_CHIRAL16_DIM].set(chiral16)
    return matrix.at[..., SM_CHIRAL16_DIM:, SM_CHIRAL16_DIM:].set(chiral16)


def sm_yukawa_hermitian_residual(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return ``max_abs(Y-Y^dagger)`` for a local Yukawa matrix field."""

    return jnp.max(jnp.abs(matrix - jnp.swapaxes(jnp.conj(matrix), -1, -2)))


def sm_dirac_beta(dtype: jnp.dtype = jnp.complex64) -> jnp.ndarray:
    """Return the Dirac beta matrix that swaps left/right Weyl spin blocks."""

    zero = jnp.zeros((2, 2), dtype=dtype)
    identity = jnp.eye(2, dtype=dtype)
    top = jnp.concatenate((zero, identity), axis=1)
    bottom = jnp.concatenate((identity, zero), axis=1)
    return jnp.concatenate((top, bottom), axis=0)


def _hermitian_function(matrix: jnp.ndarray, fn) -> jnp.ndarray:
    flat = matrix.reshape((-1, matrix.shape[-2], matrix.shape[-1]))

    def apply_fn(local: jnp.ndarray) -> jnp.ndarray:
        values, vectors = jnp.linalg.eigh(local)
        transformed = vectors * fn(values)[None, :]
        return transformed @ jnp.swapaxes(jnp.conj(vectors), -1, -2)

    return jax.vmap(apply_fn)(flat).reshape(matrix.shape)


def _yukawa_cos_sin(yukawa: jnp.ndarray, step_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.asarray(step_size, dtype=jnp.real(jnp.asarray(0, dtype=yukawa.dtype)).dtype)
    return _hermitian_function(yukawa, lambda values: jnp.cos(scale * values)), _hermitian_function(
        yukawa,
        lambda values: jnp.sin(scale * values),
    )


def sm_apply_yukawa_collision(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    step_size: float,
    couplings: SMYukawaCouplings = DEFAULT_YUKAWA_COUPLINGS,
) -> jnp.ndarray:
    """Apply exact local unitary ``exp(-i step_size beta Y(H))``."""

    lattice_shape = _validate_sm_dirac_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if step_size == 0:
        return state

    yukawa = sm_yukawa_internal_matrix(higgs, couplings=couplings).astype(state.dtype)
    cos_internal, sin_internal = _yukawa_cos_sin(yukawa, step_size)
    cos_action = jnp.einsum("...ij,...sj->...si", cos_internal, state)
    sin_internal_action = jnp.einsum("...ij,...sj->...si", sin_internal, state)
    beta_action = jnp.einsum("sr,...ri->...si", sm_dirac_beta(state.dtype), sin_internal_action)
    return cos_action - 1j * beta_action


def sm_chirality_norms(state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return squared norm in the two Weyl spin blocks of the Dirac register."""

    _validate_sm_dirac_state(state)
    left = jnp.real(jnp.sum(jnp.conj(state[..., :2, :]) * state[..., :2, :]))
    right = jnp.real(jnp.sum(jnp.conj(state[..., 2:, :]) * state[..., 2:, :]))
    return left, right


def sm_massive_yukawa_symbol(k: tuple[float, float, float], *, mass_angle: float) -> jnp.ndarray:
    """Return the spin-space constant-mass split symbol for dispersion audits."""

    beta = sm_dirac_beta(jnp.complex64)
    mass_collision = jnp.cos(jnp.asarray(mass_angle)) * jnp.eye(4, dtype=jnp.complex64) - 1j * jnp.sin(
        jnp.asarray(mass_angle),
    ) * beta
    return mass_collision @ bcc_dirac_split_symbol(k)


def sm_massive_positive_phase(k: tuple[float, float, float], *, mass_angle: float) -> jnp.ndarray:
    """Return the smallest positive quasienergy of the constant-mass symbol."""

    phases = jnp.sort(jnp.abs(jnp.angle(jnp.linalg.eigvals(sm_massive_yukawa_symbol(k, mass_angle=mass_angle)))))
    return phases[0]


def sm_massive_dispersion_residual(
    *,
    k: tuple[float, float, float] = (5e-3, 0.0, 0.0),
    mass_angle: float = 0.03,
) -> jnp.ndarray:
    """Return ``|E(k)-sqrt(|k|^2+m^2)|`` for the massive Stage 4 symbol."""

    phase = sm_massive_positive_phase(k, mass_angle=mass_angle)
    k_norm = jnp.linalg.norm(jnp.asarray(k, dtype=jnp.float32))
    target = jnp.sqrt(k_norm * k_norm + jnp.asarray(mass_angle, dtype=jnp.float32) ** 2)
    return jnp.abs(phase - target)


def higgs_yukawa_diagnostics() -> HiggsYukawaDiagnostics:
    """Return focused Stage 4 Higgs/Yukawa collision diagnostics."""

    lattice_shape = (1, 1, 1)
    state = deterministic_sm_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_higgs = jnp.zeros_like(higgs)
    yukawa = sm_yukawa_internal_matrix(higgs)
    step_size = 0.07
    updated = sm_apply_yukawa_collision(state, higgs, step_size=step_size)
    left_seed = jnp.zeros_like(state)
    left_seed = left_seed.at[0, 0, 0, 0, _q_index(0, 0)].set(1.0 + 0.0j)
    flipped = sm_apply_yukawa_collision(left_seed, higgs, step_size=step_size)
    _, right_norm = sm_chirality_norms(flipped)
    jitted_collision = jax.jit(sm_apply_yukawa_collision, static_argnames=("step_size",))

    return HiggsYukawaDiagnostics(
        yukawa_hermitian_residual=sm_yukawa_hermitian_residual(yukawa),
        zero_step_residual=jnp.max(jnp.abs(sm_apply_yukawa_collision(state, higgs, step_size=0.0) - state)),
        zero_higgs_residual=jnp.max(jnp.abs(sm_apply_yukawa_collision(state, zero_higgs, step_size=step_size) - state)),
        norm_drift=norm_drift(state, updated),
        chirality_flip_right_norm=right_norm,
        massive_dispersion_residual=sm_massive_dispersion_residual(),
        jit_delta=jnp.max(jnp.abs(jitted_collision(state, higgs, step_size=step_size) - updated)),
    )
