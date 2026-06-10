"""Three-family Higgs/Yukawa collision for QCA_SMv0.

Stage 7 embeds the Stage 5/6 generated flavor matrices into the local
Higgs/Yukawa collision.  The state layout is
``(nx, ny, nz, 4, 32, 3)``: Dirac spin, one SM chiral-16 carrier duplicated as
in Stage 2, and an explicit three-family register.  The collision flattens the
``32 x 3`` internal/family register, applies the exact unitary
``exp(-i step_size beta Y(H))``, then reshapes back.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_cp import (
    DEFAULT_CENTER_HOLONOMY_POWERS,
    CenterHolonomyPowers,
    sm_center_coefficients,
    sm_center_cp_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
    FNQuarkCharges,
    FNQuarkYukawas,
    SM_FAMILY_DIM,
    fn_ckm_from_yukawas,
    fn_visible_recirculation_transfer,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import SM_CHIRAL16_DIM, SM_INTERNAL_DIM, deterministic_sm_state
from clifford_3plus2_d5.qca_smv0.sm_higgs import (
    SM_HIGGS_DIM,
    sm_constant_higgs,
    sm_dirac_beta,
    sm_higgs_tilde,
    sm_yukawa_hermitian_residual,
)
from clifford_3plus2_d5.sim.state import norm_drift

SM_FAMILY_INTERNAL_DIM = SM_INTERNAL_DIM * SM_FAMILY_DIM


class FamilyLeptonYukawas(NamedTuple):
    """Explicit simulator lepton matrices for the three-family collision."""

    neutrino: jnp.ndarray
    electron: jnp.ndarray


class FamilyHiggsYukawaDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 7 three-family Higgs/Yukawa collision."""

    family_yukawa_hermitian_residual: jnp.ndarray
    fn_recirculated_quark_yukawa_residual: jnp.ndarray
    fn_recirculated_embedding_residual: jnp.ndarray
    quark_embedding_residual: jnp.ndarray
    wrong_door_residual: jnp.ndarray
    ckm_embedding_residual: jnp.ndarray
    zero_step_residual: jnp.ndarray
    zero_higgs_residual: jnp.ndarray
    norm_drift: jnp.ndarray
    chirality_flip_right_norm: jnp.ndarray
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


def _family_flat_index(internal_index: int, family: int) -> int:
    return internal_index * SM_FAMILY_DIM + family


def _validate_higgs_field(higgs: jnp.ndarray, lattice_shape: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    if higgs.ndim != 4 or higgs.shape[-1] != SM_HIGGS_DIM:
        raise ValueError("Higgs field must have shape (nx, ny, nz, 2)")
    if lattice_shape is not None and higgs.shape[:3] != lattice_shape:
        raise ValueError("Higgs field must match the state lattice")
    return int(higgs.shape[0]), int(higgs.shape[1]), int(higgs.shape[2])


def _validate_family_state(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family SM Dirac state must have shape (nx, ny, nz, 4, 32, 3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def _validate_family_matrix(matrix: jnp.ndarray, name: str) -> jnp.ndarray:
    arr = jnp.asarray(matrix)
    if arr.shape != (SM_FAMILY_DIM, SM_FAMILY_DIM):
        raise ValueError(f"{name} must have shape (3,3)")
    return arr.astype(jnp.complex64)


def sm_default_family_lepton_yukawas() -> FamilyLeptonYukawas:
    """Return diagonal placeholder lepton matrices for the simulator collision."""

    return FamilyLeptonYukawas(
        neutrino=jnp.diag(jnp.asarray([0.05, 0.10, 0.20], dtype=jnp.complex64)),
        electron=jnp.diag(jnp.asarray([0.02, 0.10, 0.50], dtype=jnp.complex64)),
    )


def sm_family_recirculated_quark_yukawas(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    powers: CenterHolonomyPowers = DEFAULT_CENTER_HOLONOMY_POWERS,
) -> FNQuarkYukawas:
    """Return the quark matrices consumed by the family Higgs collision.

    The source is the simulator FN rule itself: center-holonomy order-one
    coefficients enter the visible incidence/readout maps, while the powers are
    measured from the explicit hidden recirculation paths.
    """

    up_coeffs = sm_center_coefficients("up", powers=powers.up)
    down_coeffs = sm_center_coefficients("down", powers=powers.down)
    return FNQuarkYukawas(
        up=fn_visible_recirculation_transfer(lambda_rec, charges.q, charges.u, coefficients=up_coeffs),
        down=fn_visible_recirculation_transfer(lambda_rec, charges.q, charges.d, coefficients=down_coeffs),
    )


def deterministic_sm_family_state(lattice_shape: tuple[int, int, int]) -> jnp.ndarray:
    """Return a deterministic family-extended SM Dirac state."""

    base = deterministic_sm_state(lattice_shape)
    weights = jnp.asarray([1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.75j], dtype=base.dtype)
    weights = weights / jnp.sqrt(jnp.sum(jnp.abs(weights) ** 2))
    return base[..., None] * weights


def _add_family_hermitian_pair(
    matrix: jnp.ndarray,
    left_internal: int,
    right_internal: int,
    amplitude_matrix: jnp.ndarray,
) -> jnp.ndarray:
    for left_family in range(SM_FAMILY_DIM):
        for right_family in range(SM_FAMILY_DIM):
            left = _family_flat_index(left_internal, left_family)
            right = _family_flat_index(right_internal, right_family)
            amplitude = amplitude_matrix[..., left_family, right_family]
            matrix = matrix.at[..., left, right].add(amplitude)
            matrix = matrix.at[..., right, left].add(jnp.conj(amplitude))
    return matrix


def sm_family_yukawa_internal_matrix(
    higgs: jnp.ndarray,
    *,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Return the local Hermitian three-family Yukawa matrix ``(...,96,96)``."""

    _validate_higgs_field(higgs)
    if quark_yukawas is None:
        quark_yukawas = sm_family_recirculated_quark_yukawas()
    if lepton_yukawas is None:
        lepton_yukawas = sm_default_family_lepton_yukawas()

    up = _validate_family_matrix(quark_yukawas.up, "quark_yukawas.up")
    down = _validate_family_matrix(quark_yukawas.down, "quark_yukawas.down")
    neutrino = _validate_family_matrix(lepton_yukawas.neutrino, "lepton_yukawas.neutrino")
    electron = _validate_family_matrix(lepton_yukawas.electron, "lepton_yukawas.electron")
    h_tilde = sm_higgs_tilde(higgs)
    dtype = jnp.result_type(higgs, up, down, neutrino, electron, 1j)
    matrix = jnp.zeros((*higgs.shape[:3], SM_FAMILY_INTERNAL_DIM, SM_FAMILY_INTERNAL_DIM), dtype=dtype)

    for copy_offset in (0, SM_CHIRAL16_DIM):
        for color in range(3):
            for weak in range(2):
                matrix = _add_family_hermitian_pair(
                    matrix,
                    copy_offset + _q_index(color, weak),
                    copy_offset + _u_c_index(color),
                    up * h_tilde[..., weak, None, None],
                )
                matrix = _add_family_hermitian_pair(
                    matrix,
                    copy_offset + _q_index(color, weak),
                    copy_offset + _d_c_index(color),
                    down * higgs[..., weak, None, None],
                )

        for weak in range(2):
            matrix = _add_family_hermitian_pair(
                matrix,
                copy_offset + _l_index(weak),
                copy_offset + _nu_c_index(),
                neutrino * h_tilde[..., weak, None, None],
            )
            matrix = _add_family_hermitian_pair(
                matrix,
                copy_offset + _l_index(weak),
                copy_offset + _e_c_index(),
                electron * higgs[..., weak, None, None],
            )
    return matrix


def sm_family_extract_block(matrix: jnp.ndarray, left_internal: int, right_internal: int) -> jnp.ndarray:
    """Extract a ``3 x 3`` family block between two internal labels."""

    left_indices = jnp.asarray([_family_flat_index(left_internal, family) for family in range(SM_FAMILY_DIM)])
    right_indices = jnp.asarray([_family_flat_index(right_internal, family) for family in range(SM_FAMILY_DIM)])
    return matrix[..., left_indices[:, None], right_indices[None, :]]


def _hermitian_function(matrix: jnp.ndarray, fn) -> jnp.ndarray:
    flat = matrix.reshape((-1, matrix.shape[-2], matrix.shape[-1]))

    def apply_fn(local: jnp.ndarray) -> jnp.ndarray:
        values, vectors = jnp.linalg.eigh(local)
        transformed = vectors * fn(values)[None, :]
        return transformed @ jnp.swapaxes(jnp.conj(vectors), -1, -2)

    return jax.vmap(apply_fn)(flat).reshape(matrix.shape)


def _family_yukawa_cos_sin(yukawa: jnp.ndarray, step_size: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.asarray(step_size, dtype=jnp.real(jnp.asarray(0, dtype=yukawa.dtype)).dtype)
    cos_internal = _hermitian_function(yukawa, lambda values: jnp.cos(scale * values))
    sin_internal = _hermitian_function(
        yukawa,
        lambda values: jnp.sin(scale * values),
    )
    local_norm = jnp.max(jnp.abs(yukawa), axis=(-1, -2), keepdims=True)
    identity = jnp.broadcast_to(
        jnp.eye(SM_FAMILY_INTERNAL_DIM, dtype=yukawa.dtype),
        yukawa.shape,
    )
    zeros = jnp.zeros_like(yukawa)
    cos_internal = jnp.where(local_norm < 1e-12, identity, cos_internal)
    sin_internal = jnp.where(local_norm < 1e-12, zeros, sin_internal)
    return cos_internal, sin_internal


def sm_apply_family_yukawa_collision(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> jnp.ndarray:
    """Apply exact local unitary three-family Higgs/Yukawa collision."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if step_size == 0:
        return state

    flat_state = state.reshape((*lattice_shape, 4, SM_FAMILY_INTERNAL_DIM))
    yukawa = sm_family_yukawa_internal_matrix(
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    ).astype(state.dtype)
    cos_internal, sin_internal = _family_yukawa_cos_sin(yukawa, step_size)
    cos_action = jnp.einsum("...ij,...sj->...si", cos_internal, flat_state)
    sin_internal_action = jnp.einsum("...ij,...sj->...si", sin_internal, flat_state)
    beta_action = jnp.einsum("sr,...ri->...si", sm_dirac_beta(state.dtype), sin_internal_action)
    return (cos_action - 1j * beta_action).reshape(state.shape)


def sm_family_chirality_norms(state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return squared norm in the two Weyl spin blocks."""

    _validate_family_state(state)
    left = jnp.real(jnp.sum(jnp.conj(state[..., :2, :, :]) * state[..., :2, :, :]))
    right = jnp.real(jnp.sum(jnp.conj(state[..., 2:, :, :]) * state[..., 2:, :, :]))
    return left, right


def sm_family_embedding_residuals(
    higgs: jnp.ndarray,
    *,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return quark embedding, wrong-door, and CKM residuals."""

    if quark_yukawas is None:
        quark_yukawas = sm_family_recirculated_quark_yukawas()
    matrix = sm_family_yukawa_internal_matrix(higgs, quark_yukawas=quark_yukawas, lepton_yukawas=lepton_yukawas)
    h_tilde = sm_higgs_tilde(higgs)
    up_scale = h_tilde[0, 0, 0, 0]
    down_scale = higgs[0, 0, 0, 1]
    up_block = sm_family_extract_block(matrix[0, 0, 0], _q_index(0, 0), _u_c_index(0)) / up_scale
    down_block = sm_family_extract_block(matrix[0, 0, 0], _q_index(0, 1), _d_c_index(0)) / down_scale
    wrong_blocks = jnp.stack(
        [
            sm_family_extract_block(matrix[0, 0, 0], _q_index(0, 1), _u_c_index(0)),
            sm_family_extract_block(matrix[0, 0, 0], _q_index(0, 0), _d_c_index(0)),
            sm_family_extract_block(matrix[0, 0, 0], _l_index(1), _nu_c_index()),
            sm_family_extract_block(matrix[0, 0, 0], _l_index(0), _e_c_index()),
        ],
    )
    target_ckm = fn_ckm_from_yukawas(quark_yukawas.up, quark_yukawas.down)
    embedded_ckm = fn_ckm_from_yukawas(up_block, down_block)
    return (
        jnp.maximum(jnp.max(jnp.abs(up_block - quark_yukawas.up)), jnp.max(jnp.abs(down_block - quark_yukawas.down))),
        jnp.max(jnp.abs(wrong_blocks)),
        jnp.max(jnp.abs(jnp.abs(embedded_ckm) - jnp.abs(target_ckm))),
    )


def sm_family_higgs_yukawa_diagnostics() -> FamilyHiggsYukawaDiagnostics:
    """Return focused Stage 7 three-family Higgs/Yukawa diagnostics."""

    lattice_shape = (1, 1, 1)
    state = deterministic_sm_family_state(lattice_shape)
    higgs = sm_constant_higgs(lattice_shape)
    zero_higgs = jnp.zeros_like(higgs)
    quark_yukawas = sm_family_recirculated_quark_yukawas()
    reference_quark_yukawas = sm_center_cp_quark_yukawas()
    lepton_yukawas = sm_default_family_lepton_yukawas()
    matrix = sm_family_yukawa_internal_matrix(higgs, quark_yukawas=quark_yukawas, lepton_yukawas=lepton_yukawas)
    quark_embedding, wrong_door, ckm_embedding = sm_family_embedding_residuals(
        higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    recirculated_embedding, _, _ = sm_family_embedding_residuals(
        higgs,
        quark_yukawas=sm_family_recirculated_quark_yukawas(),
        lepton_yukawas=lepton_yukawas,
    )
    step_size = 0.04
    updated = sm_apply_family_yukawa_collision(
        state,
        higgs,
        step_size=step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    left_seed = jnp.zeros_like(state)
    left_seed = left_seed.at[0, 0, 0, 0, _q_index(0, 0), 0].set(1.0 + 0.0j)
    flipped = sm_apply_family_yukawa_collision(
        left_seed,
        higgs,
        step_size=step_size,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    _, right_norm = sm_family_chirality_norms(flipped)
    jitted = jax.jit(sm_apply_family_yukawa_collision, static_argnames=("step_size",))

    return FamilyHiggsYukawaDiagnostics(
        family_yukawa_hermitian_residual=sm_yukawa_hermitian_residual(matrix),
        fn_recirculated_quark_yukawa_residual=jnp.maximum(
            jnp.max(jnp.abs(quark_yukawas.up - reference_quark_yukawas.up)),
            jnp.max(jnp.abs(quark_yukawas.down - reference_quark_yukawas.down)),
        ),
        fn_recirculated_embedding_residual=recirculated_embedding,
        quark_embedding_residual=quark_embedding,
        wrong_door_residual=wrong_door,
        ckm_embedding_residual=ckm_embedding,
        zero_step_residual=jnp.max(
            jnp.abs(
                sm_apply_family_yukawa_collision(
                    state,
                    higgs,
                    step_size=0.0,
                    quark_yukawas=quark_yukawas,
                    lepton_yukawas=lepton_yukawas,
                )
                - state,
            ),
        ),
        zero_higgs_residual=jnp.max(
            jnp.abs(
                sm_apply_family_yukawa_collision(
                    state,
                    zero_higgs,
                    step_size=step_size,
                    quark_yukawas=quark_yukawas,
                    lepton_yukawas=lepton_yukawas,
                )
                - state,
            ),
        ),
        norm_drift=norm_drift(state, updated),
        chirality_flip_right_norm=right_norm,
        jit_delta=jnp.max(
            jnp.abs(
                jitted(
                    state,
                    higgs,
                    step_size=step_size,
                    quark_yukawas=quark_yukawas,
                    lepton_yukawas=lepton_yukawas,
                )
                - updated,
            ),
        ),
    )
