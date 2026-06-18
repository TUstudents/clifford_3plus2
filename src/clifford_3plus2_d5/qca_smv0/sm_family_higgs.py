"""Three-family Higgs/Yukawa collision for QCA_SMv0.

Stage 7 embeds the Stage 5/6 generated flavor matrices into the local
Higgs/Yukawa collision.  The state layout is
``(nx, ny, nz, 4, 32, 3)``: spacetime Dirac spin, the Stage 2 doubled
32-component simulator representation of one SM chiral-16 label set, and an
explicit three-family register.  The collision flattens the
``32 x 3`` internal/family register, applies the exact unitary
``exp(-i step_size beta Y(H))``, then reshapes back.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_split_symbol
from clifford_3plus2_d5.qca_smv0.sm_cp import (
    DEFAULT_CENTER_HOLONOMY_POWERS,
    CenterHolonomyPowers,
    sm_center_coefficients,
    sm_center_cp_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
    FNQuarkCoefficientMatrices,
    FNQuarkCharges,
    FNQuarkYukawas,
    FNVisibleRecirculationPairReadout,
    FNVisibleRecirculationReadout,
    FNUnitaryDilation,
    fn_apply_visible_recirculation_pair_path_state,
    fn_apply_visible_recirculation_path_state,
    SM_FAMILY_DIM,
    fn_apply_recirculation_collision,
    fn_ckm_from_yukawas,
    fn_quark_coefficients_from_yukawas,
    fn_quark_yukawas_from_masses_ckm,
    fn_prepare_visible_collision_state,
    fn_read_visible_collision_output,
    fn_recirculation_collision_dilation,
    fn_singular_masses,
    fn_unitary_dilation_residual,
    fn_visible_recirculation_pair_readout,
    fn_visible_recirculation_readout,
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


class FamilyYukawaCollisionCache(NamedTuple):
    """Precomputed exact family Yukawa collision functions."""

    cos_internal: jnp.ndarray
    sin_internal: jnp.ndarray
    block_indices: jnp.ndarray | None = None
    block_inverse_indices: jnp.ndarray | None = None
    cos_blocks: jnp.ndarray | None = None
    sin_blocks: jnp.ndarray | None = None
    internal_pair_indices: jnp.ndarray | None = None
    internal_inverse_indices: jnp.ndarray | None = None


class FamilyHiggsYukawaDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 7 three-family Higgs/Yukawa collision."""

    family_yukawa_hermitian_residual: jnp.ndarray
    fn_recirculated_quark_yukawa_residual: jnp.ndarray
    fn_recirculated_embedding_residual: jnp.ndarray
    fn_quark_dilation_unitarity_residual: jnp.ndarray
    fn_quark_door_transfer_residual: jnp.ndarray
    quark_embedding_residual: jnp.ndarray
    wrong_door_residual: jnp.ndarray
    ckm_embedding_residual: jnp.ndarray
    zero_step_residual: jnp.ndarray
    zero_higgs_residual: jnp.ndarray
    norm_drift: jnp.ndarray
    chirality_flip_right_norm: jnp.ndarray
    jit_delta: jnp.ndarray


class FamilyFNMassSpectrumProbe(NamedTuple):
    """Calibrated FN quark gaps measured from the exact family mass symbol."""

    target_yukawas: FNQuarkYukawas
    recovered_yukawas: FNQuarkYukawas
    target_up_mass_angles: jnp.ndarray
    target_down_mass_angles: jnp.ndarray
    measured_up_mass_angles: jnp.ndarray
    measured_down_mass_angles: jnp.ndarray
    up_gap_residual: jnp.ndarray
    down_gap_residual: jnp.ndarray
    ckm_abs_residual: jnp.ndarray
    up_symbol_unitarity_residual: jnp.ndarray
    down_symbol_unitarity_residual: jnp.ndarray


class FamilyFNQuarkDilations(NamedTuple):
    """Finite FN unitary dilations for the up/down quark Higgs doors."""

    up: FNUnitaryDilation
    down: FNUnitaryDilation


class FamilyFNQuarkAuxState(NamedTuple):
    """Hidden auxiliary state carried by the up/down FN quark doors."""

    up: jnp.ndarray
    down: jnp.ndarray


class FamilyFNQuarkPathReadouts(NamedTuple):
    """Explicit hidden-path FN readouts for the up/down quark Higgs doors."""

    up: FNVisibleRecirculationReadout
    down: FNVisibleRecirculationReadout
    pair: FNVisibleRecirculationPairReadout


class FamilyFNQuarkPathAuxState(NamedTuple):
    """Persistent hidden states on the explicit up/down FN path networks."""

    up: jnp.ndarray
    down: jnp.ndarray


class FamilyFNQuarkPathHiddenDims(NamedTuple):
    """Hidden dimensions for the explicit up/down FN path networks."""

    up: int
    down: int


class FamilyFNQuarkDoorOutput(NamedTuple):
    """One FN door update: raw unitary output, physical source, and aux state."""

    raw_visible: jnp.ndarray
    physical_visible: jnp.ndarray
    hidden: jnp.ndarray


class FamilyFNQuarkStateSource(NamedTuple):
    """FN quark source extracted from a full family state plus updated aux."""

    up: jnp.ndarray
    down: jnp.ndarray
    state_source: jnp.ndarray
    aux_state: FamilyFNQuarkAuxState


class FamilyFNQuarkSourceKick(NamedTuple):
    """One explicit FN source kick on the visible quark state plus aux update."""

    state: jnp.ndarray
    source: jnp.ndarray
    aux_state: FamilyFNQuarkAuxState


class FamilyFNQuarkPathDoorOutput(NamedTuple):
    """One explicit path-network FN door update."""

    raw_visible: jnp.ndarray
    physical_visible: jnp.ndarray
    hidden: jnp.ndarray


class FamilyFNQuarkPathStateSource(NamedTuple):
    """FN quark source extracted from explicit path-network memory."""

    up: jnp.ndarray
    down: jnp.ndarray
    state_remainder: jnp.ndarray
    state_source: jnp.ndarray
    aux_state: FamilyFNQuarkPathAuxState


class FamilyFNQuarkPathSourceKick(NamedTuple):
    """One explicit path-network source kick on visible quarks plus aux update."""

    state: jnp.ndarray
    source: jnp.ndarray
    aux_state: FamilyFNQuarkPathAuxState


class FamilyFNQuarkPathUnitaryCollision(NamedTuple):
    """One exact visible-hidden-return FN collision on quarks."""

    state: jnp.ndarray
    aux_state: FamilyFNQuarkPathAuxState


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


def _sm_family_quark_coefficients(
    powers: CenterHolonomyPowers,
    coefficients: FNQuarkCoefficientMatrices | None,
) -> FNQuarkCoefficientMatrices:
    if coefficients is not None:
        return FNQuarkCoefficientMatrices(
            up=_validate_family_matrix(coefficients.up, "coefficients.up"),
            down=_validate_family_matrix(coefficients.down, "coefficients.down"),
        )
    return FNQuarkCoefficientMatrices(
        up=sm_center_coefficients("up", powers=powers.up),
        down=sm_center_coefficients("down", powers=powers.down),
    )


def _fn_recirculation_hidden_dim(left_charges: tuple[int, int, int], right_charges: tuple[int, int, int]) -> int:
    return sum(int(left) + int(right) + 1 for left in left_charges for right in right_charges)


def sm_family_quark_path_hidden_dims(
    *,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
) -> FamilyFNQuarkPathHiddenDims:
    """Return exact hidden dimensions without constructing FN path unitaries."""

    return FamilyFNQuarkPathHiddenDims(
        up=_fn_recirculation_hidden_dim(charges.q, charges.u),
        down=_fn_recirculation_hidden_dim(charges.q, charges.d),
    )


def sm_family_recirculated_quark_yukawas(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    powers: CenterHolonomyPowers = DEFAULT_CENTER_HOLONOMY_POWERS,
    coefficients: FNQuarkCoefficientMatrices | None = None,
) -> FNQuarkYukawas:
    """Return the quark matrices consumed by the family Higgs collision.

    The source is the simulator FN rule itself: center-holonomy order-one
    coefficients enter the visible incidence/readout maps, while the powers are
    measured from the explicit hidden recirculation paths.
    """

    coeffs = _sm_family_quark_coefficients(powers, coefficients)
    return FNQuarkYukawas(
        up=fn_visible_recirculation_transfer(lambda_rec, charges.q, charges.u, coefficients=coeffs.up),
        down=fn_visible_recirculation_transfer(lambda_rec, charges.q, charges.d, coefficients=coeffs.down),
    )


def sm_family_recirculated_quark_dilations(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    powers: CenterHolonomyPowers = DEFAULT_CENTER_HOLONOMY_POWERS,
    coefficients: FNQuarkCoefficientMatrices | None = None,
) -> FamilyFNQuarkDilations:
    """Return finite unitary FN collision dilations for the quark doors."""

    coeffs = _sm_family_quark_coefficients(powers, coefficients)
    return FamilyFNQuarkDilations(
        up=fn_recirculation_collision_dilation(lambda_rec, charges.q, charges.u, coefficients=coeffs.up),
        down=fn_recirculation_collision_dilation(lambda_rec, charges.q, charges.d, coefficients=coeffs.down),
    )


def sm_family_recirculated_quark_path_readouts(
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    powers: CenterHolonomyPowers = DEFAULT_CENTER_HOLONOMY_POWERS,
    coefficients: FNQuarkCoefficientMatrices | None = None,
) -> FamilyFNQuarkPathReadouts:
    """Return explicit hidden-path FN readouts for quark Higgs doors."""

    coeffs = _sm_family_quark_coefficients(powers, coefficients)
    up = fn_visible_recirculation_readout(lambda_rec, charges.q, charges.u, coefficients=coeffs.up)
    down = fn_visible_recirculation_readout(lambda_rec, charges.q, charges.d, coefficients=coeffs.down)
    return FamilyFNQuarkPathReadouts(
        up=up,
        down=down,
        pair=fn_visible_recirculation_pair_readout(up, down),
    )


def sm_family_calibrated_quark_path_readouts(
    target_yukawas: FNQuarkYukawas,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
) -> FamilyFNQuarkPathReadouts:
    """Return live FN path readouts calibrated to target quark Yukawas."""

    coefficients = fn_quark_coefficients_from_yukawas(target_yukawas, lambda_rec=lambda_rec, charges=charges)
    return sm_family_recirculated_quark_path_readouts(
        lambda_rec=lambda_rec,
        charges=charges,
        coefficients=coefficients,
    )


def sm_family_quark_path_readouts_from_masses_ckm(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
) -> FamilyFNQuarkPathReadouts:
    """Return live FN path readouts calibrated from masses and CKM input."""

    target = fn_quark_yukawas_from_masses_ckm(up_masses, down_masses, ckm)
    return sm_family_calibrated_quark_path_readouts(target, lambda_rec=lambda_rec, charges=charges)


def sm_family_quark_yukawas_from_path_readouts(
    readouts: FamilyFNQuarkPathReadouts | None = None,
) -> FNQuarkYukawas:
    """Collapse explicit FN path readouts to matrices for exact unitary collisions."""

    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()
    return FNQuarkYukawas(
        up=_validate_family_matrix(readouts.up.transfer, "readouts.up.transfer"),
        down=_validate_family_matrix(readouts.down.transfer, "readouts.down.transfer"),
    )


def sm_apply_family_recirculated_quark_door(
    left_family_state: jnp.ndarray,
    higgs_component: jnp.ndarray,
    dilation: FNUnitaryDilation,
) -> jnp.ndarray:
    """Apply one Higgs-scaled quark FN door through its finite dilation.

    The returned visible component equals ``higgs_component * Y.T @ left``.
    The hidden/auxiliary part is evolved by the exact unitary dilation; the
    normalization rescales the exposed contraction back to the physical source.
    """

    prepared = fn_prepare_visible_collision_state(left_family_state)
    evolved = fn_apply_recirculation_collision(dilation, prepared)
    visible = dilation.normalization * fn_read_visible_collision_output(evolved)
    higgs = jnp.asarray(higgs_component, dtype=visible.dtype)
    if higgs.ndim == 0:
        return higgs * visible
    return higgs[..., None] * visible


def sm_zero_family_fn_quark_aux_state(batch_shape: tuple[int, ...] = ()) -> FamilyFNQuarkAuxState:
    """Return a zero hidden auxiliary state for the up/down FN quark doors."""

    shape = (*batch_shape, SM_FAMILY_DIM)
    zeros = jnp.zeros(shape, dtype=jnp.complex64)
    return FamilyFNQuarkAuxState(up=zeros, down=zeros)


def sm_zero_family_fn_quark_state_aux(lattice_shape: tuple[int, int, int]) -> FamilyFNQuarkAuxState:
    """Return zero FN aux channels for every site, spin, color, and weak door."""

    return sm_zero_family_fn_quark_aux_state((*lattice_shape, 4, 3, 2))


def sm_zero_family_fn_quark_path_aux_state(
    batch_shape: tuple[int, ...] = (),
    readouts: FamilyFNQuarkPathReadouts | None = None,
) -> FamilyFNQuarkPathAuxState:
    """Return zero hidden path states for the up/down FN quark doors."""

    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()
    up_dim = int(readouts.up.network.unitary.shape[0])
    down_dim = int(readouts.down.network.unitary.shape[0])
    return FamilyFNQuarkPathAuxState(
        up=jnp.zeros((*batch_shape, up_dim), dtype=jnp.complex64),
        down=jnp.zeros((*batch_shape, down_dim), dtype=jnp.complex64),
    )


def sm_zero_family_fn_quark_path_state_aux(lattice_shape: tuple[int, int, int]) -> FamilyFNQuarkPathAuxState:
    """Return zero explicit FN path states for every site/spin/color/weak door."""

    return sm_zero_family_fn_quark_path_aux_state((*lattice_shape, 4, 3, 2))


def sm_apply_family_fn_quark_door_state(
    left_family_state: jnp.ndarray,
    hidden_state: jnp.ndarray,
    higgs_component: jnp.ndarray,
    dilation: FNUnitaryDilation,
) -> FamilyFNQuarkDoorOutput:
    """Evolve one FN quark door with a persistent hidden auxiliary state.

    The unitary part acts on ``[visible_left_family, hidden_aux]`` and returns
    ``[raw_visible_right_family, updated_hidden_aux]``.  The physical visible
    source is the normalized raw visible component multiplied by the local
    Higgs door component.
    """

    left = jnp.asarray(left_family_state, dtype=jnp.complex64)
    hidden = jnp.asarray(hidden_state, dtype=jnp.complex64)
    if left.shape[-1] != SM_FAMILY_DIM or hidden.shape[-1] != SM_FAMILY_DIM:
        raise ValueError("left_family_state and hidden_state must end with family dimension 3")
    state = jnp.concatenate([left, hidden], axis=-1)
    evolved = fn_apply_recirculation_collision(dilation, state)
    raw_visible = fn_read_visible_collision_output(evolved)
    updated_hidden = evolved[..., SM_FAMILY_DIM:]
    higgs = jnp.asarray(higgs_component, dtype=raw_visible.dtype)
    physical = dilation.normalization * raw_visible
    if higgs.ndim == 0:
        physical = higgs * physical
    else:
        physical = higgs[..., None] * physical
    return FamilyFNQuarkDoorOutput(raw_visible=raw_visible, physical_visible=physical, hidden=updated_hidden)


def sm_apply_family_fn_quark_path_door_state(
    left_family_state: jnp.ndarray,
    hidden_state: jnp.ndarray,
    higgs_component: jnp.ndarray,
    readout: FNVisibleRecirculationReadout,
) -> FamilyFNQuarkPathDoorOutput:
    """Evolve one quark door through the explicit FN path network."""

    output = fn_apply_visible_recirculation_path_state(readout, left_family_state, hidden_state)
    higgs = jnp.asarray(higgs_component, dtype=output.raw_visible.dtype)
    physical = output.raw_visible
    if higgs.ndim == 0:
        physical = higgs * physical
    else:
        physical = higgs[..., None] * physical
    return FamilyFNQuarkPathDoorOutput(
        raw_visible=output.raw_visible,
        physical_visible=physical,
        hidden=output.hidden,
    )


def sm_apply_family_fn_quark_pair_path_door_state(
    left_family_state: jnp.ndarray,
    up_hidden_state: jnp.ndarray,
    down_hidden_state: jnp.ndarray,
    up_higgs_component: jnp.ndarray,
    down_higgs_component: jnp.ndarray,
    readouts: FamilyFNQuarkPathReadouts,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evolve one left quark door through a shared up/down FN path injection."""

    output = fn_apply_visible_recirculation_pair_path_state(
        readouts.pair,
        left_family_state,
        up_hidden_state,
        down_hidden_state,
    )
    up_higgs = jnp.asarray(up_higgs_component, dtype=output.up_raw_visible.dtype)
    down_higgs = jnp.asarray(down_higgs_component, dtype=output.down_raw_visible.dtype)
    up_visible = output.up_raw_visible
    down_visible = output.down_raw_visible
    if up_higgs.ndim == 0:
        up_visible = up_higgs * up_visible
    else:
        up_visible = up_higgs[..., None] * up_visible
    if down_higgs.ndim == 0:
        down_visible = down_higgs * down_visible
    else:
        down_visible = down_higgs[..., None] * down_visible
    return up_visible, down_visible, output.visible, output.up_hidden, output.down_hidden


def _positive_sqrt_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return the positive square root of a Hermitian matrix or matrix batch."""

    arr = jnp.asarray(matrix, dtype=jnp.complex64)
    hermitian = 0.5 * (arr + jnp.swapaxes(jnp.conj(arr), -1, -2))
    values, vectors = jnp.linalg.eigh(hermitian)
    roots = jnp.sqrt(jnp.clip(values, min=0.0))
    weighted = vectors * roots[..., None, :]
    return weighted @ jnp.swapaxes(jnp.conj(vectors), -1, -2)


def _matvec(matrix: jnp.ndarray, vector: jnp.ndarray) -> jnp.ndarray:
    return jnp.squeeze(jnp.matmul(matrix, vector[..., None]), axis=-1)


def _apply_contraction_coupling(
    source_state: jnp.ndarray,
    target_state: jnp.ndarray,
    contraction: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a Julia/Halmos coupling for ``contraction: source -> target``."""

    source = jnp.asarray(source_state, dtype=jnp.complex64)
    target = jnp.asarray(target_state, dtype=jnp.complex64)
    block = jnp.asarray(contraction, dtype=jnp.complex64)
    if block.shape[-1] != source.shape[-1] or block.shape[-2] != target.shape[-1]:
        raise ValueError("contraction dimensions must match source and target states")
    adjoint = jnp.swapaxes(jnp.conj(block), -1, -2)
    source_identity = jnp.eye(block.shape[-1], dtype=block.dtype)
    target_identity = jnp.eye(block.shape[-2], dtype=block.dtype)
    source_defect = _positive_sqrt_matrix(source_identity - adjoint @ block)
    target_defect = _positive_sqrt_matrix(target_identity - block @ adjoint)
    new_source = _matvec(source_defect, source) - _matvec(adjoint, target)
    new_target = _matvec(block, source) + _matvec(target_defect, target)
    return new_source, new_target


def _step_sqrt_mix(step_size: float | jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    dt = jnp.asarray(step_size, dtype=real_dtype)
    return jnp.sqrt(jnp.clip(jnp.abs(dt), 0.0, 1.0)).astype(dtype)


def _step_sign(step_size: float | jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    real_dtype = jnp.real(jnp.asarray(0, dtype=dtype)).dtype
    dt = jnp.asarray(step_size, dtype=real_dtype)
    return jnp.where(dt < 0, -1.0, 1.0).astype(dtype)


def _bounded_complex_coupling(value: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=dtype)
    magnitude = jnp.abs(arr)
    denominator = jnp.where(magnitude > 1e-12, magnitude, 1.0)
    return jnp.tanh(magnitude).astype(dtype) * arr / denominator


def sm_apply_family_fn_quark_pair_path_unitary_door_state(
    left_family_state: jnp.ndarray,
    up_right_family_state: jnp.ndarray,
    down_right_family_state: jnp.ndarray,
    up_hidden_state: jnp.ndarray,
    down_hidden_state: jnp.ndarray,
    up_higgs_component: jnp.ndarray,
    down_higgs_component: jnp.ndarray,
    readouts: FamilyFNQuarkPathReadouts,
    *,
    step_size: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply one exact local FN path door on visible, hidden, and return slots.

    This is the constructive endpoint for the quark FN rule: the visible left
    source, up/down hidden path memories, and up/down right return slots are
    updated by contraction dilations.  The induced right-handed transfer is
    first order in ``step_size`` because both entry and Higgs exit carry
    ``sqrt(step_size)``.
    """

    left = jnp.asarray(left_family_state, dtype=jnp.complex64)
    up_right = jnp.asarray(up_right_family_state, dtype=jnp.complex64)
    down_right = jnp.asarray(down_right_family_state, dtype=jnp.complex64)
    up_hidden = jnp.asarray(up_hidden_state, dtype=jnp.complex64)
    down_hidden = jnp.asarray(down_hidden_state, dtype=jnp.complex64)
    if left.shape[-1] != SM_FAMILY_DIM or up_right.shape[-1] != SM_FAMILY_DIM or down_right.shape[-1] != SM_FAMILY_DIM:
        raise ValueError("visible family states must end with dimension 3")
    up_dim = int(readouts.up.network.unitary.shape[0])
    down_dim = int(readouts.down.network.unitary.shape[0])
    if up_hidden.shape[-1] != up_dim:
        raise ValueError(f"up_hidden_state must end with hidden dimension {up_dim}")
    if down_hidden.shape[-1] != down_dim:
        raise ValueError(f"down_hidden_state must end with hidden dimension {down_dim}")
    if step_size == 0:
        return left, up_right, down_right, up_hidden, down_hidden

    dtype = jnp.result_type(left, up_right, down_right, up_hidden, down_hidden, up_higgs_component, down_higgs_component, 1j)
    mix = _step_sqrt_mix(step_size, dtype)
    step_phase = _step_sign(step_size, dtype)
    hidden = jnp.concatenate((up_hidden.astype(dtype), down_hidden.astype(dtype)), axis=-1)
    combined_entry = jnp.concatenate((readouts.up.entry, readouts.down.entry), axis=0).astype(dtype)
    entry = mix * combined_entry / readouts.pair.entry_scale.astype(dtype)
    left_after, injected = _apply_contraction_coupling(left.astype(dtype), hidden, entry)
    injected_up = injected[..., :up_dim]
    injected_down = injected[..., up_dim:]
    evolved_up = _matvec(readouts.up.network.unitary.astype(dtype), injected_up)
    evolved_down = _matvec(readouts.down.network.unitary.astype(dtype), injected_down)

    up_higgs = _bounded_complex_coupling(up_higgs_component, dtype)
    down_higgs = _bounded_complex_coupling(down_higgs_component, dtype)
    up_exit_factor = -1j * step_phase * mix * up_higgs
    down_exit_factor = -1j * step_phase * mix * down_higgs
    up_exit = up_exit_factor[..., None, None] * (readouts.up.exit / readouts.up.exit_scale).astype(dtype)
    down_exit = down_exit_factor[..., None, None] * (readouts.down.exit / readouts.down.exit_scale).astype(dtype)
    up_hidden_after, up_right_after = _apply_contraction_coupling(evolved_up, up_right.astype(dtype), up_exit)
    down_hidden_after, down_right_after = _apply_contraction_coupling(evolved_down, down_right.astype(dtype), down_exit)
    return left_after, up_right_after, down_right_after, up_hidden_after, down_hidden_after


def sm_apply_family_fn_quark_aux_state(
    up_left_family_state: jnp.ndarray,
    down_left_family_state: jnp.ndarray,
    up_higgs_component: jnp.ndarray,
    down_higgs_component: jnp.ndarray,
    aux_state: FamilyFNQuarkAuxState,
    dilations: FamilyFNQuarkDilations,
) -> tuple[jnp.ndarray, jnp.ndarray, FamilyFNQuarkAuxState]:
    """Apply the up/down FN quark door updates and return physical sources."""

    up = sm_apply_family_fn_quark_door_state(up_left_family_state, aux_state.up, up_higgs_component, dilations.up)
    down = sm_apply_family_fn_quark_door_state(
        down_left_family_state,
        aux_state.down,
        down_higgs_component,
        dilations.down,
    )
    return up.physical_visible, down.physical_visible, FamilyFNQuarkAuxState(up=up.hidden, down=down.hidden)


def sm_family_fn_quark_path_state_source(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
) -> FamilyFNQuarkPathStateSource:
    """Return lattice-local quark sources from explicit FN path-network memory."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    expected_prefix = (*lattice_shape, 4, 3, 2)
    up_dim = int(readouts.up.network.unitary.shape[0])
    down_dim = int(readouts.down.network.unitary.shape[0])
    if aux_state.up.shape != (*expected_prefix, up_dim) or aux_state.down.shape != (*expected_prefix, down_dim):
        raise ValueError("FN quark path aux state has incompatible shape")

    h_tilde = sm_higgs_tilde(higgs)
    source = jnp.zeros_like(state)
    state_remainder = state
    up_sources = jnp.zeros((*lattice_shape, 4, 3, SM_FAMILY_DIM), dtype=state.dtype)
    down_sources = jnp.zeros_like(up_sources)
    updated_up_aux = jnp.zeros_like(aux_state.up)
    updated_down_aux = jnp.zeros_like(aux_state.down)

    for spin in range(4):
        for color in range(3):
            up_total = jnp.zeros((*lattice_shape, SM_FAMILY_DIM), dtype=state.dtype)
            down_total = jnp.zeros_like(up_total)
            for weak in range(2):
                left = state[..., spin, _q_index(color, weak), :]
                up_visible, down_visible, left_remainder, up_hidden, down_hidden = (
                    sm_apply_family_fn_quark_pair_path_door_state(
                        left,
                        aux_state.up[..., spin, color, weak, :],
                        aux_state.down[..., spin, color, weak, :],
                        h_tilde[..., weak],
                        higgs[..., weak],
                        readouts,
                    )
                )
                up_total = up_total + up_visible.astype(state.dtype)
                down_total = down_total + down_visible.astype(state.dtype)
                state_remainder = state_remainder.at[..., spin, _q_index(color, weak), :].set(
                    left_remainder.astype(state.dtype),
                )
                updated_up_aux = updated_up_aux.at[..., spin, color, weak, :].set(up_hidden)
                updated_down_aux = updated_down_aux.at[..., spin, color, weak, :].set(down_hidden)
            up_sources = up_sources.at[..., spin, color, :].set(up_total)
            down_sources = down_sources.at[..., spin, color, :].set(down_total)
            source = source.at[..., spin, _u_c_index(color), :].set(up_total)
            source = source.at[..., spin, _d_c_index(color), :].set(down_total)
    return FamilyFNQuarkPathStateSource(
        up=up_sources,
        down=down_sources,
        state_remainder=state_remainder,
        state_source=source,
        aux_state=FamilyFNQuarkPathAuxState(up=updated_up_aux, down=updated_down_aux),
    )


def sm_family_fn_quark_path_energy_local_density(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
) -> jnp.ndarray:
    """Return ``Re psi^dag beta source_FN(H)`` for explicit quark paths."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    source = sm_family_fn_quark_path_state_source(state, higgs, aux_state=aux_state, readouts=readouts).state_source
    beta_source = jnp.einsum("sr,...rif->...sif", sm_dirac_beta(state.dtype), source)
    local = jnp.sum(jnp.conj(state) * beta_source, axis=(-3, -2, -1))
    return jnp.real(local)


def sm_family_fn_quark_path_higgs_force(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
) -> jnp.ndarray:
    """Return the Higgs force generated by explicit FN quark path sources."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)

    def total_energy(real_part: jnp.ndarray, imag_part: jnp.ndarray) -> jnp.ndarray:
        field = real_part + 1j * imag_part
        return jnp.sum(
            sm_family_fn_quark_path_energy_local_density(
                state,
                field,
                aux_state=aux_state,
                readouts=readouts,
            ),
        )

    grad_real, grad_imag = jax.grad(total_energy, argnums=(0, 1))(jnp.real(higgs), jnp.imag(higgs))
    return -0.5 * (grad_real + 1j * grad_imag)


def sm_family_fn_quark_state_source(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkAuxState | None = None,
    dilations: FamilyFNQuarkDilations | None = None,
) -> FamilyFNQuarkStateSource:
    """Return lattice-local FN quark sources and updated hidden auxiliaries.

    Each quark doublet component feeds an independent FN auxiliary channel.
    The visible right-family returns are summed over the weak Higgs door and
    written into a full state-shaped source on the up/down singlet slots.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_state_aux(lattice_shape)
    if dilations is None:
        dilations = sm_family_recirculated_quark_dilations()
    expected_aux_shape = (*lattice_shape, 4, 3, 2, SM_FAMILY_DIM)
    if aux_state.up.shape != expected_aux_shape or aux_state.down.shape != expected_aux_shape:
        raise ValueError("FN quark aux state must have shape (nx, ny, nz, 4, 3, 2, 3)")

    h_tilde = sm_higgs_tilde(higgs)
    source = jnp.zeros_like(state)
    up_sources = jnp.zeros((*lattice_shape, 4, 3, SM_FAMILY_DIM), dtype=state.dtype)
    down_sources = jnp.zeros_like(up_sources)
    updated_up_aux = jnp.zeros_like(aux_state.up)
    updated_down_aux = jnp.zeros_like(aux_state.down)

    for spin in range(4):
        for color in range(3):
            up_total = jnp.zeros((*lattice_shape, SM_FAMILY_DIM), dtype=state.dtype)
            down_total = jnp.zeros_like(up_total)
            for weak in range(2):
                left = state[..., spin, _q_index(color, weak), :]
                up = sm_apply_family_fn_quark_door_state(
                    left,
                    aux_state.up[..., spin, color, weak, :],
                    h_tilde[..., weak],
                    dilations.up,
                )
                down = sm_apply_family_fn_quark_door_state(
                    left,
                    aux_state.down[..., spin, color, weak, :],
                    higgs[..., weak],
                    dilations.down,
                )
                up_total = up_total + up.physical_visible.astype(state.dtype)
                down_total = down_total + down.physical_visible.astype(state.dtype)
                updated_up_aux = updated_up_aux.at[..., spin, color, weak, :].set(up.hidden)
                updated_down_aux = updated_down_aux.at[..., spin, color, weak, :].set(down.hidden)
            up_sources = up_sources.at[..., spin, color, :].set(up_total)
            down_sources = down_sources.at[..., spin, color, :].set(down_total)
            source = source.at[..., spin, _u_c_index(color), :].set(up_total)
            source = source.at[..., spin, _d_c_index(color), :].set(down_total)
    return FamilyFNQuarkStateSource(
        up=up_sources,
        down=down_sources,
        state_source=source,
        aux_state=FamilyFNQuarkAuxState(up=updated_up_aux, down=updated_down_aux),
    )


def sm_apply_family_fn_quark_source_kick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkAuxState | None = None,
    dilations: FamilyFNQuarkDilations | None = None,
    *,
    step_size: float,
) -> FamilyFNQuarkSourceKick:
    """Advance the persistent FN quark doors and kick visible quark amplitudes.

    This is the simulator-level recirculation step: the quark source is produced
    by the hidden FN dilation, then inserted into the Dirac equation as
    ``-i dt beta source``.  It is deliberately a source kick, not a claim that
    the retarded FN bath has already been folded into one closed visible
    exponential.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_state_aux(lattice_shape)
    if step_size == 0:
        return FamilyFNQuarkSourceKick(state=state, source=jnp.zeros_like(state), aux_state=aux_state)

    source = sm_family_fn_quark_state_source(state, higgs, aux_state=aux_state, dilations=dilations)
    beta_source = jnp.einsum("sr,...rif->...sif", sm_dirac_beta(state.dtype), source.state_source)
    dt = jnp.asarray(step_size, dtype=jnp.real(jnp.asarray(0, dtype=state.dtype)).dtype)
    updated = state - 1j * dt * beta_source
    return FamilyFNQuarkSourceKick(state=updated, source=source.state_source, aux_state=source.aux_state)


def sm_apply_family_fn_quark_path_source_kick(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    *,
    step_size: float,
) -> FamilyFNQuarkPathSourceKick:
    """Kick visible quarks using the explicit FN path-network source."""

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    if step_size == 0:
        return FamilyFNQuarkPathSourceKick(state=state, source=jnp.zeros_like(state), aux_state=aux_state)

    source = sm_family_fn_quark_path_state_source(state, higgs, aux_state=aux_state, readouts=readouts)
    beta_source = jnp.einsum("sr,...rif->...sif", sm_dirac_beta(state.dtype), source.state_source)
    dt = jnp.asarray(step_size, dtype=jnp.real(jnp.asarray(0, dtype=state.dtype)).dtype)
    mix = jnp.clip(dt, -1.0, 1.0)
    updated_base = state + mix * (source.state_remainder - state)
    updated_aux = FamilyFNQuarkPathAuxState(
        up=aux_state.up + mix * (source.aux_state.up - aux_state.up),
        down=aux_state.down + mix * (source.aux_state.down - aux_state.down),
    )
    updated = updated_base - 1j * dt * beta_source
    return FamilyFNQuarkPathSourceKick(state=updated, source=source.state_source, aux_state=updated_aux)


def sm_apply_family_fn_quark_path_unitary_collision(
    state: jnp.ndarray,
    higgs: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
    readouts: FamilyFNQuarkPathReadouts | None = None,
    *,
    step_size: float,
) -> FamilyFNQuarkPathUnitaryCollision:
    """Apply the exact local quark FN path collision on visible and hidden slots.

    Unlike ``sm_apply_family_fn_quark_path_source_kick``, this routine does not
    add a source term to the visible state.  It evolves the quark doublet slot,
    the right-handed return slots, and the persistent hidden FN paths by local
    unitary contraction dilations.  The right-handed return is written into the
    Dirac-beta partner spin slot, matching the spin pairing of the standard
    family Yukawa collision.
    """

    lattice_shape = _validate_family_state(state)
    _validate_higgs_field(higgs, lattice_shape)
    if readouts is None:
        readouts = sm_family_recirculated_quark_path_readouts()
    if aux_state is None:
        aux_state = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
    expected_prefix = (*lattice_shape, 4, 3, 2)
    up_dim = int(readouts.up.network.unitary.shape[0])
    down_dim = int(readouts.down.network.unitary.shape[0])
    if aux_state.up.shape != (*expected_prefix, up_dim) or aux_state.down.shape != (*expected_prefix, down_dim):
        raise ValueError("FN quark path aux state has incompatible shape")
    if step_size == 0:
        return FamilyFNQuarkPathUnitaryCollision(state=state, aux_state=aux_state)

    target_spins = jnp.asarray([2, 3, 0, 1])
    h_tilde = sm_higgs_tilde(higgs)
    q_slots = state[..., :6, :].reshape((*lattice_shape, 4, 3, 2, SM_FAMILY_DIM))
    up_slots = state[..., 6:9, :]
    down_slots = state[..., 9:12, :]
    up_right_by_source_spin = jnp.take(up_slots, target_spins, axis=3)
    down_right_by_source_spin = jnp.take(down_slots, target_spins, axis=3)

    q_by_weak = jnp.moveaxis(q_slots, -2, 0)
    up_aux_by_weak = jnp.moveaxis(aux_state.up, -2, 0)
    down_aux_by_weak = jnp.moveaxis(aux_state.down, -2, 0)
    up_higgs_by_weak = jnp.moveaxis(h_tilde, -1, 0)
    down_higgs_by_weak = jnp.moveaxis(higgs, -1, 0)

    def weak_step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        up_right, down_right = carry
        left, up_hidden, down_hidden, up_higgs, down_higgs = inputs
        left_after, up_right_after, down_right_after, up_hidden_after, down_hidden_after = (
            sm_apply_family_fn_quark_pair_path_unitary_door_state(
                left,
                up_right,
                down_right,
                up_hidden,
                down_hidden,
                up_higgs[..., None, None],
                down_higgs[..., None, None],
                readouts,
                step_size=step_size,
            )
        )
        return (up_right_after, down_right_after), (left_after, up_hidden_after, down_hidden_after)

    (up_right_after_by_source_spin, down_right_after_by_source_spin), (
        q_after_by_weak,
        up_aux_after_by_weak,
        down_aux_after_by_weak,
    ) = jax.lax.scan(
        weak_step,
        (up_right_by_source_spin, down_right_by_source_spin),
        (q_by_weak, up_aux_by_weak, down_aux_by_weak, up_higgs_by_weak, down_higgs_by_weak),
    )

    q_after = jnp.moveaxis(q_after_by_weak, 0, -2).reshape((*lattice_shape, 4, 6, SM_FAMILY_DIM))
    up_aux_after = jnp.moveaxis(up_aux_after_by_weak, 0, -2)
    down_aux_after = jnp.moveaxis(down_aux_after_by_weak, 0, -2)
    up_after = jnp.take(up_right_after_by_source_spin, target_spins, axis=3)
    down_after = jnp.take(down_right_after_by_source_spin, target_spins, axis=3)
    quark_after = jnp.concatenate(
        (q_after.astype(state.dtype), up_after.astype(state.dtype), down_after.astype(state.dtype)),
        axis=-2,
    )
    updated = jnp.concatenate((quark_after, state[..., 12:, :]), axis=-2)

    return FamilyFNQuarkPathUnitaryCollision(
        state=updated,
        aux_state=FamilyFNQuarkPathAuxState(up=up_aux_after, down=down_aux_after),
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


def _family_pair_block_indices(left_internal: int, right_internal: int) -> list[int]:
    return [
        *[_family_flat_index(left_internal, family) for family in range(SM_FAMILY_DIM)],
        *[_family_flat_index(right_internal, family) for family in range(SM_FAMILY_DIM)],
    ]


def _unitary_gauge_yukawa_internal_pairs() -> list[tuple[int, int]]:
    pairs = []
    for copy_offset in (0, SM_CHIRAL16_DIM):
        for color in range(3):
            pairs.append((copy_offset + _q_index(color, 0), copy_offset + _u_c_index(color)))
            pairs.append((copy_offset + _q_index(color, 1), copy_offset + _d_c_index(color)))
        pairs.append((copy_offset + _l_index(0), copy_offset + _nu_c_index()))
        pairs.append((copy_offset + _l_index(1), copy_offset + _e_c_index()))
    return pairs


def _unitary_gauge_yukawa_internal_pair_indices() -> jnp.ndarray:
    return jnp.asarray(_unitary_gauge_yukawa_internal_pairs(), dtype=jnp.int32)


def _unitary_gauge_yukawa_block_indices() -> jnp.ndarray:
    blocks = []
    for left, right in _unitary_gauge_yukawa_internal_pairs():
        blocks.append(_family_pair_block_indices(left, right))
    return jnp.asarray(blocks, dtype=jnp.int32)


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
        jnp.eye(yukawa.shape[-1], dtype=yukawa.dtype),
        yukawa.shape,
    )
    zeros = jnp.zeros_like(yukawa)
    cos_internal = jnp.where(local_norm < 1e-12, identity, cos_internal)
    sin_internal = jnp.where(local_norm < 1e-12, zeros, sin_internal)
    return cos_internal, sin_internal


def _dirac_beta_swap_internal(action: jnp.ndarray) -> jnp.ndarray:
    """Apply beta to arrays with shape ``(..., spin, internal)``."""

    return jnp.concatenate((action[..., 2:, :], action[..., :2, :]), axis=-2)


def _dirac_beta_swap_block(action: jnp.ndarray) -> jnp.ndarray:
    """Apply beta to arrays with shape ``(..., spin, block, internal)``."""

    return jnp.concatenate((action[..., 2:, :, :], action[..., :2, :, :]), axis=-3)


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
    beta_action = _dirac_beta_swap_internal(sin_internal_action)
    return (cos_action - 1j * beta_action).reshape(state.shape)


def sm_family_yukawa_collision_cache(
    higgs: jnp.ndarray,
    *,
    step_size: float,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    assume_uniform: bool = False,
    use_unitary_gauge_blocks: bool = False,
) -> FamilyYukawaCollisionCache:
    """Precompute the exact local Yukawa collision functions.

    With ``assume_uniform=True``, only the first lattice site is used.  This is
    the production path for constant-Higgs rollout configs and avoids carrying a
    full spatial field of identical ``96 x 96`` collision functions.
    """

    _validate_higgs_field(higgs)
    local_higgs = higgs[:1, :1, :1] if assume_uniform else higgs
    yukawa = sm_family_yukawa_internal_matrix(
        local_higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    ).astype(jnp.complex64)
    cos_internal, sin_internal = _family_yukawa_cos_sin(yukawa, step_size)
    if assume_uniform:
        cos_internal = cos_internal[0, 0, 0]
        sin_internal = sin_internal[0, 0, 0]
    if assume_uniform and use_unitary_gauge_blocks:
        block_indices = _unitary_gauge_yukawa_block_indices()
        flat_block_indices = block_indices.reshape(-1)
        internal_pair_indices = _unitary_gauge_yukawa_internal_pair_indices()
        flat_internal_indices = internal_pair_indices.reshape(-1)
        cos_blocks = cos_internal[block_indices[:, :, None], block_indices[:, None, :]]
        sin_blocks = sin_internal[block_indices[:, :, None], block_indices[:, None, :]]
        return FamilyYukawaCollisionCache(
            cos_internal=jnp.zeros((0, 0), dtype=cos_internal.dtype),
            sin_internal=jnp.zeros((0, 0), dtype=sin_internal.dtype),
            block_indices=block_indices,
            block_inverse_indices=jnp.argsort(flat_block_indices),
            cos_blocks=cos_blocks,
            sin_blocks=sin_blocks,
            internal_pair_indices=internal_pair_indices,
            internal_inverse_indices=jnp.argsort(flat_internal_indices),
        )
    return FamilyYukawaCollisionCache(cos_internal=cos_internal, sin_internal=sin_internal)


def sm_apply_family_yukawa_collision_from_cache(
    state: jnp.ndarray,
    cache: FamilyYukawaCollisionCache,
) -> jnp.ndarray:
    """Apply a precomputed exact family Yukawa collision."""

    lattice_shape = _validate_family_state(state)
    flat_state = state.reshape((*lattice_shape, 4, SM_FAMILY_INTERNAL_DIM))
    if (
        cache.internal_pair_indices is not None
        and cache.internal_inverse_indices is not None
        and cache.cos_blocks is not None
        and cache.sin_blocks is not None
    ):
        internal_order = cache.internal_pair_indices.reshape(-1)
        ordered_state = jnp.take(state, internal_order, axis=-2)
        block_count = int(cache.cos_blocks.shape[0])
        block_state = ordered_state.reshape((*lattice_shape, 4, block_count, 2, SM_FAMILY_DIM)).reshape(
            (*lattice_shape, 4, block_count, 2 * SM_FAMILY_DIM),
        )
        cos_blocks = jnp.asarray(cache.cos_blocks, dtype=state.dtype)
        sin_blocks = jnp.asarray(cache.sin_blocks, dtype=state.dtype)
        cos_action = jnp.einsum("bij,...sbj->...sbi", cos_blocks, block_state)
        sin_action = jnp.einsum("bij,...sbj->...sbi", sin_blocks, block_state)
        beta_action = _dirac_beta_swap_block(sin_action)
        block_output = cos_action - 1j * beta_action
        ordered_internal = block_output.reshape((*lattice_shape, 4, 2 * block_count, SM_FAMILY_DIM))
        return jnp.take(ordered_internal, cache.internal_inverse_indices, axis=-2)
    if (
        cache.block_indices is not None
        and cache.block_inverse_indices is not None
        and cache.cos_blocks is not None
        and cache.sin_blocks is not None
    ):
        block_state = jnp.take(flat_state, cache.block_indices, axis=-1)
        cos_blocks = jnp.asarray(cache.cos_blocks, dtype=state.dtype)
        sin_blocks = jnp.asarray(cache.sin_blocks, dtype=state.dtype)
        cos_action = jnp.einsum("bij,...sbj->...sbi", cos_blocks, block_state)
        sin_action = jnp.einsum("bij,...sbj->...sbi", sin_blocks, block_state)
        beta_action = _dirac_beta_swap_block(sin_action)
        block_output = cos_action - 1j * beta_action
        ordered = block_output.reshape((*lattice_shape, 4, SM_FAMILY_INTERNAL_DIM))
        return jnp.take(ordered, cache.block_inverse_indices, axis=-1).reshape(state.shape)
    cos_internal = jnp.asarray(cache.cos_internal, dtype=state.dtype)
    sin_internal = jnp.asarray(cache.sin_internal, dtype=state.dtype)
    if cos_internal.shape == (SM_FAMILY_INTERNAL_DIM, SM_FAMILY_INTERNAL_DIM):
        cos_action = jnp.einsum("ij,...sj->...si", cos_internal, flat_state)
        sin_internal_action = jnp.einsum("ij,...sj->...si", sin_internal, flat_state)
    else:
        if cos_internal.shape != (*lattice_shape, SM_FAMILY_INTERNAL_DIM, SM_FAMILY_INTERNAL_DIM):
            raise ValueError("cached Yukawa collision has incompatible shape")
        cos_action = jnp.einsum("...ij,...sj->...si", cos_internal, flat_state)
        sin_internal_action = jnp.einsum("...ij,...sj->...si", sin_internal, flat_state)
    beta_action = _dirac_beta_swap_internal(sin_internal_action)
    return (cos_action - 1j * beta_action).reshape(state.shape)


def sm_family_quark_sector_internal_yukawa(yukawa: jnp.ndarray, higgs_scale: float | jnp.ndarray = 1.0) -> jnp.ndarray:
    """Return the active ``Q_f <-> f_R`` family Yukawa block for one quark door."""

    matrix = _validate_family_matrix(yukawa, "yukawa")
    scale = jnp.asarray(higgs_scale, dtype=matrix.dtype)
    zeros = jnp.zeros_like(matrix)
    upper = jnp.concatenate((zeros, scale * matrix), axis=1)
    lower = jnp.concatenate((jnp.swapaxes(jnp.conj(scale * matrix), -1, -2), zeros), axis=1)
    return jnp.concatenate((upper, lower), axis=0)


def sm_family_quark_sector_symbol(
    k: jnp.ndarray | tuple[float, float, float],
    yukawa: jnp.ndarray,
    higgs_scale: float | jnp.ndarray = 1.0,
    step_size: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the exact one-sector family mass collision composed with BCC streaming."""

    internal = sm_family_quark_sector_internal_yukawa(yukawa, higgs_scale).astype(jnp.complex64)
    cos_internal, sin_internal = _family_yukawa_cos_sin(internal, step_size)
    spin_identity = jnp.eye(4, dtype=jnp.complex64)
    beta = sm_dirac_beta(jnp.complex64)
    collision = (
        jnp.einsum("sr,ij->sirj", spin_identity, cos_internal)
        - 1j * jnp.einsum("sr,ij->sirj", beta, sin_internal)
    ).reshape((4 * internal.shape[0], 4 * internal.shape[0]))
    stream = jnp.kron(bcc_dirac_split_symbol(k, dtype=jnp.complex64), jnp.eye(internal.shape[0], dtype=jnp.complex64))
    return collision @ stream


def _matched_family_mass_angles(
    symbol: jnp.ndarray,
    target_angles: jnp.ndarray,
) -> jnp.ndarray:
    phases = jnp.sort(jnp.abs(jnp.angle(jnp.linalg.eigvals(symbol))))
    distances = jnp.abs(phases[:, None] - target_angles[None, :])
    return phases[jnp.argmin(distances, axis=0)]


def sm_family_quark_sector_mass_angles(
    k: jnp.ndarray | tuple[float, float, float],
    yukawa: jnp.ndarray,
    higgs_scale: float | jnp.ndarray = 1.0,
    step_size: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return quasienergy gaps matched to the sector's singular mass angles."""

    scale = jnp.asarray(higgs_scale, dtype=jnp.float32)
    dt = jnp.asarray(step_size, dtype=jnp.float32)
    target_angles = jnp.sort(dt * scale * fn_singular_masses(yukawa))
    symbol = sm_family_quark_sector_symbol(k, yukawa, higgs_scale, step_size)
    return _matched_family_mass_angles(symbol, target_angles)


def sm_family_fn_mass_spectrum_probe(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray | None = None,
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    higgs_vev: float = 1.0,
    step_size: float = 1.0,
    k: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FamilyFNMassSpectrumProbe:
    """Measure calibrated FN quark masses as quasienergy gaps of the exact symbol."""

    target_yukawas = fn_quark_yukawas_from_masses_ckm(up_masses, down_masses, ckm)
    readouts = sm_family_quark_path_readouts_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
    )
    recovered_yukawas = sm_family_quark_yukawas_from_path_readouts(readouts)
    higgs_scale = jnp.sqrt(jnp.asarray(higgs_vev, dtype=jnp.float32))
    dt = jnp.asarray(step_size, dtype=jnp.float32)
    target_up = jnp.sort(dt * higgs_scale * fn_singular_masses(recovered_yukawas.up))
    target_down = jnp.sort(dt * higgs_scale * fn_singular_masses(recovered_yukawas.down))
    up_symbol = sm_family_quark_sector_symbol(k, recovered_yukawas.up, higgs_scale, step_size)
    down_symbol = sm_family_quark_sector_symbol(k, recovered_yukawas.down, higgs_scale, step_size)
    measured_up = _matched_family_mass_angles(up_symbol, target_up)
    measured_down = _matched_family_mass_angles(down_symbol, target_down)
    target_ckm = fn_ckm_from_yukawas(target_yukawas.up, target_yukawas.down)
    recovered_ckm = fn_ckm_from_yukawas(recovered_yukawas.up, recovered_yukawas.down)
    return FamilyFNMassSpectrumProbe(
        target_yukawas=target_yukawas,
        recovered_yukawas=recovered_yukawas,
        target_up_mass_angles=target_up,
        target_down_mass_angles=target_down,
        measured_up_mass_angles=measured_up,
        measured_down_mass_angles=measured_down,
        up_gap_residual=jnp.max(jnp.abs(measured_up - target_up)),
        down_gap_residual=jnp.max(jnp.abs(measured_down - target_down)),
        ckm_abs_residual=jnp.max(jnp.abs(jnp.abs(recovered_ckm) - jnp.abs(target_ckm))),
        up_symbol_unitarity_residual=jnp.max(
            jnp.abs(jnp.swapaxes(jnp.conj(up_symbol), -1, -2) @ up_symbol - jnp.eye(up_symbol.shape[0], dtype=up_symbol.dtype)),
        ),
        down_symbol_unitarity_residual=jnp.max(
            jnp.abs(
                jnp.swapaxes(jnp.conj(down_symbol), -1, -2) @ down_symbol - jnp.eye(down_symbol.shape[0], dtype=down_symbol.dtype),
            ),
        ),
    )


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
    dilations = sm_family_recirculated_quark_dilations()
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
    probe = jnp.asarray([1.0 + 0.0j, -0.35 + 0.2j, 0.1 - 0.45j], dtype=jnp.complex64)
    up_door = sm_apply_family_recirculated_quark_door(probe, sm_higgs_tilde(higgs)[0, 0, 0, 0], dilations.up)
    down_door = sm_apply_family_recirculated_quark_door(probe, higgs[0, 0, 0, 1], dilations.down)
    expected_up_door = sm_higgs_tilde(higgs)[0, 0, 0, 0] * (jnp.swapaxes(quark_yukawas.up, -1, -2) @ probe)
    expected_down_door = higgs[0, 0, 0, 1] * (jnp.swapaxes(quark_yukawas.down, -1, -2) @ probe)

    return FamilyHiggsYukawaDiagnostics(
        family_yukawa_hermitian_residual=sm_yukawa_hermitian_residual(matrix),
        fn_recirculated_quark_yukawa_residual=jnp.maximum(
            jnp.max(jnp.abs(quark_yukawas.up - reference_quark_yukawas.up)),
            jnp.max(jnp.abs(quark_yukawas.down - reference_quark_yukawas.down)),
        ),
        fn_recirculated_embedding_residual=recirculated_embedding,
        fn_quark_dilation_unitarity_residual=jnp.maximum(
            fn_unitary_dilation_residual(dilations.up),
            fn_unitary_dilation_residual(dilations.down),
        ),
        fn_quark_door_transfer_residual=jnp.maximum(
            jnp.max(jnp.abs(up_door - expected_up_door)),
            jnp.max(jnp.abs(down_door - expected_down_door)),
        ),
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
