"""Production-facing rollout runner for QCA_SMv0.

This module is the compact simulator front door.  It deliberately composes the
existing BCC, static-gauge, Higgs, FN, and center-CP layers instead of adding a
new physics rule.  The runner is a field-QCA rollout: global arrays hold fields,
while each update is a local BCC stencil plus an optional site-local collision.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_split_axis_step, bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.sm_cp import (
    CenterCPPhenomenologyVerdict,
    CenterCPTextureSeed,
    sm_center_cp_phenomenology_verdict,
    sm_center_cp_order_one_coefficients,
    sm_center_cp_verdict_magnitudes,
    sm_center_powers_from_wilson_flux_rule,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyFNQuarkPathAuxState,
    FamilyFNQuarkPathReadouts,
    FamilyLeptonYukawas,
    FamilyYukawaCollisionCache,
    deterministic_sm_family_state,
    sm_apply_family_fn_quark_path_unitary_collision,
    sm_apply_family_yukawa_collision,
    sm_apply_family_yukawa_collision_from_cache,
    sm_family_recirculated_quark_path_readouts,
    sm_family_yukawa_collision_cache,
    sm_zero_family_fn_quark_path_state_aux,
)
from clifford_3plus2_d5.qca_smv0.sm_fn import (
    DEFAULT_FN_QUARK_CHARGES,
    FN_LAMBDA_WOLFENSTEIN,
    FNQuarkCharges,
    FNQuarkCoefficientMatrices,
    FNQuarkYukawas,
    SM_FAMILY_DIM,
    fn_effective_yukawa,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
    SM_CHIRAL16_DIM,
    SM_INTERNAL_DIM,
    deterministic_sm_state,
    sm_free_dirac_family_step,
    sm_free_dirac_family_split_axis_step,
    sm_free_dirac_internal_step,
    sm_free_dirac_internal_split_axis_step,
    sm_gauged_family_dirac_step,
    sm_gauged_dirac_step,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE


class QCASMRolloutConfig(NamedTuple):
    """Configuration for one QCA_SMv0 field rollout."""

    links: jnp.ndarray | None = None
    higgs: jnp.ndarray | None = None
    yukawa_step_size: float = 0.0
    collision_mode: str = "effective_yukawa"
    quark_yukawas: FNQuarkYukawas | None = None
    quark_path_readouts: FamilyFNQuarkPathReadouts | None = None
    lepton_yukawas: FamilyLeptonYukawas | None = None
    family_yukawa_collision_cache: FamilyYukawaCollisionCache | None = None
    yukawa_collision_strategy: str = "fast"
    stream_mode: str = "hop_sum"
    record_observables: bool = True
    record_density: bool = False


class QCASMState(NamedTuple):
    """Persistent QCA_SMv0 rollout state.

    ``visible_state`` is the field array.  ``fn_path_aux_state`` carries the
    hidden FN recirculation path memory used only by the exact reference mode
    ``collision_mode='fn_dilation'``; it is not part of the production
    Standard-Model fibre.  Static backgrounds such as gauge links and the Higgs
    field remain in the rollout config.
    """

    visible_state: jnp.ndarray
    fn_path_aux_state: FamilyFNQuarkPathAuxState | None = None


class QCASMStateMemoryFootprint(NamedTuple):
    """Array-size estimate for one persistent QCA_SMv0 state."""

    visible_complex_elements: int
    fn_path_aux_complex_elements: int
    total_complex_elements: int
    complex64_bytes: int
    complex128_bytes: int


class QCASMArrayMemoryFootprint(NamedTuple):
    """Actual array-memory footprint for a JAX tree."""

    array_elements: int
    array_bytes: int


class QCASMRolloutMemoryFootprint(NamedTuple):
    """Runtime array footprint for a rollout state plus branch config."""

    state: QCASMStateMemoryFootprint
    state_array_elements: int
    state_array_bytes: int
    config_array_elements: int
    config_array_bytes: int
    total_array_elements: int
    total_array_bytes: int


class QCASMRolloutResult(NamedTuple):
    """Result of a compact QCA_SMv0 rollout."""

    final_state: jnp.ndarray
    final_qca_state: QCASMState
    norm_history: jnp.ndarray
    extended_norm_history: jnp.ndarray
    max_density_history: jnp.ndarray
    density_history: jnp.ndarray
    final_fn_path_aux_state: FamilyFNQuarkPathAuxState | None
    used_gauge_links: bool
    used_higgs_fn_collision: bool
    used_fn_dilation_collision: bool
    collision_mode: str
    steps_completed: int
    memory_footprint: QCASMStateMemoryFootprint
    rollout_memory_footprint: QCASMRolloutMemoryFootprint


class QCASMCalibratedRolloutConfig(NamedTuple):
    """Center-CP calibrated rollout config and its phenomenology verdict."""

    config: QCASMRolloutConfig
    verdict: CenterCPPhenomenologyVerdict


class QCASMProductionContract(NamedTuple):
    """Boolean contract summary for the compressed production hot path."""

    uses_production_api: bool
    state_only: bool
    gauge_links_present: bool
    structured_collision_cache_present: bool
    raw_yukawa_arrays_present: bool
    raw_readout_arrays_present: bool
    higgs_field_present: bool
    lean_effective_yukawa: bool


class QCASMProductionRolloutSetup(NamedTuple):
    """Prepared production hot-path rollout state and memory footprint."""

    config: QCASMRolloutConfig
    initial_qca_state: QCASMState
    rollout_memory_footprint: QCASMRolloutMemoryFootprint


class QCASMProductionRolloutResult(NamedTuple):
    """Pure state-evolution result for the production hot path."""

    final_qca_state: QCASMState
    initial_rollout_memory_footprint: QCASMRolloutMemoryFootprint
    final_rollout_memory_footprint: QCASMRolloutMemoryFootprint


class QCASMFamilyCarrierPopulations(NamedTuple):
    """Carrier-aware population reductions for the ``4 x 32 x 3`` field.

    ``sector`` is ordered as ``(Q, u^c, d^c, L, e^c, nu^c)`` and includes both
    duplicated chiral-16 copies of the 32-component simulator register.
    ``dirac_chirality`` is ordered as ``(left Weyl pair, right Weyl pair)``.
    ``internal_copy`` is ordered as the first and second 16-component simulator
    copies.  These are raw squared-norm populations, not normalized fractions.
    """

    sector: jnp.ndarray
    dirac_chirality: jnp.ndarray
    family: jnp.ndarray
    internal_copy: jnp.ndarray


class QCASMProductionRolloutObservables(NamedTuple):
    """Production state rollout plus lean initial/final scalar observables."""

    production: QCASMProductionRolloutResult
    norm_initial: jnp.ndarray
    norm_final: jnp.ndarray
    extended_norm_initial: jnp.ndarray
    extended_norm_final: jnp.ndarray
    max_density_initial: jnp.ndarray
    max_density_final: jnp.ndarray
    carrier_populations_initial: QCASMFamilyCarrierPopulations
    carrier_populations_final: QCASMFamilyCarrierPopulations
    used_higgs_fn_collision: bool
    used_fn_dilation_collision: bool
    collision_mode: str
    steps_completed: int


class QCASMProductionRolloutObservableArrays(NamedTuple):
    """JIT-friendly production rollout output containing arrays only."""

    final_qca_state: QCASMState
    norm_initial: jnp.ndarray
    norm_final: jnp.ndarray
    extended_norm_initial: jnp.ndarray
    extended_norm_final: jnp.ndarray
    max_density_initial: jnp.ndarray
    max_density_final: jnp.ndarray
    carrier_populations_initial: QCASMFamilyCarrierPopulations
    carrier_populations_final: QCASMFamilyCarrierPopulations


class QCASMCalibratedProductionRolloutSetup(NamedTuple):
    """Prepared production rollout plus its masses/CKM calibration verdict."""

    setup: QCASMProductionRolloutSetup
    verdict: CenterCPPhenomenologyVerdict


class QCASMCalibratedProductionRolloutResult(NamedTuple):
    """Production rollout result plus its masses/CKM calibration verdict."""

    production: QCASMProductionRolloutResult
    verdict: CenterCPPhenomenologyVerdict


class QCASMCalibratedCoefficientDiagnostics(NamedTuple):
    """Compact public coefficient diagnostics for a calibrated production run."""

    selected_label: str
    passed: bool
    status: str
    failure_reasons: tuple[str, ...]
    up_magnitudes: jnp.ndarray
    down_magnitudes: jnp.ndarray
    up_center_powers: jnp.ndarray
    down_center_powers: jnp.ndarray
    coefficient_residual: jnp.ndarray
    phase_residual: jnp.ndarray
    up_mass_log_rms: jnp.ndarray
    down_mass_log_rms: jnp.ndarray
    ckm_abs_residual: jnp.ndarray
    jarlskog_relative_residual: jnp.ndarray
    magnitude_min: jnp.ndarray
    magnitude_max: jnp.ndarray
    magnitude_mean: jnp.ndarray
    steps_completed: int


class QCASMCalibratedProductionRolloutObservables(NamedTuple):
    """Lean production observables plus the masses/CKM calibration verdict."""

    production_observables: QCASMProductionRolloutObservables
    verdict: CenterCPPhenomenologyVerdict
    coefficient_diagnostics: QCASMCalibratedCoefficientDiagnostics
    production_contract: QCASMProductionContract


class QCASMCalibratedQuarkFamilyResponse(NamedTuple):
    """Field-level family response from pure ``Q`` basis states.

    Rows are input ``Q`` families.  Columns are output singlet families.  The
    amplitude matrices are normalized complex projections onto the target
    carrier sector after running the calibrated production update.  Population
    matrices are the corresponding total squared norms in that target sector.
    The expected one-tick amplitudes are the corresponding normalized local
    projections read directly from the calibrated structured Yukawa cache.
    """

    verdict: CenterCPPhenomenologyVerdict
    up_amplitude: jnp.ndarray
    down_amplitude: jnp.ndarray
    up_expected_one_tick_amplitude: jnp.ndarray
    down_expected_one_tick_amplitude: jnp.ndarray
    up_expected_one_tick_residual: jnp.ndarray
    down_expected_one_tick_residual: jnp.ndarray
    up_population: jnp.ndarray
    down_population: jnp.ndarray
    up_final_norm: jnp.ndarray
    down_final_norm: jnp.ndarray
    steps_completed: int


def _quark_yukawas_from_coefficients(
    coefficients: FNQuarkCoefficientMatrices,
    *,
    lambda_rec: float,
    charges: FNQuarkCharges,
) -> FNQuarkYukawas:
    return FNQuarkYukawas(
        up=fn_effective_yukawa(lambda_rec, charges.q, charges.u, coefficients=coefficients.up),
        down=fn_effective_yukawa(lambda_rec, charges.q, charges.d, coefficients=coefficients.down),
    )


def _zero_family_lepton_yukawas() -> FamilyLeptonYukawas:
    zeros = jnp.zeros((SM_FAMILY_DIM, SM_FAMILY_DIM), dtype=jnp.complex64)
    return FamilyLeptonYukawas(neutrino=zeros, electron=zeros)


def _production_lepton_yukawas(lepton_yukawas: FamilyLeptonYukawas | None) -> FamilyLeptonYukawas:
    return lepton_yukawas if lepton_yukawas is not None else _zero_family_lepton_yukawas()


def sm_qca_calibrated_coefficient_diagnostics(
    verdict: CenterCPPhenomenologyVerdict,
) -> QCASMCalibratedCoefficientDiagnostics:
    """Return user-facing coefficient diagnostics from a center-CP verdict."""

    factorization = verdict.fit.factorization
    residuals = verdict.fit.residuals
    return QCASMCalibratedCoefficientDiagnostics(
        selected_label=verdict.selected_label,
        passed=bool(verdict.passed),
        status=verdict.status,
        failure_reasons=tuple(verdict.failure_reasons),
        up_magnitudes=jnp.asarray(factorization.magnitudes.up),
        down_magnitudes=jnp.asarray(factorization.magnitudes.down),
        up_center_powers=jnp.asarray(factorization.center_powers.up, dtype=jnp.int32),
        down_center_powers=jnp.asarray(factorization.center_powers.down, dtype=jnp.int32),
        coefficient_residual=factorization.coefficient_residual,
        phase_residual=factorization.phase_residual,
        up_mass_log_rms=residuals.up_mass_log_rms,
        down_mass_log_rms=residuals.down_mass_log_rms,
        ckm_abs_residual=residuals.ckm_abs_residual,
        jarlskog_relative_residual=residuals.jarlskog_relative_residual,
        magnitude_min=residuals.magnitude_min,
        magnitude_max=residuals.magnitude_max,
        magnitude_mean=residuals.magnitude_mean,
        steps_completed=int(verdict.fit.steps_completed),
    )


def _qca_spatial_shape(state: jnp.ndarray) -> tuple[int, int, int]:
    if state.ndim not in (4, 5, 6):
        raise ValueError("state must be a Dirac, SM-internal, or family SM-internal field")
    if state.shape[:3] == ():
        raise ValueError("state must have three spatial axes")
    if state.ndim == 4 and state.shape[-1] != 4:
        raise ValueError("free Dirac state must have shape (nx,ny,nz,4)")
    if state.ndim == 5 and state.shape[-2:] != (4, SM_INTERNAL_DIM):
        raise ValueError("SM state must have shape (nx,ny,nz,4,32)")
    if state.ndim == 6 and state.shape[-3:-1] != (4, SM_INTERNAL_DIM):
        raise ValueError("family SM state must have shape (nx,ny,nz,4,32,3)")
    return int(state.shape[0]), int(state.shape[1]), int(state.shape[2])


def sm_qca_density(state: jnp.ndarray) -> jnp.ndarray:
    """Return site density by summing all non-spatial amplitudes."""

    _qca_spatial_shape(state)
    return jnp.sum(jnp.abs(state) ** 2, axis=tuple(range(3, state.ndim)))


def _sector_indices(base_indices: tuple[int, ...]) -> jnp.ndarray:
    return jnp.asarray(
        (*base_indices, *(index + SM_CHIRAL16_DIM for index in base_indices)),
        dtype=jnp.int32,
    )


SM_QCA_CARRIER_SECTOR_LABELS = ("Q", "u_c", "d_c", "L", "e_c", "nu_c")


def _validate_axis_index(value: int, *, size: int, label: str) -> int:
    index = int(value)
    if index < 0 or index >= size:
        raise ValueError(f"{label} must be in [0,{size}), got {value}")
    return index


def _carrier_base_index(sector: str, *, color: int = 0, weak: int = 0) -> int:
    """Return the chiral-16 carrier index for one SM sector label."""

    if sector == "Q":
        return 2 * _validate_axis_index(color, size=3, label="color") + _validate_axis_index(
            weak,
            size=2,
            label="weak",
        )
    if sector == "u_c":
        return 6 + _validate_axis_index(color, size=3, label="color")
    if sector == "d_c":
        return 9 + _validate_axis_index(color, size=3, label="color")
    if sector == "L":
        return 12 + _validate_axis_index(weak, size=2, label="weak")
    if sector == "e_c":
        return 14
    if sector == "nu_c":
        return 15
    raise ValueError(f"sector must be one of {SM_QCA_CARRIER_SECTOR_LABELS}, got {sector!r}")


def sm_qca_family_carrier_basis_state(
    lattice_shape: tuple[int, int, int],
    *,
    site: tuple[int, int, int] = (0, 0, 0),
    dirac: int = 0,
    sector: str = "Q",
    color: int = 0,
    weak: int = 0,
    family: int = 0,
    internal_copy: int = 0,
    amplitude: complex = 1.0 + 0.0j,
    dtype: jnp.dtype = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a one-component normalized family-SM carrier state.

    This is the production-facing initial-condition helper for actual carrier
    experiments.  It writes one amplitude into the visible field layout
    ``(nx,ny,nz,4,32,3)``.  ``sector`` uses the chiral labels
    ``Q, u_c, d_c, L, e_c, nu_c``; ``internal_copy`` selects one of the two
    duplicated chiral-16 simulator copies inside the 32-component register.
    """

    nx, ny, nz = (int(dim) for dim in lattice_shape)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("lattice dimensions must be positive")
    x = _validate_axis_index(site[0], size=nx, label="site[0]")
    y = _validate_axis_index(site[1], size=ny, label="site[1]")
    z = _validate_axis_index(site[2], size=nz, label="site[2]")
    dirac_index = _validate_axis_index(dirac, size=4, label="dirac")
    family_index = _validate_axis_index(family, size=SM_FAMILY_DIM, label="family")
    copy_index = _validate_axis_index(internal_copy, size=2, label="internal_copy")
    internal_index = copy_index * SM_CHIRAL16_DIM + _carrier_base_index(sector, color=color, weak=weak)
    state = jnp.zeros((nx, ny, nz, 4, SM_INTERNAL_DIM, SM_FAMILY_DIM), dtype=dtype)
    return state.at[x, y, z, dirac_index, internal_index, family_index].set(
        jnp.asarray(amplitude, dtype=dtype),
    )


def _sm_qca_family_carrier_basis_stack(
    lattice_shape: tuple[int, int, int],
    *,
    site: tuple[int, int, int],
    dirac: int,
    sector: str,
    color: int,
    weak: int,
    internal_copy: int,
    dtype: jnp.dtype = DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            sm_qca_family_carrier_basis_state(
                lattice_shape,
                site=site,
                dirac=dirac,
                sector=sector,
                color=color,
                weak=weak,
                family=family,
                internal_copy=internal_copy,
                dtype=dtype,
            )
            for family in range(SM_FAMILY_DIM)
        ),
        axis=0,
    )


def _sm_qca_target_family_populations(
    final_states: jnp.ndarray,
    *,
    sector: str,
    color: int,
    weak: int,
    internal_copy: int,
) -> jnp.ndarray:
    """Return output family populations for a batch of final carrier states."""

    if final_states.ndim != 7 or final_states.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("batched final states must have shape (batch,nx,ny,nz,4,32,3)")
    internal_index = internal_copy * SM_CHIRAL16_DIM + _carrier_base_index(sector, color=color, weak=weak)
    density = jnp.abs(final_states[..., internal_index, :]) ** 2
    return jnp.sum(density, axis=(1, 2, 3, 4))


def _sm_qca_target_family_amplitudes(
    final_states: jnp.ndarray,
    *,
    sector: str,
    color: int,
    weak: int,
    internal_copy: int,
) -> jnp.ndarray:
    """Return normalized complex output-family amplitudes for final states.

    The projection is uniform over spatial and Dirac support for one selected
    target carrier component.  With this normalization, ``abs(amplitude)**2`` is
    bounded by the target population for the same sector/family entry.
    """

    if final_states.ndim != 7 or final_states.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("batched final states must have shape (batch,nx,ny,nz,4,32,3)")
    internal_index = internal_copy * SM_CHIRAL16_DIM + _carrier_base_index(sector, color=color, weak=weak)
    target = final_states[..., internal_index, :]
    component_count = final_states.shape[1] * final_states.shape[2] * final_states.shape[3] * final_states.shape[4]
    normalization = jnp.sqrt(jnp.asarray(component_count, dtype=jnp.float32))
    return jnp.sum(target, axis=(1, 2, 3, 4)) / normalization


def _sm_qca_expected_quark_one_tick_amplitudes(
    config: QCASMRolloutConfig,
    lattice_shape: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local one-tick ``Q -> u_c/d_c`` amplitudes from the cache.

    The sign and transpose follow the structured collision convention:
    ``sin_right_left_blocks[k]`` maps left-family inputs to right-family
    outputs, while the local unitary contributes the Dirac-beta factor ``-i``.
    The normalization matches ``_sm_qca_target_family_amplitudes``.
    """

    cache = config.family_yukawa_collision_cache
    if cache is None or cache.sin_right_left_blocks is None:
        raise ValueError("expected quark response requires a structured effective-Yukawa cache")
    nx, ny, nz = (int(dim) for dim in lattice_shape)
    component_count = nx * ny * nz * 4
    normalization = jnp.sqrt(jnp.asarray(component_count, dtype=jnp.float32))
    up = -1j * jnp.swapaxes(cache.sin_right_left_blocks[0], -1, -2) / normalization
    down = -1j * jnp.swapaxes(cache.sin_right_left_blocks[1], -1, -2) / normalization
    return up, down


def sm_qca_family_carrier_populations(state: jnp.ndarray) -> QCASMFamilyCarrierPopulations:
    """Return compact carrier populations for a three-family SM field state."""

    _qca_spatial_shape(state)
    if state.ndim != 6 or state.shape[-3:] != (4, SM_INTERNAL_DIM, SM_FAMILY_DIM):
        raise ValueError("family carrier populations require state shape (nx,ny,nz,4,32,3)")

    density = jnp.abs(state) ** 2
    sectors = (
        _sector_indices((0, 1, 2, 3, 4, 5)),
        _sector_indices((6, 7, 8)),
        _sector_indices((9, 10, 11)),
        _sector_indices((12, 13)),
        _sector_indices((14,)),
        _sector_indices((15,)),
    )
    sector_populations = jnp.stack(
        tuple(jnp.sum(jnp.take(density, indices, axis=4)) for indices in sectors),
        axis=0,
    )
    return QCASMFamilyCarrierPopulations(
        sector=sector_populations,
        dirac_chirality=jnp.stack((jnp.sum(density[..., :2, :, :]), jnp.sum(density[..., 2:, :, :]))),
        family=jnp.sum(density, axis=(0, 1, 2, 3, 4)),
        internal_copy=jnp.stack(
            (
                jnp.sum(density[..., :SM_CHIRAL16_DIM, :]),
                jnp.sum(density[..., SM_CHIRAL16_DIM:, :]),
            ),
        ),
    )


def sm_qca_total_norm(state: jnp.ndarray) -> jnp.ndarray:
    """Return total squared norm of a QCA_SMv0 field state."""

    return jnp.sum(sm_qca_density(state))


def sm_qca_fn_path_aux_norm(aux_state: FamilyFNQuarkPathAuxState | None) -> jnp.ndarray:
    """Return total squared norm in the explicit FN path hidden slots."""

    if aux_state is None:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sum(jnp.abs(aux_state.up) ** 2) + jnp.sum(jnp.abs(aux_state.down) ** 2)


def sm_qca_extended_total_norm(
    state: jnp.ndarray,
    aux_state: FamilyFNQuarkPathAuxState | None = None,
) -> jnp.ndarray:
    """Return visible plus hidden FN-path norm."""

    return sm_qca_total_norm(state) + sm_qca_fn_path_aux_norm(aux_state)


def sm_qca_state_memory_footprint(
    qca_state: QCASMState,
) -> QCASMStateMemoryFootprint:
    """Return visible and hidden complex-element counts for a rollout state."""

    visible_complex_elements = int(qca_state.visible_state.size)
    fn_path_aux_complex_elements = 0
    if qca_state.fn_path_aux_state is not None:
        fn_path_aux_complex_elements = int(qca_state.fn_path_aux_state.up.size + qca_state.fn_path_aux_state.down.size)
    total_complex_elements = visible_complex_elements + fn_path_aux_complex_elements
    return QCASMStateMemoryFootprint(
        visible_complex_elements=visible_complex_elements,
        fn_path_aux_complex_elements=fn_path_aux_complex_elements,
        total_complex_elements=total_complex_elements,
        complex64_bytes=8 * total_complex_elements,
        complex128_bytes=16 * total_complex_elements,
    )


def _tree_array_memory_footprint(tree: object) -> QCASMArrayMemoryFootprint:
    array_elements = 0
    array_bytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype") or not hasattr(leaf, "size"):
            continue
        array = jnp.asarray(leaf)
        array_elements += int(array.size)
        array_bytes += int(array.size * array.dtype.itemsize)
    return QCASMArrayMemoryFootprint(array_elements=array_elements, array_bytes=array_bytes)


def _qca_config_runtime_arrays(config: QCASMRolloutConfig) -> tuple[object, ...]:
    arrays: list[object] = []
    if config.links is not None:
        arrays.append(config.links)
    if config.yukawa_step_size == 0.0:
        return tuple(arrays)

    _validate_collision_mode(config.collision_mode)
    if config.collision_mode == "fn_dilation":
        if config.higgs is None:
            return tuple(arrays)
        arrays.append(config.higgs)
        if config.quark_path_readouts is not None:
            arrays.append(config.quark_path_readouts)
        return tuple(arrays)

    if config.family_yukawa_collision_cache is not None:
        arrays.append(config.family_yukawa_collision_cache)
        return tuple(arrays)

    if config.higgs is None:
        return tuple(arrays)
    arrays.append(config.higgs)
    if config.quark_yukawas is not None:
        arrays.append(config.quark_yukawas)
    if config.lepton_yukawas is not None:
        arrays.append(config.lepton_yukawas)
    return tuple(arrays)


def sm_qca_config_memory_footprint(config: QCASMRolloutConfig) -> QCASMArrayMemoryFootprint:
    """Return actual runtime array bytes used by the selected config branch."""

    return _tree_array_memory_footprint(_qca_config_runtime_arrays(config))


def sm_qca_rollout_memory_footprint(
    qca_state: QCASMState,
    config: QCASMRolloutConfig,
) -> QCASMRolloutMemoryFootprint:
    """Return runtime state plus selected-config array memory."""

    state = sm_qca_state_memory_footprint(qca_state)
    state_arrays = _tree_array_memory_footprint(qca_state)
    config_arrays = sm_qca_config_memory_footprint(config)
    return QCASMRolloutMemoryFootprint(
        state=state,
        state_array_elements=state_arrays.array_elements,
        state_array_bytes=state_arrays.array_bytes,
        config_array_elements=config_arrays.array_elements,
        config_array_bytes=config_arrays.array_bytes,
        total_array_elements=state_arrays.array_elements + config_arrays.array_elements,
        total_array_bytes=state_arrays.array_bytes + config_arrays.array_bytes,
    )


def _validate_stream_mode(stream_mode: str) -> None:
    if stream_mode not in ("hop_sum", "split_axis"):
        raise ValueError("stream_mode must be 'hop_sum' or 'split_axis'")


def _resolve_stream_mode_for_links(links: jnp.ndarray | None, stream_mode: str | None) -> str:
    resolved = ("hop_sum" if links is not None else "split_axis") if stream_mode is None else stream_mode
    _validate_stream_mode(resolved)
    return resolved


def _validate_static_links_for_rollout(
    links: jnp.ndarray | None,
    lattice_shape: tuple[int, int, int],
    stream_mode: str,
) -> None:
    if links is None:
        return
    if stream_mode == "split_axis":
        raise ValueError("static gauge links require stream_mode='hop_sum'")
    if links.ndim != 6 or links.shape[:3] != lattice_shape or links.shape[-3:] != (8, SM_INTERNAL_DIM, SM_INTERNAL_DIM):
        raise ValueError("links must have shape (nx, ny, nz, 8, 32, 32)")


def _family_stream_step(state: jnp.ndarray, links: jnp.ndarray | None, stream_mode: str) -> jnp.ndarray:
    _validate_stream_mode(stream_mode)
    if links is None:
        if stream_mode == "split_axis":
            return sm_free_dirac_family_split_axis_step(state)
        return sm_free_dirac_family_step(state)
    if stream_mode == "split_axis":
        raise ValueError("split_axis stream_mode is only implemented for free family streams")
    return sm_gauged_family_dirac_step(state, links)


def _validate_collision_mode(collision_mode: str) -> None:
    if collision_mode not in ("fn_dilation", "effective_yukawa"):
        raise ValueError("collision_mode must be 'fn_dilation' or 'effective_yukawa'")


def _validate_yukawa_collision_strategy(strategy: str) -> None:
    if strategy not in ("memory", "fast"):
        raise ValueError("yukawa_collision_strategy must be 'memory' or 'fast'")


def _validate_production_hot_path(config: QCASMRolloutConfig) -> None:
    _validate_collision_mode(config.collision_mode)
    _validate_stream_mode(config.stream_mode)
    _validate_yukawa_collision_strategy(config.yukawa_collision_strategy)
    if config.collision_mode != "effective_yukawa":
        raise ValueError("production hot path requires collision_mode='effective_yukawa'")
    if config.links is not None and config.stream_mode == "split_axis":
        raise ValueError("production static gauge links require stream_mode='hop_sum'")
    if config.record_observables or config.record_density:
        raise ValueError("production hot path requires state-only output; disable record_observables and record_density")
    if config.yukawa_step_size != 0.0 and config.family_yukawa_collision_cache is None:
        raise ValueError("production hot path requires a compressed effective-Yukawa cache")


def sm_qca_production_contract(
    config: QCASMRolloutConfig,
    *,
    uses_production_api: bool,
) -> QCASMProductionContract:
    """Return the lean-production contract state for a rollout config."""

    raw_yukawa_arrays_present = config.quark_yukawas is not None or config.lepton_yukawas is not None
    raw_readout_arrays_present = config.quark_path_readouts is not None
    structured_collision_cache_present = config.family_yukawa_collision_cache is not None
    higgs_field_present = config.higgs is not None
    gauge_links_present = config.links is not None
    state_only = not config.record_observables and not config.record_density
    lean_effective_yukawa = bool(
        uses_production_api
        and config.collision_mode == "effective_yukawa"
        and state_only
        and not raw_yukawa_arrays_present
        and not raw_readout_arrays_present
        and not higgs_field_present
        and (config.yukawa_step_size == 0.0 or structured_collision_cache_present)
    )
    return QCASMProductionContract(
        uses_production_api=bool(uses_production_api),
        state_only=state_only,
        gauge_links_present=gauge_links_present,
        structured_collision_cache_present=structured_collision_cache_present,
        raw_yukawa_arrays_present=raw_yukawa_arrays_present,
        raw_readout_arrays_present=raw_readout_arrays_present,
        higgs_field_present=higgs_field_present,
        lean_effective_yukawa=lean_effective_yukawa,
    )


def _lean_cached_effective_yukawa_config(config: QCASMRolloutConfig) -> QCASMRolloutConfig:
    """Drop source Yukawa/readout arrays once the compressed cache is present."""

    if config.collision_mode != "effective_yukawa":
        return config
    if config.family_yukawa_collision_cache is None and config.yukawa_step_size != 0.0:
        return config
    return config._replace(
        higgs=None,
        quark_yukawas=None,
        quark_path_readouts=None,
        lepton_yukawas=None,
    )


def _has_effective_yukawa_cache(config: QCASMRolloutConfig) -> bool:
    return config.collision_mode == "effective_yukawa" and config.family_yukawa_collision_cache is not None


def _uses_higgs_fn_collision(config: QCASMRolloutConfig) -> bool:
    if config.yukawa_step_size == 0.0:
        return False
    return config.higgs is not None or _has_effective_yukawa_cache(config)


def sm_qca_prepare_state(
    visible_state: jnp.ndarray,
    config: QCASMRolloutConfig = QCASMRolloutConfig(),
    *,
    initial_fn_path_aux_state: FamilyFNQuarkPathAuxState | None = None,
) -> QCASMState:
    """Return a persistent rollout state with hidden FN memory initialized."""

    spatial_shape = _qca_spatial_shape(visible_state)
    _validate_collision_mode(config.collision_mode)
    _validate_stream_mode(config.stream_mode)
    _validate_yukawa_collision_strategy(config.yukawa_collision_strategy)
    aux_state = initial_fn_path_aux_state
    if (
        config.collision_mode == "fn_dilation"
        and config.higgs is not None
        and config.yukawa_step_size != 0.0
    ):
        if visible_state.ndim != 6:
            raise ValueError("FN dilation collision requires a family SM state")
        if aux_state is None:
            aux_state = sm_zero_family_fn_quark_path_state_aux(spatial_shape)
    return QCASMState(visible_state=visible_state, fn_path_aux_state=aux_state)


def _sm_qca_rollout_step_with_aux(
    state: jnp.ndarray,
    config: QCASMRolloutConfig = QCASMRolloutConfig(),
    fn_path_aux_state: FamilyFNQuarkPathAuxState | None = None,
) -> tuple[jnp.ndarray, FamilyFNQuarkPathAuxState | None]:
    """Apply one compact QCA_SMv0 tick.

    The tick is gauge/BCC streaming followed by an optional site-local
    three-family Higgs/FN collision.  Dynamic gauge and Higgs field evolution
    intentionally remain in the heavier production-tick modules.
    """

    _qca_spatial_shape(state)
    if state.ndim == 4:
        if config.links is not None:
            raise ValueError("gauge links require an SM-internal state")
        streamed = bcc_dirac_split_axis_step(state) if config.stream_mode == "split_axis" else bcc_dirac_step(state)
    elif state.ndim == 5:
        if config.links is None:
            streamed = (
                sm_free_dirac_internal_split_axis_step(state)
                if config.stream_mode == "split_axis"
                else sm_free_dirac_internal_step(state)
            )
        else:
            if config.stream_mode == "split_axis":
                raise ValueError("split_axis stream_mode is only implemented for free SM streams")
            streamed = sm_gauged_dirac_step(state, config.links)
    else:
        streamed = _family_stream_step(state, config.links, config.stream_mode)

    has_cached_effective_yukawa = _has_effective_yukawa_cache(config)
    if config.yukawa_step_size == 0.0 or (config.higgs is None and not has_cached_effective_yukawa):
        return streamed, fn_path_aux_state
    if streamed.ndim != 6:
        raise ValueError("Higgs/FN collision requires a family SM state")
    _validate_collision_mode(config.collision_mode)
    _validate_yukawa_collision_strategy(config.yukawa_collision_strategy)
    if config.collision_mode == "fn_dilation":
        if config.higgs is None:
            raise ValueError("FN dilation collision requires a Higgs field")
        lattice_shape = _qca_spatial_shape(streamed)
        if fn_path_aux_state is None:
            fn_path_aux_state = sm_zero_family_fn_quark_path_state_aux(lattice_shape)
        collision = sm_apply_family_fn_quark_path_unitary_collision(
            streamed,
            config.higgs,
            fn_path_aux_state,
            config.quark_path_readouts,
            step_size=config.yukawa_step_size,
        )
        return collision.state, collision.aux_state
    if config.family_yukawa_collision_cache is not None:
        return sm_apply_family_yukawa_collision_from_cache(
            streamed,
            config.family_yukawa_collision_cache,
            strategy=config.yukawa_collision_strategy,
        ), fn_path_aux_state
    return (
        sm_apply_family_yukawa_collision(
            streamed,
            config.higgs,
            step_size=config.yukawa_step_size,
            quark_yukawas=config.quark_yukawas,
            lepton_yukawas=config.lepton_yukawas,
        ),
        fn_path_aux_state,
    )


def sm_qca_rollout_step(state: jnp.ndarray, config: QCASMRolloutConfig = QCASMRolloutConfig()) -> jnp.ndarray:
    """Apply one compact QCA_SMv0 tick and return the visible field state."""

    updated, _ = _sm_qca_rollout_step_with_aux(state, config)
    return updated


def sm_qca_state_step(
    qca_state: QCASMState,
    config: QCASMRolloutConfig = QCASMRolloutConfig(),
) -> QCASMState:
    """Apply one compact QCA_SMv0 tick to a persistent state object."""

    updated, aux_state = _sm_qca_rollout_step_with_aux(
        qca_state.visible_state,
        config,
        qca_state.fn_path_aux_state,
    )
    return QCASMState(visible_state=updated, fn_path_aux_state=aux_state)


def sm_run_qca_state_rollout(
    qca_state: QCASMState,
    config: QCASMRolloutConfig,
    steps: int,
) -> QCASMState:
    """Run a pure state rollout without diagnostic reductions."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")

    def loop_step(
        _index: jnp.ndarray,
        carry: QCASMState,
    ) -> QCASMState:
        return sm_qca_state_step(carry, config)

    return jax.lax.fori_loop(
        0,
        steps,
        loop_step,
        qca_state,
    )


def sm_jit_qca_state_rollout(
    config: QCASMRolloutConfig,
    steps: int,
    *,
    donate_state: bool = True,
):
    """Return the JIT-compiled production state rollout kernel.

    This is the GPU hot-path entry point: it runs state evolution without
    per-step observable reductions and can donate the carried state buffer so
    XLA may alias input storage into the final state.
    """

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")

    def state_rollout_kernel(local_qca_state: QCASMState) -> QCASMState:
        return sm_run_qca_state_rollout(local_qca_state, config, steps=steps)

    return jax.jit(state_rollout_kernel, donate_argnums=(0,) if donate_state else ())


def sm_jit_qca_production_rollout(
    config: QCASMRolloutConfig,
    steps: int,
    *,
    donate_state: bool = True,
):
    """Return a JIT kernel for the compressed production hot path."""

    _validate_production_hot_path(config)
    return sm_jit_qca_state_rollout(config, steps, donate_state=donate_state)


def sm_jit_qca_production_rollout_observable_arrays(
    config: QCASMRolloutConfig,
    steps: int,
    *,
    donate_state: bool = True,
):
    """Return a JIT kernel for production final state plus scalar observables."""

    _validate_production_hot_path(config)
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")

    def observed_rollout_kernel(local_qca_state: QCASMState) -> QCASMProductionRolloutObservableArrays:
        final_qca_state = sm_run_qca_state_rollout(local_qca_state, config, steps=steps)
        return QCASMProductionRolloutObservableArrays(
            final_qca_state=final_qca_state,
            norm_initial=sm_qca_total_norm(local_qca_state.visible_state),
            norm_final=sm_qca_total_norm(final_qca_state.visible_state),
            extended_norm_initial=sm_qca_extended_total_norm(
                local_qca_state.visible_state,
                local_qca_state.fn_path_aux_state,
            ),
            extended_norm_final=sm_qca_extended_total_norm(
                final_qca_state.visible_state,
                final_qca_state.fn_path_aux_state,
            ),
            max_density_initial=jnp.max(sm_qca_density(local_qca_state.visible_state)),
            max_density_final=jnp.max(sm_qca_density(final_qca_state.visible_state)),
            carrier_populations_initial=sm_qca_family_carrier_populations(local_qca_state.visible_state),
            carrier_populations_final=sm_qca_family_carrier_populations(final_qca_state.visible_state),
        )

    return jax.jit(observed_rollout_kernel, donate_argnums=(0,) if donate_state else ())


def _sm_qca_observables(
    qca_state: QCASMState,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    density = sm_qca_density(qca_state.visible_state)
    return (
        jnp.sum(density),
        sm_qca_extended_total_norm(qca_state.visible_state, qca_state.fn_path_aux_state),
        jnp.max(density),
        density,
    )


def sm_run_qca_rollout(
    config: QCASMRolloutConfig,
    initial_state: jnp.ndarray,
    steps: int,
    *,
    initial_fn_path_aux_state: FamilyFNQuarkPathAuxState | None = None,
) -> QCASMRolloutResult:
    """Run a compact QCA_SMv0 field rollout and record basic observables."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    qca_state = sm_qca_prepare_state(
        initial_state,
        config,
        initial_fn_path_aux_state=initial_fn_path_aux_state,
    )
    initial_norm, initial_extended_norm, initial_max_density, initial_density = _sm_qca_observables(qca_state)

    if not config.record_observables:
        final_qca_state = sm_run_qca_state_rollout(qca_state, config, steps)
        final_norm, final_extended_norm, final_max_density, _final_density = _sm_qca_observables(final_qca_state)
        norm_history = jnp.stack((initial_norm, final_norm))
        extended_norm_history = jnp.stack((initial_extended_norm, final_extended_norm))
        max_density_history = jnp.stack((initial_max_density, final_max_density))
        spatial_shape = _qca_spatial_shape(initial_state)
        density_history = jnp.zeros((0, *spatial_shape), dtype=jnp.real(initial_state).dtype)
    elif config.record_density:

        def scan_step(
            carry: QCASMState,
            _unused: None,
        ) -> tuple[QCASMState, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            next_state = sm_qca_state_step(carry, config)
            return next_state, _sm_qca_observables(next_state)

        final_qca_state, (step_norms, step_extended_norms, step_max_densities, step_densities) = jax.lax.scan(
            scan_step,
            qca_state,
            None,
            length=steps,
        )
        density_history = jnp.concatenate((initial_density[None, ...], step_densities), axis=0)
    else:

        def scan_step(
            carry: QCASMState,
            _unused: None,
        ) -> tuple[QCASMState, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
            next_state = sm_qca_state_step(carry, config)
            density = sm_qca_density(next_state.visible_state)
            return (
                next_state,
                (
                    jnp.sum(density),
                    sm_qca_extended_total_norm(next_state.visible_state, next_state.fn_path_aux_state),
                    jnp.max(density),
                ),
            )

        final_qca_state, (step_norms, step_extended_norms, step_max_densities) = jax.lax.scan(
            scan_step,
            qca_state,
            None,
            length=steps,
        )
        spatial_shape = _qca_spatial_shape(initial_state)
        density_history = jnp.zeros((0, *spatial_shape), dtype=jnp.real(initial_state).dtype)

    if config.record_observables:
        norm_history = jnp.concatenate((initial_norm[None], step_norms), axis=0)
        extended_norm_history = jnp.concatenate((initial_extended_norm[None], step_extended_norms), axis=0)
        max_density_history = jnp.concatenate((initial_max_density[None], step_max_densities), axis=0)
    spatial_shape = _qca_spatial_shape(initial_state)
    return QCASMRolloutResult(
        final_state=final_qca_state.visible_state,
        final_qca_state=final_qca_state,
        norm_history=norm_history,
        extended_norm_history=extended_norm_history,
        max_density_history=max_density_history,
        density_history=density_history,
        final_fn_path_aux_state=final_qca_state.fn_path_aux_state,
        used_gauge_links=config.links is not None,
        used_higgs_fn_collision=_uses_higgs_fn_collision(config),
        used_fn_dilation_collision=(
            config.higgs is not None
            and config.yukawa_step_size != 0.0
            and config.collision_mode == "fn_dilation"
        ),
        collision_mode=config.collision_mode,
        steps_completed=steps,
        memory_footprint=sm_qca_state_memory_footprint(final_qca_state),
        rollout_memory_footprint=sm_qca_rollout_memory_footprint(final_qca_state, config),
    )


def sm_qca_center_cp_rollout_config(
    lattice_shape: tuple[int, int, int],
    *,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    record_observables: bool = True,
    record_density: bool = False,
    collision_mode: str = "effective_yukawa",
    yukawa_collision_strategy: str = "fast",
) -> QCASMRolloutConfig:
    """Return a local Higgs/FN config using the generated center-CP powers."""

    _validate_collision_mode(collision_mode)
    resolved_stream_mode = _resolve_stream_mode_for_links(links, stream_mode)
    _validate_yukawa_collision_strategy(yukawa_collision_strategy)
    _validate_static_links_for_rollout(links, lattice_shape, resolved_stream_mode)
    powers = sm_center_powers_from_wilson_flux_rule()
    coefficients = sm_center_cp_order_one_coefficients(sm_center_cp_verdict_magnitudes(), powers=powers)
    quark_yukawas = _quark_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    active_lepton_yukawas = _production_lepton_yukawas(lepton_yukawas)
    needs_effective_cache = collision_mode == "effective_yukawa" and yukawa_step_size != 0.0
    runtime_higgs = (
        sm_constant_higgs(lattice_shape, vev=higgs_vev)
        if collision_mode == "fn_dilation" and yukawa_step_size != 0.0
        else None
    )
    cache_higgs = sm_constant_higgs((1, 1, 1), vev=higgs_vev) if needs_effective_cache else None
    quark_path_readouts = (
        sm_family_recirculated_quark_path_readouts(
            lambda_rec=lambda_rec,
            charges=charges,
            coefficients=coefficients,
        )
        if collision_mode == "fn_dilation"
        else None
    )
    collision_cache = (
        sm_family_yukawa_collision_cache(
            cache_higgs,
            step_size=yukawa_step_size,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=active_lepton_yukawas,
            assume_uniform=True,
            use_unitary_gauge_blocks=True,
        )
        if cache_higgs is not None
        else None
    )
    return QCASMRolloutConfig(
        links=links,
        higgs=runtime_higgs,
        yukawa_step_size=yukawa_step_size,
        collision_mode=collision_mode,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=active_lepton_yukawas,
        quark_path_readouts=quark_path_readouts,
        family_yukawa_collision_cache=collision_cache,
        yukawa_collision_strategy=yukawa_collision_strategy,
        stream_mode=resolved_stream_mode,
        record_observables=record_observables,
        record_density=record_density,
    )


def sm_qca_production_rollout_config(
    lattice_shape: tuple[int, int, int],
    *,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
) -> QCASMRolloutConfig:
    """Return the default compressed, state-only QCA_SMv0 production config."""

    config = sm_qca_center_cp_rollout_config(
        lattice_shape,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        record_observables=False,
        record_density=False,
        collision_mode="effective_yukawa",
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    config = _lean_cached_effective_yukawa_config(config)
    _validate_production_hot_path(config)
    return config


def sm_qca_prepare_production_rollout(
    lattice_shape: tuple[int, int, int],
    *,
    initial_state: jnp.ndarray | None = None,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
) -> QCASMProductionRolloutSetup:
    """Prepare the compressed production hot path with upfront memory accounting."""

    state = deterministic_qca_family_state(lattice_shape) if initial_state is None else initial_state
    if _qca_spatial_shape(state) != lattice_shape:
        raise ValueError("initial_state spatial shape must match lattice_shape")
    config = sm_qca_production_rollout_config(
        lattice_shape,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    qca_state = sm_qca_prepare_state(state, config)
    return QCASMProductionRolloutSetup(
        config=config,
        initial_qca_state=qca_state,
        rollout_memory_footprint=sm_qca_rollout_memory_footprint(qca_state, config),
    )


def sm_qca_prepare_production_rollout_from_config(
    initial_state: jnp.ndarray,
    config: QCASMRolloutConfig,
) -> QCASMProductionRolloutSetup:
    """Prepare a production setup from an already calibrated hot-path config."""

    config = _lean_cached_effective_yukawa_config(config)
    _validate_production_hot_path(config)
    qca_state = sm_qca_prepare_state(initial_state, config)
    return QCASMProductionRolloutSetup(
        config=config,
        initial_qca_state=qca_state,
        rollout_memory_footprint=sm_qca_rollout_memory_footprint(qca_state, config),
    )


def sm_run_qca_production_rollout(
    setup: QCASMProductionRolloutSetup,
    steps: int,
) -> QCASMProductionRolloutResult:
    """Run pure state evolution on a prepared compressed production setup."""

    _validate_production_hot_path(setup.config)
    final_qca_state = sm_run_qca_state_rollout(setup.initial_qca_state, setup.config, steps)
    return QCASMProductionRolloutResult(
        final_qca_state=final_qca_state,
        initial_rollout_memory_footprint=setup.rollout_memory_footprint,
        final_rollout_memory_footprint=sm_qca_rollout_memory_footprint(final_qca_state, setup.config),
    )


def sm_run_qca_production_rollout_with_observables(
    setup: QCASMProductionRolloutSetup,
    steps: int,
) -> QCASMProductionRolloutObservables:
    """Run production state evolution and return only initial/final observables."""

    _validate_production_hot_path(setup.config)
    initial_state = setup.initial_qca_state
    production = sm_run_qca_production_rollout(setup, steps)
    final_state = production.final_qca_state
    return QCASMProductionRolloutObservables(
        production=production,
        norm_initial=sm_qca_total_norm(initial_state.visible_state),
        norm_final=sm_qca_total_norm(final_state.visible_state),
        extended_norm_initial=sm_qca_extended_total_norm(
            initial_state.visible_state,
            initial_state.fn_path_aux_state,
        ),
        extended_norm_final=sm_qca_extended_total_norm(
            final_state.visible_state,
            final_state.fn_path_aux_state,
        ),
        max_density_initial=jnp.max(sm_qca_density(initial_state.visible_state)),
        max_density_final=jnp.max(sm_qca_density(final_state.visible_state)),
        carrier_populations_initial=sm_qca_family_carrier_populations(initial_state.visible_state),
        carrier_populations_final=sm_qca_family_carrier_populations(final_state.visible_state),
        used_higgs_fn_collision=_uses_higgs_fn_collision(setup.config),
        used_fn_dilation_collision=False,
        collision_mode=setup.config.collision_mode,
        steps_completed=steps,
    )


def sm_run_jitted_qca_production_rollout_with_observables(
    setup: QCASMProductionRolloutSetup,
    steps: int,
    *,
    donate_state: bool = True,
) -> QCASMProductionRolloutObservables:
    """Run the lean production-observables path through a JIT-compiled kernel."""

    _validate_production_hot_path(setup.config)
    observed = sm_jit_qca_production_rollout_observable_arrays(
        setup.config,
        steps,
        donate_state=donate_state,
    )(setup.initial_qca_state)
    production = QCASMProductionRolloutResult(
        final_qca_state=observed.final_qca_state,
        initial_rollout_memory_footprint=setup.rollout_memory_footprint,
        final_rollout_memory_footprint=sm_qca_rollout_memory_footprint(observed.final_qca_state, setup.config),
    )
    return QCASMProductionRolloutObservables(
        production=production,
        norm_initial=observed.norm_initial,
        norm_final=observed.norm_final,
        extended_norm_initial=observed.extended_norm_initial,
        extended_norm_final=observed.extended_norm_final,
        max_density_initial=observed.max_density_initial,
        max_density_final=observed.max_density_final,
        carrier_populations_initial=observed.carrier_populations_initial,
        carrier_populations_final=observed.carrier_populations_final,
        used_higgs_fn_collision=_uses_higgs_fn_collision(setup.config),
        used_fn_dilation_collision=False,
        collision_mode=setup.config.collision_mode,
        steps_completed=steps,
    )


def sm_run_qca_calibrated_production_rollout(
    calibrated_setup: QCASMCalibratedProductionRolloutSetup,
    steps: int,
) -> QCASMCalibratedProductionRolloutResult:
    """Run a prepared masses/CKM-calibrated compressed production rollout."""

    return QCASMCalibratedProductionRolloutResult(
        production=sm_run_qca_production_rollout(calibrated_setup.setup, steps),
        verdict=calibrated_setup.verdict,
    )


def sm_run_qca_calibrated_production_rollout_with_observables(
    calibrated_setup: QCASMCalibratedProductionRolloutSetup,
    steps: int,
) -> QCASMCalibratedProductionRolloutObservables:
    """Run a calibrated production rollout with only initial/final observables."""

    return QCASMCalibratedProductionRolloutObservables(
        production_observables=sm_run_qca_production_rollout_with_observables(
            calibrated_setup.setup,
            steps,
        ),
        verdict=calibrated_setup.verdict,
        coefficient_diagnostics=sm_qca_calibrated_coefficient_diagnostics(calibrated_setup.verdict),
        production_contract=sm_qca_production_contract(calibrated_setup.setup.config, uses_production_api=True),
    )


def sm_run_jitted_qca_calibrated_production_rollout_with_observables(
    calibrated_setup: QCASMCalibratedProductionRolloutSetup,
    steps: int,
    *,
    donate_state: bool = True,
) -> QCASMCalibratedProductionRolloutObservables:
    """Run a calibrated lean-observable production rollout through JIT."""

    return QCASMCalibratedProductionRolloutObservables(
        production_observables=sm_run_jitted_qca_production_rollout_with_observables(
            calibrated_setup.setup,
            steps,
            donate_state=donate_state,
        ),
        verdict=calibrated_setup.verdict,
        coefficient_diagnostics=sm_qca_calibrated_coefficient_diagnostics(calibrated_setup.verdict),
        production_contract=sm_qca_production_contract(calibrated_setup.setup.config, uses_production_api=True),
    )


def sm_qca_rollout_config_from_masses_ckm(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    record_observables: bool = True,
    record_density: bool = False,
    collision_mode: str = "effective_yukawa",
    yukawa_collision_strategy: str = "fast",
) -> QCASMCalibratedRolloutConfig:
    """Configure a rollout from masses/CKM through the center-CP FN layer."""

    _validate_collision_mode(collision_mode)
    resolved_stream_mode = _resolve_stream_mode_for_links(links, stream_mode)
    _validate_yukawa_collision_strategy(yukawa_collision_strategy)
    _validate_static_links_for_rollout(links, lattice_shape, resolved_stream_mode)
    generated_seed = CenterCPTextureSeed(
        powers=sm_center_powers_from_wilson_flux_rule(),
        initial_magnitudes=sm_center_cp_verdict_magnitudes(),
        label="wilson_flux_rule",
    )
    verdict = sm_center_cp_phenomenology_verdict(
        up_masses,
        down_masses,
        ckm,
        lambda_rec=lambda_rec,
        charges=charges,
        candidate_seeds=(generated_seed,),
        steps=center_fit_steps,
    )
    coefficients = verdict.fit.factorization.reconstructed_coefficients
    quark_yukawas = _quark_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    active_lepton_yukawas = _production_lepton_yukawas(lepton_yukawas)
    needs_effective_cache = collision_mode == "effective_yukawa" and yukawa_step_size != 0.0
    runtime_higgs = (
        sm_constant_higgs(lattice_shape, vev=higgs_vev)
        if collision_mode == "fn_dilation" and yukawa_step_size != 0.0
        else None
    )
    cache_higgs = sm_constant_higgs((1, 1, 1), vev=higgs_vev) if needs_effective_cache else None
    quark_path_readouts = (
        sm_family_recirculated_quark_path_readouts(
            lambda_rec=lambda_rec,
            charges=charges,
            coefficients=coefficients,
        )
        if collision_mode == "fn_dilation"
        else None
    )
    collision_cache = (
        sm_family_yukawa_collision_cache(
            cache_higgs,
            step_size=yukawa_step_size,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=active_lepton_yukawas,
            assume_uniform=True,
            use_unitary_gauge_blocks=True,
        )
        if cache_higgs is not None
        else None
    )
    return QCASMCalibratedRolloutConfig(
        config=QCASMRolloutConfig(
            links=links,
            higgs=runtime_higgs,
            yukawa_step_size=yukawa_step_size,
            collision_mode=collision_mode,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=active_lepton_yukawas,
            quark_path_readouts=quark_path_readouts,
            family_yukawa_collision_cache=collision_cache,
            yukawa_collision_strategy=yukawa_collision_strategy,
            stream_mode=resolved_stream_mode,
            record_observables=record_observables,
            record_density=record_density,
        ),
        verdict=verdict,
    )


def sm_qca_prepare_calibrated_production_rollout(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    initial_state: jnp.ndarray | None = None,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
) -> QCASMCalibratedProductionRolloutSetup:
    """Prepare a compressed production rollout directly from masses and CKM."""

    state = deterministic_qca_family_state(lattice_shape) if initial_state is None else initial_state
    if _qca_spatial_shape(state) != lattice_shape:
        raise ValueError("initial_state spatial shape must match lattice_shape")
    calibrated = sm_qca_rollout_config_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        center_fit_steps=center_fit_steps,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        record_observables=False,
        record_density=False,
        collision_mode="effective_yukawa",
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    calibrated = calibrated._replace(
        config=_lean_cached_effective_yukawa_config(calibrated.config),
    )
    _validate_production_hot_path(calibrated.config)
    setup = sm_qca_prepare_production_rollout_from_config(state, calibrated.config)
    return QCASMCalibratedProductionRolloutSetup(setup=setup, verdict=calibrated.verdict)


def sm_run_qca_calibrated_production_rollout_from_masses_ckm(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    steps: int,
    initial_state: jnp.ndarray | None = None,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
) -> QCASMCalibratedProductionRolloutObservables:
    """Prepare and run a compressed production rollout directly from masses/CKM."""

    setup = sm_qca_prepare_calibrated_production_rollout(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        initial_state=initial_state,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        center_fit_steps=center_fit_steps,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    return sm_run_qca_calibrated_production_rollout_with_observables(setup, steps=steps)


def sm_run_jitted_qca_calibrated_production_rollout_from_masses_ckm(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    steps: int,
    initial_state: jnp.ndarray | None = None,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
    donate_state: bool = True,
) -> QCASMCalibratedProductionRolloutObservables:
    """Prepare and run the JIT compiled production path directly from masses/CKM."""

    setup = sm_qca_prepare_calibrated_production_rollout(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        initial_state=initial_state,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        center_fit_steps=center_fit_steps,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    return sm_run_jitted_qca_calibrated_production_rollout_with_observables(
        setup,
        steps=steps,
        donate_state=donate_state,
    )


def sm_run_jitted_qca_calibrated_carrier_basis_probe(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    steps: int,
    site: tuple[int, int, int] = (0, 0, 0),
    dirac: int = 0,
    sector: str = "Q",
    color: int = 0,
    weak: int = 0,
    family: int = 0,
    internal_copy: int = 0,
    amplitude: complex = 1.0 + 0.0j,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
    donate_state: bool = True,
) -> QCASMCalibratedProductionRolloutObservables:
    """Run a calibrated production rollout from one labeled carrier basis state.

    This is the field-level quark/FN probe: calibration still comes from masses
    and CKM, but the evolved input is a single physical carrier component on the
    production fibre, not the mixed deterministic audit state.
    """

    initial_state = sm_qca_family_carrier_basis_state(
        lattice_shape,
        site=site,
        dirac=dirac,
        sector=sector,
        color=color,
        weak=weak,
        family=family,
        internal_copy=internal_copy,
        amplitude=amplitude,
    )
    return sm_run_jitted_qca_calibrated_production_rollout_from_masses_ckm(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        steps=steps,
        initial_state=initial_state,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=lepton_yukawas,
        center_fit_steps=center_fit_steps,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
        donate_state=donate_state,
    )


def sm_qca_prepared_quark_family_response(
    setup: QCASMCalibratedProductionRolloutSetup,
    *,
    steps: int = 1,
    site: tuple[int, int, int] = (0, 0, 0),
    dirac: int = 0,
    color: int = 0,
    internal_copy: int = 0,
) -> QCASMCalibratedQuarkFamilyResponse:
    """Measure quark response using an already prepared production setup.

    This is the production-core response primitive: it reuses the calibrated
    structured collision cache and production config, initializes three weak-up
    ``Q`` family basis states and three weak-down ``Q`` family basis states,
    runs them through the same JIT-compiled field rollout, and reports the
    complex target ``u_c``/``d_c`` family amplitudes plus their populations.
    """

    config = setup.setup.config
    _validate_production_hot_path(config)
    lattice_shape = _qca_spatial_shape(setup.setup.initial_qca_state.visible_state)
    dtype = setup.setup.initial_qca_state.visible_state.dtype
    up_initial = _sm_qca_family_carrier_basis_stack(
        lattice_shape,
        site=site,
        dirac=dirac,
        sector="Q",
        color=color,
        weak=0,
        internal_copy=internal_copy,
        dtype=dtype,
    )
    down_initial = _sm_qca_family_carrier_basis_stack(
        lattice_shape,
        site=site,
        dirac=dirac,
        sector="Q",
        color=color,
        weak=1,
        internal_copy=internal_copy,
        dtype=dtype,
    )

    def evolve_one(visible_state: jnp.ndarray) -> jnp.ndarray:
        return sm_run_qca_state_rollout(QCASMState(visible_state=visible_state), config, steps=steps).visible_state

    evolve_stack = jax.jit(jax.vmap(evolve_one))
    up_final = evolve_stack(up_initial)
    down_final = evolve_stack(down_initial)
    up_amplitude = _sm_qca_target_family_amplitudes(
        up_final,
        sector="u_c",
        color=color,
        weak=0,
        internal_copy=internal_copy,
    )
    down_amplitude = _sm_qca_target_family_amplitudes(
        down_final,
        sector="d_c",
        color=color,
        weak=0,
        internal_copy=internal_copy,
    )
    up_expected, down_expected = _sm_qca_expected_quark_one_tick_amplitudes(config, lattice_shape)
    return QCASMCalibratedQuarkFamilyResponse(
        verdict=setup.verdict,
        up_amplitude=up_amplitude,
        down_amplitude=down_amplitude,
        up_expected_one_tick_amplitude=up_expected,
        down_expected_one_tick_amplitude=down_expected,
        up_expected_one_tick_residual=jnp.max(jnp.abs(up_amplitude - up_expected)),
        down_expected_one_tick_residual=jnp.max(jnp.abs(down_amplitude - down_expected)),
        up_population=_sm_qca_target_family_populations(
            up_final,
            sector="u_c",
            color=color,
            weak=0,
            internal_copy=internal_copy,
        ),
        down_population=_sm_qca_target_family_populations(
            down_final,
            sector="d_c",
            color=color,
            weak=0,
            internal_copy=internal_copy,
        ),
        up_final_norm=jnp.sum(jnp.abs(up_final) ** 2, axis=(1, 2, 3, 4, 5, 6)),
        down_final_norm=jnp.sum(jnp.abs(down_final) ** 2, axis=(1, 2, 3, 4, 5, 6)),
        steps_completed=steps,
    )


def sm_qca_calibrated_quark_family_response(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    steps: int = 1,
    site: tuple[int, int, int] = (0, 0, 0),
    dirac: int = 0,
    color: int = 0,
    internal_copy: int = 0,
    links: jnp.ndarray | None = None,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str | None = None,
    yukawa_collision_strategy: str = "fast",
) -> QCASMCalibratedQuarkFamilyResponse:
    """Measure quark family response by evolving pure carrier basis states.

    This convenience wrapper prepares the compressed production update from
    masses/CKM, then delegates to ``sm_qca_prepared_quark_family_response`` so
    the measured response uses the same production contract as ordinary
    calibrated rollouts.
    """

    setup = sm_qca_prepare_calibrated_production_rollout(
        up_masses,
        down_masses,
        ckm,
        lattice_shape,
        links=links,
        lambda_rec=lambda_rec,
        charges=charges,
        lepton_yukawas=None,
        center_fit_steps=center_fit_steps,
        yukawa_step_size=yukawa_step_size,
        higgs_vev=higgs_vev,
        stream_mode=stream_mode,
        yukawa_collision_strategy=yukawa_collision_strategy,
    )
    return sm_qca_prepared_quark_family_response(
        setup,
        steps=steps,
        site=site,
        dirac=dirac,
        color=color,
        internal_copy=internal_copy,
    )


def deterministic_qca_family_state(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
    *,
    dtype=DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a deterministic family state for compact rollout smoke tests."""

    return deterministic_sm_family_state(lattice_shape).astype(dtype)


def deterministic_qca_sm_state(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
    *,
    dtype=DEFAULT_COMPLEX_DTYPE,
) -> jnp.ndarray:
    """Return a deterministic SM-internal state for compact rollout smoke tests."""

    return deterministic_sm_state(lattice_shape).astype(dtype)
