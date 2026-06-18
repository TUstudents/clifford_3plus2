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
    fn_effective_yukawa,
)
from clifford_3plus2_d5.qca_smv0.sm_gauge import (
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
    if config.higgs is None or config.yukawa_step_size == 0.0:
        return tuple(arrays)

    _validate_collision_mode(config.collision_mode)
    if config.collision_mode == "fn_dilation":
        arrays.append(config.higgs)
        if config.quark_path_readouts is not None:
            arrays.append(config.quark_path_readouts)
        return tuple(arrays)

    if config.family_yukawa_collision_cache is not None:
        arrays.append(config.family_yukawa_collision_cache)
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

    if config.higgs is None or config.yukawa_step_size == 0.0:
        return streamed, fn_path_aux_state
    if streamed.ndim != 6:
        raise ValueError("Higgs/FN collision requires a family SM state")
    _validate_collision_mode(config.collision_mode)
    if config.collision_mode == "fn_dilation":
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

    def scan_step(
        carry: QCASMState,
        _unused: None,
    ) -> tuple[QCASMState, None]:
        return sm_qca_state_step(carry, config), None

    final_qca_state, _unused = jax.lax.scan(
        scan_step,
        qca_state,
        None,
        length=steps,
    )
    return final_qca_state


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

        def scan_step(
            carry: QCASMState,
            _unused: None,
        ) -> tuple[QCASMState, None]:
            return sm_qca_state_step(carry, config), None

        final_qca_state, _unused = jax.lax.scan(
            scan_step,
            qca_state,
            None,
            length=steps,
        )
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
        used_higgs_fn_collision=config.higgs is not None and config.yukawa_step_size != 0.0,
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
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str = "hop_sum",
    record_observables: bool = True,
    record_density: bool = False,
    collision_mode: str = "fn_dilation",
) -> QCASMRolloutConfig:
    """Return a local Higgs/FN config using the generated center-CP powers."""

    _validate_collision_mode(collision_mode)
    _validate_stream_mode(stream_mode)
    powers = sm_center_powers_from_wilson_flux_rule()
    coefficients = sm_center_cp_order_one_coefficients(sm_center_cp_verdict_magnitudes(), powers=powers)
    quark_yukawas = _quark_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    higgs = sm_constant_higgs(lattice_shape, vev=higgs_vev)
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
            higgs,
            step_size=yukawa_step_size,
            quark_yukawas=quark_yukawas,
            assume_uniform=True,
            use_unitary_gauge_blocks=True,
        )
        if collision_mode == "effective_yukawa" and yukawa_step_size != 0.0
        else None
    )
    return QCASMRolloutConfig(
        higgs=higgs,
        yukawa_step_size=yukawa_step_size,
        collision_mode=collision_mode,
        quark_yukawas=quark_yukawas,
        quark_path_readouts=quark_path_readouts,
        family_yukawa_collision_cache=collision_cache,
        stream_mode=stream_mode,
        record_observables=record_observables,
        record_density=record_density,
    )


def sm_qca_rollout_config_from_masses_ckm(
    up_masses: jnp.ndarray,
    down_masses: jnp.ndarray,
    ckm: jnp.ndarray,
    lattice_shape: tuple[int, int, int],
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    center_fit_steps: int = 0,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    stream_mode: str = "hop_sum",
    record_observables: bool = True,
    record_density: bool = False,
    collision_mode: str = "fn_dilation",
) -> QCASMCalibratedRolloutConfig:
    """Configure a rollout from masses/CKM through the center-CP FN layer."""

    _validate_collision_mode(collision_mode)
    _validate_stream_mode(stream_mode)
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
    higgs = sm_constant_higgs(lattice_shape, vev=higgs_vev)
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
            higgs,
            step_size=yukawa_step_size,
            quark_yukawas=quark_yukawas,
            assume_uniform=True,
            use_unitary_gauge_blocks=True,
        )
        if collision_mode == "effective_yukawa" and yukawa_step_size != 0.0
        else None
    )
    return QCASMCalibratedRolloutConfig(
        config=QCASMRolloutConfig(
            higgs=higgs,
            yukawa_step_size=yukawa_step_size,
            collision_mode=collision_mode,
            quark_yukawas=quark_yukawas,
            quark_path_readouts=quark_path_readouts,
            family_yukawa_collision_cache=collision_cache,
            stream_mode=stream_mode,
            record_observables=record_observables,
            record_density=record_density,
        ),
        verdict=verdict,
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
