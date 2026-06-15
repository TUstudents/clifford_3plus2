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

from clifford_3plus2_d5.qca_smv0.bulk_bcc import bcc_dirac_step
from clifford_3plus2_d5.qca_smv0.sm_cp import (
    CenterCPPhenomenologyVerdict,
    CenterCPTextureSeed,
    sm_center_cp_phenomenology_verdict,
    sm_center_cp_order_one_coefficients,
    sm_center_cp_verdict_magnitudes,
    sm_center_powers_from_wilson_flux_rule,
)
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import (
    FamilyLeptonYukawas,
    deterministic_sm_family_state,
    sm_apply_family_yukawa_collision,
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
    sm_free_dirac_internal_step,
    sm_gauged_dirac_step,
)
from clifford_3plus2_d5.qca_smv0.sm_higgs import sm_constant_higgs
from clifford_3plus2_d5.sim.backend import DEFAULT_COMPLEX_DTYPE


class QCASMRolloutConfig(NamedTuple):
    """Configuration for one QCA_SMv0 field rollout."""

    links: jnp.ndarray | None = None
    higgs: jnp.ndarray | None = None
    yukawa_step_size: float = 0.0
    quark_yukawas: FNQuarkYukawas | None = None
    lepton_yukawas: FamilyLeptonYukawas | None = None
    record_density: bool = False


class QCASMRolloutResult(NamedTuple):
    """Result of a compact QCA_SMv0 rollout."""

    final_state: jnp.ndarray
    norm_history: jnp.ndarray
    max_density_history: jnp.ndarray
    density_history: jnp.ndarray
    used_gauge_links: bool
    used_higgs_fn_collision: bool
    steps_completed: int


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


def _family_stream_step(state: jnp.ndarray, links: jnp.ndarray | None) -> jnp.ndarray:
    by_family = jnp.moveaxis(state, -1, 0)
    if links is None:
        stepped = jax.vmap(sm_free_dirac_internal_step)(by_family)
    else:
        stepped = jax.vmap(lambda family_state: sm_gauged_dirac_step(family_state, links))(by_family)
    return jnp.moveaxis(stepped, 0, -1)


def sm_qca_rollout_step(state: jnp.ndarray, config: QCASMRolloutConfig = QCASMRolloutConfig()) -> jnp.ndarray:
    """Apply one compact QCA_SMv0 tick.

    The tick is gauge/BCC streaming followed by an optional site-local
    three-family Higgs/FN collision.  Dynamic gauge and Higgs field evolution
    intentionally remain in the heavier production-tick modules.
    """

    _qca_spatial_shape(state)
    if state.ndim == 4:
        if config.links is not None:
            raise ValueError("gauge links require an SM-internal state")
        streamed = bcc_dirac_step(state)
    elif state.ndim == 5:
        streamed = sm_free_dirac_internal_step(state) if config.links is None else sm_gauged_dirac_step(state, config.links)
    else:
        streamed = _family_stream_step(state, config.links)

    if config.higgs is None or config.yukawa_step_size == 0.0:
        return streamed
    if streamed.ndim != 6:
        raise ValueError("Higgs/FN collision requires a family SM state")
    return sm_apply_family_yukawa_collision(
        streamed,
        config.higgs,
        step_size=config.yukawa_step_size,
        quark_yukawas=config.quark_yukawas,
        lepton_yukawas=config.lepton_yukawas,
    )


def sm_run_qca_rollout(
    config: QCASMRolloutConfig,
    initial_state: jnp.ndarray,
    steps: int,
) -> QCASMRolloutResult:
    """Run a compact QCA_SMv0 field rollout and record basic observables."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    _qca_spatial_shape(initial_state)
    state = initial_state
    norm_history = [sm_qca_total_norm(state)]
    max_density_history = [jnp.max(sm_qca_density(state))]
    density_history = [sm_qca_density(state)] if config.record_density else []
    for _ in range(steps):
        state = sm_qca_rollout_step(state, config)
        density = sm_qca_density(state)
        norm_history.append(jnp.sum(density))
        max_density_history.append(jnp.max(density))
        if config.record_density:
            density_history.append(density)
    spatial_shape = _qca_spatial_shape(initial_state)
    empty_density = jnp.zeros((0, *spatial_shape), dtype=jnp.real(initial_state).dtype)
    return QCASMRolloutResult(
        final_state=state,
        norm_history=jnp.stack(norm_history),
        max_density_history=jnp.stack(max_density_history),
        density_history=jnp.stack(density_history) if density_history else empty_density,
        used_gauge_links=config.links is not None,
        used_higgs_fn_collision=config.higgs is not None and config.yukawa_step_size != 0.0,
        steps_completed=steps,
    )


def sm_qca_center_cp_rollout_config(
    lattice_shape: tuple[int, int, int],
    *,
    lambda_rec: float = FN_LAMBDA_WOLFENSTEIN,
    charges: FNQuarkCharges = DEFAULT_FN_QUARK_CHARGES,
    yukawa_step_size: float = 0.01,
    higgs_vev: float = 1.0,
    record_density: bool = False,
) -> QCASMRolloutConfig:
    """Return a local Higgs/FN config using the generated center-CP powers."""

    powers = sm_center_powers_from_wilson_flux_rule()
    coefficients = sm_center_cp_order_one_coefficients(sm_center_cp_verdict_magnitudes(), powers=powers)
    quark_yukawas = _quark_yukawas_from_coefficients(coefficients, lambda_rec=lambda_rec, charges=charges)
    return QCASMRolloutConfig(
        higgs=sm_constant_higgs(lattice_shape, vev=higgs_vev),
        yukawa_step_size=yukawa_step_size,
        quark_yukawas=quark_yukawas,
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
    record_density: bool = False,
) -> QCASMCalibratedRolloutConfig:
    """Configure a rollout from masses/CKM through the center-CP FN layer."""

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
    return QCASMCalibratedRolloutConfig(
        config=QCASMRolloutConfig(
            higgs=sm_constant_higgs(lattice_shape, vev=higgs_vev),
            yukawa_step_size=yukawa_step_size,
            quark_yukawas=quark_yukawas,
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
