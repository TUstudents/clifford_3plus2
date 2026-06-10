"""Finite-horizon echo-spectrum audit for the production map.

Stage 34 compares the Stage 32 echo Gram across two trajectory lengths.  This
turns the local tangent metric into a first finite-time stability observable:
the inverse-pulled echo gains should stay finite, nonzero, and locally stable
as the production trajectory is extended from one tick to two ticks.

This is a diagnostic of the current discrete production map, not a continuum
Lyapunov theorem or a new dynamics rule.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_echo_gram import (
    sm_physical_right_production_echo_gram_diagnostics,
)


class PhysicalRightProductionEchoHorizonDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 34 finite-horizon echo spectrum."""

    short_steps: jnp.ndarray
    long_steps: jnp.ndarray
    perturbation_size: jnp.ndarray
    short_base_roundtrip_residual: jnp.ndarray
    long_base_roundtrip_residual: jnp.ndarray
    short_min_gain: jnp.ndarray
    short_max_gain: jnp.ndarray
    long_min_gain: jnp.ndarray
    long_max_gain: jnp.ndarray
    min_gain_growth_ratio: jnp.ndarray
    max_gain_growth_ratio: jnp.ndarray
    max_abs_log_gain_growth_per_tick: jnp.ndarray
    condition_number_delta: jnp.ndarray
    offdiag_correlation_delta: jnp.ndarray
    max_link_unitarity_residual: jnp.ndarray


def _min_gain(eigenvalue: jnp.ndarray, epsilon: jnp.ndarray) -> jnp.ndarray:
    nonnegative_eigenvalue = jnp.maximum(
        eigenvalue,
        jnp.asarray(0.0, dtype=eigenvalue.dtype),
    )
    return jnp.sqrt(nonnegative_eigenvalue) / epsilon


def sm_physical_right_production_echo_horizon_diagnostics(
    *,
    short_steps: int = 1,
    long_steps: int = 2,
    step_size: float = 1e-3,
    epsilon: float = 1e-4,
) -> PhysicalRightProductionEchoHorizonDiagnostics:
    """Return Stage 34 finite-horizon echo-spectrum diagnostics."""

    if short_steps < 1:
        raise ValueError(f"short_steps must be positive, got {short_steps}")
    if long_steps <= short_steps:
        raise ValueError(f"long_steps must exceed short_steps, got {long_steps} <= {short_steps}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    short = sm_physical_right_production_echo_gram_diagnostics(
        steps=short_steps,
        step_size=step_size,
        epsilon=epsilon,
    )
    long = sm_physical_right_production_echo_gram_diagnostics(
        steps=long_steps,
        step_size=step_size,
        epsilon=epsilon,
    )
    eps = short.perturbation_size
    short_min_gain = _min_gain(short.gram_min_eigenvalue, eps)
    short_max_gain = _min_gain(short.gram_max_eigenvalue, eps)
    long_min_gain = _min_gain(long.gram_min_eigenvalue, eps)
    long_max_gain = _min_gain(long.gram_max_eigenvalue, eps)
    min_gain_ratio = long_min_gain / jnp.maximum(
        short_min_gain,
        jnp.asarray(1e-30, dtype=short_min_gain.dtype),
    )
    max_gain_ratio = long_max_gain / jnp.maximum(
        short_max_gain,
        jnp.asarray(1e-30, dtype=short_max_gain.dtype),
    )
    horizon_delta = jnp.asarray(long_steps - short_steps, dtype=jnp.float32)
    gain_ratios = jnp.asarray([min_gain_ratio, max_gain_ratio])
    max_log_gain_growth = jnp.max(jnp.abs(jnp.log(gain_ratios))) / horizon_delta

    return PhysicalRightProductionEchoHorizonDiagnostics(
        short_steps=jnp.asarray(short_steps, dtype=jnp.int32),
        long_steps=jnp.asarray(long_steps, dtype=jnp.int32),
        perturbation_size=eps,
        short_base_roundtrip_residual=short.base_roundtrip_residual,
        long_base_roundtrip_residual=long.base_roundtrip_residual,
        short_min_gain=short_min_gain,
        short_max_gain=short_max_gain,
        long_min_gain=long_min_gain,
        long_max_gain=long_max_gain,
        min_gain_growth_ratio=min_gain_ratio,
        max_gain_growth_ratio=max_gain_ratio,
        max_abs_log_gain_growth_per_tick=max_log_gain_growth,
        condition_number_delta=jnp.abs(long.gram_condition_number - short.gram_condition_number),
        offdiag_correlation_delta=jnp.abs(long.max_offdiag_correlation - short.max_offdiag_correlation),
        max_link_unitarity_residual=jnp.max(
            jnp.asarray(
                [
                    short.max_inverse_sm_link_unitarity_residual,
                    short.max_inverse_higgs_link_unitarity_residual,
                    long.max_inverse_sm_link_unitarity_residual,
                    long.max_inverse_higgs_link_unitarity_residual,
                ],
            ),
        ),
    )
