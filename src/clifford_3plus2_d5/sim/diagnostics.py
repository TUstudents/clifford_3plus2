"""Generic numerical diagnostics for simulation sidecars."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from clifford_3plus2_d5.sim.state import norm_drift, state_norm_squared


@dataclass(frozen=True)
class SimulationMetrics:
    """Small generic metrics payload for simulation smoke tests."""

    initial_norm_squared: float
    final_norm_squared: float
    norm_drift: float
    all_finite: bool


def all_finite(*arrays: jnp.ndarray) -> bool:
    """Return whether all provided arrays contain only finite values."""

    return all(bool(jnp.all(jnp.isfinite(array))) for array in arrays)


def state_transition_metrics(before: jnp.ndarray, after: jnp.ndarray) -> SimulationMetrics:
    """Return basic finite-state transition diagnostics."""

    initial = state_norm_squared(before)
    final = state_norm_squared(after)
    return SimulationMetrics(
        initial_norm_squared=float(initial),
        final_norm_squared=float(final),
        norm_drift=float(norm_drift(before, after)),
        all_finite=all_finite(before, after),
    )
