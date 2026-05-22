"""Prototype tiny-lattice runner for the spacetime-QCA coupled lab.

This is the experimental Session 46 runner.  It intentionally stays in the
``lab`` package: deterministic tiny-lattice data, Python-level stepping, and
debug-friendly observable recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.io import save_npz_json
from clifford_3plus2_d5.spacetime_qca.jax_coupled_higgs import (
    HiggsCoupledSector,
    YukawaUpdateMode,
)
from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    ScalingInitialState,
    ScalingRunConfig,
    ScalingSnapshot,
    jax_advance_scaling_fields,
    jax_default_scaling_initial_state,
    jax_scaling_snapshot,
)


@dataclass(frozen=True)
class SimulationRunConfig:
    """Memory-safe configuration for the Session 46 tiny runner."""

    lattice_shape: tuple[int, int, int] = (1, 1, 1)
    sector: HiggsCoupledSector = "u1_y"
    steps: int = 4
    record_every: int = 1
    step_size: float = 0.0025
    matter_coupling: float = 1.0
    higgs_coupling: float = 0.0
    yukawa_coupling: float = 1.0
    beta: float = 1.0
    vev_squared: float = 1.0
    quartic: float = 1.0
    force_epsilon: float = 1e-3
    current_epsilon: float = 1e-3
    higgs_current_epsilon: float = 1e-3
    yukawa_mode: YukawaUpdateMode = "unitary"
    gauss_projection_steps: int = 0
    gauss_projection_step_size: float = 0.0
    dtype_label: str = "complex64"
    label: str = "tiny_spacetime_qca"
    description: str = "Session 46 deterministic tiny-lattice run"


@dataclass(frozen=True)
class SimulationHistory:
    """Recorded scalar observable history for a tiny runner execution."""

    step_indices: jnp.ndarray
    fermion_norm: jnp.ndarray
    gauge_hamiltonian_density: jnp.ndarray
    higgs_total_energy: jnp.ndarray
    higgs_energy_density_mean: jnp.ndarray
    gauss_residual_norm: jnp.ndarray
    yukawa_norm_drift: jnp.ndarray
    total_energy_proxy: jnp.ndarray


@dataclass(frozen=True)
class SimulationResult:
    """Complete Session 46 tiny-run result."""

    config: SimulationRunConfig
    initial_fields: ScalingInitialState
    final_fields: ScalingInitialState
    history: SimulationHistory
    all_finite: jnp.ndarray
    metadata: dict[str, Any]


def _validate_runner_config(config: SimulationRunConfig) -> None:
    if len(config.lattice_shape) != 3 or any(size <= 0 for size in config.lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {config.lattice_shape}")
    if config.steps < 0:
        raise ValueError(f"steps must be nonnegative, got {config.steps}")
    if config.record_every <= 0:
        raise ValueError(f"record_every must be positive, got {config.record_every}")
    if config.gauss_projection_steps < 0:
        raise ValueError(f"gauss_projection_steps must be nonnegative, got {config.gauss_projection_steps}")
    if config.gauss_projection_step_size < 0:
        raise ValueError(
            f"gauss_projection_step_size must be nonnegative, got {config.gauss_projection_step_size}",
        )


def _scaling_config(config: SimulationRunConfig) -> ScalingRunConfig:
    return ScalingRunConfig(
        lattice_shape=config.lattice_shape,
        sector=config.sector,
        step_size=config.step_size,
        matter_coupling=config.matter_coupling,
        higgs_coupling=config.higgs_coupling,
        yukawa_coupling=config.yukawa_coupling,
        beta=config.beta,
        vev_squared=config.vev_squared,
        quartic=config.quartic,
        force_epsilon=config.force_epsilon,
        current_epsilon=config.current_epsilon,
        higgs_current_epsilon=config.higgs_current_epsilon,
        yukawa_mode=config.yukawa_mode,
        gauss_projection_steps=config.gauss_projection_steps,
        gauss_projection_step_size=config.gauss_projection_step_size,
    )


def _snapshot_tuple(snapshot: ScalingSnapshot) -> tuple[jnp.ndarray, ...]:
    return (
        snapshot.fermion_norm,
        snapshot.gauge_hamiltonian_density,
        snapshot.higgs_total_energy,
        snapshot.higgs_energy_density_mean,
        snapshot.gauss_residual_norm,
        snapshot.yukawa_norm_drift,
        snapshot.total_energy_proxy,
    )


def _history_from_snapshots(
    step_indices: tuple[int, ...],
    snapshots: tuple[ScalingSnapshot, ...],
) -> SimulationHistory:
    rows = tuple(_snapshot_tuple(snapshot) for snapshot in snapshots)
    columns = tuple(jnp.asarray([row[index] for row in rows]) for index in range(7))
    return SimulationHistory(
        step_indices=jnp.asarray(step_indices, dtype=jnp.int32),
        fermion_norm=columns[0],
        gauge_hamiltonian_density=columns[1],
        higgs_total_energy=columns[2],
        higgs_energy_density_mean=columns[3],
        gauss_residual_norm=columns[4],
        yukawa_norm_drift=columns[5],
        total_energy_proxy=columns[6],
    )


def _history_arrays(history: SimulationHistory) -> tuple[jnp.ndarray, ...]:
    return (
        history.step_indices,
        history.fermion_norm,
        history.gauge_hamiltonian_density,
        history.higgs_total_energy,
        history.higgs_energy_density_mean,
        history.gauss_residual_norm,
        history.yukawa_norm_drift,
        history.total_energy_proxy,
    )


def _fields_arrays(fields: ScalingInitialState) -> tuple[jnp.ndarray, ...]:
    return (
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
    )


def _all_arrays_finite(*arrays: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(array)) for array in arrays]))


def simulation_metadata(config: SimulationRunConfig) -> dict[str, Any]:
    """Return JSON-safe metadata for a tiny runner config."""

    metadata = asdict(config)
    metadata["lattice_shape"] = tuple(config.lattice_shape)
    metadata["runner"] = "spacetime_qca.lab.tiny_runner"
    metadata["format"] = "npz-plus-json-v1"
    return metadata


def run_simulation(config: SimulationRunConfig | None = None) -> SimulationResult:
    """Run a deterministic tiny-lattice coupled simulation."""

    if config is None:
        config = SimulationRunConfig()
    _validate_runner_config(config)
    scaling_config = _scaling_config(config)

    initial_fields = jax_default_scaling_initial_state(scaling_config)
    fields = initial_fields
    step_indices: list[int] = [0]
    snapshots: list[ScalingSnapshot] = [
        jax_scaling_snapshot(
            fields.state,
            fields.links,
            fields.momenta,
            fields.phi,
            fields.higgs_momentum,
            fields.higgs_links,
            config=scaling_config,
            reference_state=initial_fields.state,
        ),
    ]

    for step_index in range(1, config.steps + 1):
        fields = jax_advance_scaling_fields(fields, scaling_config)
        should_record = step_index % config.record_every == 0 or step_index == config.steps
        if should_record:
            step_indices.append(step_index)
            snapshots.append(
                jax_scaling_snapshot(
                    fields.state,
                    fields.links,
                    fields.momenta,
                    fields.phi,
                    fields.higgs_momentum,
                    fields.higgs_links,
                    config=scaling_config,
                    reference_state=initial_fields.state,
                ),
            )

    history = _history_from_snapshots(tuple(step_indices), tuple(snapshots))
    all_finite = _all_arrays_finite(*_fields_arrays(fields), *_history_arrays(history))
    return SimulationResult(
        config=config,
        initial_fields=initial_fields,
        final_fields=fields,
        history=history,
        all_finite=all_finite,
        metadata=simulation_metadata(config),
    )


def simulation_summary(result: SimulationResult) -> dict[str, Any]:
    """Return a compact JSON-safe summary for console output."""

    history = result.history
    return {
        "label": result.config.label,
        "lattice_shape": tuple(result.config.lattice_shape),
        "sector": result.config.sector,
        "steps": result.config.steps,
        "recorded_steps": tuple(int(step) for step in np.asarray(history.step_indices)),
        "yukawa_mode": result.config.yukawa_mode,
        "all_finite": bool(result.all_finite),
        "initial_fermion_norm": float(np.asarray(history.fermion_norm[0])),
        "final_fermion_norm": float(np.asarray(history.fermion_norm[-1])),
        "final_gauss_residual_norm": float(np.asarray(history.gauss_residual_norm[-1])),
        "final_total_energy_proxy": float(np.asarray(history.total_energy_proxy[-1])),
    }


def save_simulation_result(result: SimulationResult, path: str | Path) -> tuple[Path, Path]:
    """Save a result as ``.npz`` arrays plus a JSON metadata sidecar."""

    arrays = {
        "step_indices": np.asarray(result.history.step_indices),
        "fermion_norm": np.asarray(result.history.fermion_norm),
        "gauge_hamiltonian_density": np.asarray(result.history.gauge_hamiltonian_density),
        "higgs_total_energy": np.asarray(result.history.higgs_total_energy),
        "higgs_energy_density_mean": np.asarray(result.history.higgs_energy_density_mean),
        "gauss_residual_norm": np.asarray(result.history.gauss_residual_norm),
        "yukawa_norm_drift": np.asarray(result.history.yukawa_norm_drift),
        "total_energy_proxy": np.asarray(result.history.total_energy_proxy),
        "final_state": np.asarray(result.final_fields.state),
        "final_links": np.asarray(result.final_fields.links),
        "final_momenta": np.asarray(result.final_fields.momenta),
        "final_phi": np.asarray(result.final_fields.phi),
        "final_higgs_momentum": np.asarray(result.final_fields.higgs_momentum),
        "final_higgs_links": np.asarray(result.final_fields.higgs_links),
    }
    json_payload = {
        "metadata": result.metadata,
        "summary": simulation_summary(result),
    }
    return save_npz_json(arrays, json_payload, path)
