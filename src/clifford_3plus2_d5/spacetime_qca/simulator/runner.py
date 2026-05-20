"""Scan-backed spacetime-QCA simulator runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from clifford_3plus2_d5.sim.io import save_npz_json
from clifford_3plus2_d5.sim.runner import GenericRunConfig, run_recorded_scan
from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    jax_advance_scaling_fields,
    jax_default_scaling_initial_state,
)
from clifford_3plus2_d5.spacetime_qca.simulator.config import (
    SpacetimeSimulationConfig,
    scaling_config_from_spacetime_config,
    spacetime_simulation_metadata,
    validate_spacetime_simulation_config,
)
from clifford_3plus2_d5.spacetime_qca.simulator.fields import (
    SpacetimeFields,
    fields_from_scaling_state,
    fields_to_scaling_state,
)
from clifford_3plus2_d5.spacetime_qca.simulator.observables import spacetime_observables


@dataclass(frozen=True)
class SpacetimeSimulationResult:
    """Recorded scan-backed spacetime-QCA simulation result."""

    config: SpacetimeSimulationConfig
    initial_fields: SpacetimeFields
    final_fields: SpacetimeFields
    step_indices: jnp.ndarray
    observations: dict[str, jnp.ndarray]
    all_finite: jnp.ndarray
    metadata: dict[str, Any]


def run_spacetime_simulation(
    config: SpacetimeSimulationConfig | None = None,
) -> SpacetimeSimulationResult:
    """Run the main scan-backed spacetime-QCA simulator."""

    if config is None:
        config = SpacetimeSimulationConfig()
    validate_spacetime_simulation_config(config)
    scaling_config = scaling_config_from_spacetime_config(config)
    initial_scaling_fields = jax_default_scaling_initial_state(scaling_config)
    initial_fields = fields_from_scaling_state(initial_scaling_fields)

    def step_fn(fields: SpacetimeFields) -> SpacetimeFields:
        next_scaling_fields = jax_advance_scaling_fields(fields_to_scaling_state(fields), scaling_config)
        return fields_from_scaling_state(next_scaling_fields)

    def observe_fn(fields: SpacetimeFields) -> dict[str, jnp.ndarray]:
        return spacetime_observables(fields, config=scaling_config, reference_state=initial_fields.state)

    generic_result = run_recorded_scan(
        initial_fields,
        step_fn,
        observe_fn,
        GenericRunConfig(steps=config.steps, record_every=config.record_every, use_jit=config.use_jit),
    )
    return SpacetimeSimulationResult(
        config=config,
        initial_fields=initial_fields,
        final_fields=generic_result.final_state,
        step_indices=generic_result.step_indices,
        observations=generic_result.observations,
        all_finite=generic_result.all_finite,
        metadata=spacetime_simulation_metadata(config),
    )


def spacetime_simulation_summary(result: SpacetimeSimulationResult) -> dict[str, Any]:
    """Return a compact JSON-safe summary for a main simulator result."""

    observations = result.observations
    return {
        "label": result.config.label,
        "lattice_shape": tuple(result.config.lattice_shape),
        "sector": result.config.sector,
        "steps": result.config.steps,
        "recorded_steps": tuple(int(step) for step in np.asarray(result.step_indices)),
        "yukawa_mode": result.config.yukawa_mode,
        "use_jit": result.config.use_jit,
        "all_finite": bool(result.all_finite),
        "initial_fermion_norm": float(np.asarray(observations["fermion_norm"][0])),
        "final_fermion_norm": float(np.asarray(observations["fermion_norm"][-1])),
        "final_gauss_residual_norm": float(np.asarray(observations["gauss_residual_norm"][-1])),
        "final_total_energy_proxy": float(np.asarray(observations["total_energy_proxy"][-1])),
    }


def save_spacetime_simulation_result(
    result: SpacetimeSimulationResult,
    path: str | Path,
) -> tuple[Path, Path]:
    """Save main simulator observables and final fields to ``.npz`` plus JSON."""

    arrays = {
        "step_indices": np.asarray(result.step_indices),
        **{key: np.asarray(value) for key, value in result.observations.items()},
        "final_state": np.asarray(result.final_fields.state),
        "final_links": np.asarray(result.final_fields.links),
        "final_momenta": np.asarray(result.final_fields.momenta),
        "final_phi": np.asarray(result.final_fields.phi),
        "final_higgs_momentum": np.asarray(result.final_fields.higgs_momentum),
        "final_higgs_links": np.asarray(result.final_fields.higgs_links),
    }
    payload = {
        "metadata": result.metadata,
        "summary": spacetime_simulation_summary(result),
    }
    return save_npz_json(arrays, payload, path)
