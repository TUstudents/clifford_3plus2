"""Physics-specific observable extraction for spacetime-QCA simulator runs."""

from __future__ import annotations

import jax.numpy as jnp

from clifford_3plus2_d5.spacetime_qca.jax_scaling import (
    ScalingRunConfig,
    jax_scaling_snapshot,
)
from clifford_3plus2_d5.spacetime_qca.simulator.fields import SpacetimeFields


def spacetime_observables(
    fields: SpacetimeFields,
    *,
    config: ScalingRunConfig,
    reference_state: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Return scalar observables for one simulator field state."""

    snapshot = jax_scaling_snapshot(
        fields.state,
        fields.links,
        fields.momenta,
        fields.phi,
        fields.higgs_momentum,
        fields.higgs_links,
        config=config,
        reference_state=reference_state,
    )
    return {
        "fermion_norm": snapshot.fermion_norm,
        "gauge_hamiltonian_density": snapshot.gauge_hamiltonian_density,
        "higgs_total_energy": snapshot.higgs_total_energy,
        "higgs_energy_density_mean": snapshot.higgs_energy_density_mean,
        "gauss_residual_norm": snapshot.gauss_residual_norm,
        "yukawa_norm_drift": snapshot.yukawa_norm_drift,
        "total_energy_proxy": snapshot.total_energy_proxy,
    }
