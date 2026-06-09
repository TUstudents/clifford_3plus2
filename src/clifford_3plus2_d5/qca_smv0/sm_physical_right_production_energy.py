"""Energy monitor for the QCA_SMv0 physical-right production rollout.

Stage 24 records the energy components already present in earlier stages over
the Stage 21/22 physical-right production dynamics:

- pure SM gauge Hamiltonian density;
- Higgs Hamiltonian density;
- physical-right BCC streaming bilinear;
- local three-family Yukawa energy.

This is a monitor, not a new dynamics rule.  It does not claim exact
conservation for the hybrid quantum/classical production tick and does not add
boundary conditions, quantized registers, or Gauss projection.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_dynamics import sm_gauge_hamiltonian_density
from clifford_3plus2_d5.qca_smv0.sm_family_higgs import FamilyLeptonYukawas
from clifford_3plus2_d5.qca_smv0.sm_family_production_tick import (
    sm_zero_family_lepton_yukawas,
    sm_zero_quark_yukawas,
)
from clifford_3plus2_d5.qca_smv0.sm_fermion_higgs import sm_yukawa_energy_density
from clifford_3plus2_d5.qca_smv0.sm_fn import FNQuarkYukawas
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import (
    DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    HiggsDynamicsParameters,
    sm_higgs_hamiltonian_density,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_current import (
    sm_physical_right_streaming_energy_density,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_gauss import (
    sm_physical_right_production_vacuum_state,
)
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.sim.observables import ObservableMap, observations_all_finite, stack_observations
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionEnergyDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 24 physical-right production energy monitor."""

    vacuum_initial_total_energy_abs: jnp.ndarray
    vacuum_final_total_energy_abs: jnp.ndarray
    deterministic_initial_total_energy: jnp.ndarray
    deterministic_final_total_energy: jnp.ndarray
    deterministic_total_energy_delta_abs: jnp.ndarray
    deterministic_max_total_energy_abs: jnp.ndarray
    deterministic_gauge_energy_positive: jnp.ndarray
    deterministic_higgs_energy_positive: jnp.ndarray
    deterministic_streaming_energy_abs: jnp.ndarray
    deterministic_yukawa_energy_abs: jnp.ndarray
    zero_yukawa_final_total_energy_difference_abs: jnp.ndarray
    rollout_family_norm_drift: jnp.ndarray
    history_count: jnp.ndarray
    history_all_finite: jnp.ndarray


def sm_physical_right_production_energy_observables(
    state: PhysicalRightProductionRolloutState,
    *,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
) -> ObservableMap:
    """Return scalar energy components for a physical-right production state."""

    gauge = sm_gauge_hamiltonian_density(state.sm_links, state.sm_momenta, beta=beta)
    higgs = sm_higgs_hamiltonian_density(
        state.higgs,
        state.higgs_momenta,
        state.higgs_links,
        parameters=parameters,
    )
    streaming = sm_physical_right_streaming_energy_density(state.family_state, state.sm_links)
    yukawa = sm_yukawa_energy_density(
        state.family_state,
        state.higgs,
        quark_yukawas=quark_yukawas,
        lepton_yukawas=lepton_yukawas,
    )
    total = gauge + higgs + streaming + yukawa
    return {
        "family_norm": state_norm_squared(state.family_state),
        "gauge_energy": gauge,
        "higgs_energy": higgs,
        "streaming_energy": streaming,
        "yukawa_energy": yukawa,
        "total_energy": total,
    }


def sm_physical_right_production_energy_history(
    initial_state: PhysicalRightProductionRolloutState | None = None,
    *,
    steps: int = 2,
    step_size: float = 0.001,
    beta: float = 1.0,
    parameters: HiggsDynamicsParameters = DEFAULT_HIGGS_DYNAMICS_PARAMETERS,
    quark_yukawas: FNQuarkYukawas | None = None,
    lepton_yukawas: FamilyLeptonYukawas | None = None,
    wilson_epsilon: float = 1e-3,
    higgs_force_epsilon: float = 1e-3,
    fermion_current_epsilon: float = 3e-2,
) -> tuple[PhysicalRightProductionRolloutState, dict[str, jnp.ndarray]]:
    """Run a short production rollout and record energy components."""

    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    state = initial_state or sm_physical_right_production_initial_state()
    observations: list[ObservableMap] = [
        sm_physical_right_production_energy_observables(
            state,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
        ),
    ]
    for _ in range(steps):
        state = sm_physical_right_production_step(
            state,
            step_size=step_size,
            beta=beta,
            parameters=parameters,
            quark_yukawas=quark_yukawas,
            lepton_yukawas=lepton_yukawas,
            wilson_epsilon=wilson_epsilon,
            higgs_force_epsilon=higgs_force_epsilon,
            fermion_current_epsilon=fermion_current_epsilon,
        )
        observations.append(
            sm_physical_right_production_energy_observables(
                state,
                beta=beta,
                parameters=parameters,
                quark_yukawas=quark_yukawas,
                lepton_yukawas=lepton_yukawas,
            ),
        )
    return state, stack_observations(tuple(observations))


def sm_physical_right_production_energy_diagnostics() -> PhysicalRightProductionEnergyDiagnostics:
    """Return focused Stage 24 physical-right production energy diagnostics."""

    steps = 2
    step_size = 0.001
    vacuum_initial = sm_physical_right_production_vacuum_state()
    vacuum_final, vacuum_history = sm_physical_right_production_energy_history(
        vacuum_initial,
        steps=steps,
        step_size=step_size,
    )
    deterministic_initial = sm_physical_right_production_initial_state()
    deterministic_final, deterministic_history = sm_physical_right_production_energy_history(
        deterministic_initial,
        steps=steps,
        step_size=step_size,
    )
    zero_quarks = sm_zero_quark_yukawas()
    zero_leptons = sm_zero_family_lepton_yukawas()
    _, zero_yukawa_history = sm_physical_right_production_energy_history(
        deterministic_initial,
        steps=steps,
        step_size=step_size,
        quark_yukawas=zero_quarks,
        lepton_yukawas=zero_leptons,
    )
    total = deterministic_history["total_energy"]
    family_norms = deterministic_history["family_norm"]

    return PhysicalRightProductionEnergyDiagnostics(
        vacuum_initial_total_energy_abs=jnp.abs(vacuum_history["total_energy"][0]),
        vacuum_final_total_energy_abs=jnp.abs(vacuum_history["total_energy"][-1]),
        deterministic_initial_total_energy=total[0],
        deterministic_final_total_energy=total[-1],
        deterministic_total_energy_delta_abs=jnp.abs(total[-1] - total[0]),
        deterministic_max_total_energy_abs=jnp.max(jnp.abs(total)),
        deterministic_gauge_energy_positive=jnp.min(deterministic_history["gauge_energy"]),
        deterministic_higgs_energy_positive=jnp.min(deterministic_history["higgs_energy"]),
        deterministic_streaming_energy_abs=jnp.max(jnp.abs(deterministic_history["streaming_energy"])),
        deterministic_yukawa_energy_abs=jnp.max(jnp.abs(deterministic_history["yukawa_energy"])),
        zero_yukawa_final_total_energy_difference_abs=jnp.abs(total[-1] - zero_yukawa_history["total_energy"][-1]),
        rollout_family_norm_drift=jnp.abs(state_norm_squared(deterministic_final.family_state) - family_norms[0]),
        history_count=jnp.asarray(total.shape[0], dtype=jnp.int32),
        history_all_finite=observations_all_finite(vacuum_history) & observations_all_finite(deterministic_history),
    )
