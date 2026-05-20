"""Production-facing spacetime-QCA simulator API."""

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
from clifford_3plus2_d5.spacetime_qca.simulator.presets import sm_smoke, su2_l_tiny, u1_y_tiny
from clifford_3plus2_d5.spacetime_qca.simulator.runner import (
    SpacetimeSimulationResult,
    run_spacetime_simulation,
    save_spacetime_simulation_result,
    spacetime_simulation_summary,
)

__all__ = [
    "SpacetimeFields",
    "SpacetimeSimulationConfig",
    "SpacetimeSimulationResult",
    "fields_from_scaling_state",
    "fields_to_scaling_state",
    "run_spacetime_simulation",
    "save_spacetime_simulation_result",
    "scaling_config_from_spacetime_config",
    "sm_smoke",
    "spacetime_observables",
    "spacetime_simulation_metadata",
    "spacetime_simulation_summary",
    "su2_l_tiny",
    "u1_y_tiny",
    "validate_spacetime_simulation_config",
]
