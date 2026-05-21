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
from clifford_3plus2_d5.spacetime_qca.simulator.kernel_profile import (
    KernelProfileCase,
    default_kernel_profile_cases,
    recommend_kernel_bottleneck,
    run_kernel_profile_case,
    run_spacetime_kernel_profile,
)
from clifford_3plus2_d5.spacetime_qca.simulator.observables import spacetime_observables
from clifford_3plus2_d5.spacetime_qca.simulator.presets import sm_smoke, su2_l_tiny, u1_y_tiny
from clifford_3plus2_d5.spacetime_qca.simulator.profiling import (
    SpacetimeProfileCase,
    default_spacetime_profile_cases,
    recommend_bottleneck,
    run_spacetime_profile,
)
from clifford_3plus2_d5.spacetime_qca.simulator.runner import (
    SpacetimeSimulationResult,
    run_spacetime_simulation,
    save_spacetime_simulation_result,
    spacetime_simulation_summary,
)
from clifford_3plus2_d5.spacetime_qca.simulator.step_breakdown_profile import (
    DEFAULT_FORCE_COMPARISON_CHUNK_SIZES,
    StepBreakdownCase,
    default_step_breakdown_cases,
    recommend_force_chunk_tuning,
    recommend_step_breakdown_bottleneck,
    run_force_chunk_comparison,
    run_spacetime_step_breakdown_profile,
    run_step_breakdown_case,
)

__all__ = [
    "SpacetimeFields",
    "KernelProfileCase",
    "StepBreakdownCase",
    "SpacetimeProfileCase",
    "SpacetimeSimulationConfig",
    "SpacetimeSimulationResult",
    "DEFAULT_FORCE_COMPARISON_CHUNK_SIZES",
    "default_kernel_profile_cases",
    "default_step_breakdown_cases",
    "default_spacetime_profile_cases",
    "recommend_force_chunk_tuning",
    "fields_from_scaling_state",
    "fields_to_scaling_state",
    "recommend_kernel_bottleneck",
    "recommend_step_breakdown_bottleneck",
    "recommend_bottleneck",
    "run_kernel_profile_case",
    "run_force_chunk_comparison",
    "run_spacetime_kernel_profile",
    "run_spacetime_simulation",
    "run_spacetime_step_breakdown_profile",
    "run_spacetime_profile",
    "run_step_breakdown_case",
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
