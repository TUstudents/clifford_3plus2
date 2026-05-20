"""Shared JAX simulation infrastructure.

This package contains generic array, link, diagnostic, and benchmark helpers.
Physics-specific choices such as the BCC Dirac walk, BCC plaquettes, Wilson
normalization, and gauge-force policy remain in sidecar modules.
"""

from clifford_3plus2_d5.sim.backend import (
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_REAL_DTYPE,
    JaxTiming,
    as_jax,
    as_numpy,
    jax_default_device,
    time_jitted_call,
)
from clifford_3plus2_d5.sim.benchmarks import benchmark_jitted_kernel
from clifford_3plus2_d5.sim.diagnostics import (
    SimulationMetrics,
    all_finite,
    state_transition_metrics,
)
from clifford_3plus2_d5.sim.io import load_json_metadata, save_npz_json
from clifford_3plus2_d5.sim.lattice import (
    Displacement3D,
    LatticeShape3D,
    source_roll,
    source_rolls,
    validate_displacements,
    validate_lattice_shape,
)
from clifford_3plus2_d5.sim.links import (
    jax_constant_link_field,
    jax_identity_link_field,
    jax_transform_link_field,
    validate_link_field,
)
from clifford_3plus2_d5.sim.observables import (
    ObservableMap,
    observations_all_finite,
    pytree_all_finite,
    select_observation_steps,
    stack_observations,
)
from clifford_3plus2_d5.sim.profiling import CallProfile, profile_callable
from clifford_3plus2_d5.sim.runner import (
    GenericRunConfig,
    GenericRunResult,
    recorded_step_indices,
    run_recorded_loop,
    run_recorded_scan,
    validate_run_config,
)
from clifford_3plus2_d5.sim.state import (
    flatten_dirac_internal_state,
    norm_drift,
    state_norm_squared,
    sympy_matrix_to_numpy,
    sympy_scalar_to_complex,
    unflatten_dirac_internal_state,
    zero_jax_dirac_internal_state,
    zero_jax_dirac_state,
)

__all__ = [
    "DEFAULT_COMPLEX_DTYPE",
    "DEFAULT_REAL_DTYPE",
    "Displacement3D",
    "GenericRunConfig",
    "GenericRunResult",
    "JaxTiming",
    "LatticeShape3D",
    "ObservableMap",
    "CallProfile",
    "SimulationMetrics",
    "all_finite",
    "as_jax",
    "as_numpy",
    "benchmark_jitted_kernel",
    "flatten_dirac_internal_state",
    "jax_constant_link_field",
    "jax_default_device",
    "jax_identity_link_field",
    "jax_transform_link_field",
    "load_json_metadata",
    "norm_drift",
    "observations_all_finite",
    "profile_callable",
    "pytree_all_finite",
    "recorded_step_indices",
    "run_recorded_loop",
    "run_recorded_scan",
    "save_npz_json",
    "select_observation_steps",
    "source_roll",
    "source_rolls",
    "stack_observations",
    "state_norm_squared",
    "state_transition_metrics",
    "sympy_matrix_to_numpy",
    "sympy_scalar_to_complex",
    "time_jitted_call",
    "unflatten_dirac_internal_state",
    "validate_displacements",
    "validate_lattice_shape",
    "validate_link_field",
    "validate_run_config",
    "zero_jax_dirac_internal_state",
    "zero_jax_dirac_state",
]
