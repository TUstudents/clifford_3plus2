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
    "JaxTiming",
    "LatticeShape3D",
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
    "norm_drift",
    "source_roll",
    "source_rolls",
    "state_norm_squared",
    "state_transition_metrics",
    "sympy_matrix_to_numpy",
    "sympy_scalar_to_complex",
    "time_jitted_call",
    "unflatten_dirac_internal_state",
    "validate_displacements",
    "validate_lattice_shape",
    "validate_link_field",
    "zero_jax_dirac_internal_state",
    "zero_jax_dirac_state",
]
