"""Profiling smoke test for the local-force production step.

Stage 40 records bounded timing metadata for the Stage 37/38 local-force
production path.  Timings are machine-dependent and are not used as physics or
performance claims; the certificate only checks that eager and JIT-profiled
paths complete, remain finite, and agree on the output state.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.qca_smv0.sm_gauge import sm_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_higgs_dynamics import sm_higgs_link_unitarity_residual
from clifford_3plus2_d5.qca_smv0.sm_physical_right_production_rollout import (
    PhysicalRightProductionRolloutState,
    sm_physical_right_production_initial_state,
    sm_physical_right_production_step,
)
from clifford_3plus2_d5.sim.profiling import profile_callable
from clifford_3plus2_d5.sim.state import state_norm_squared


class PhysicalRightProductionLocalProfileDiagnostics(NamedTuple):
    """Focused diagnostics for Stage 40 local-force production profiling."""

    site_count: jnp.ndarray
    eager_profile_all_finite: jnp.ndarray
    eager_profile_python_seconds: jnp.ndarray
    eager_profile_payload_key_count: jnp.ndarray
    jit_compile_seconds: jnp.ndarray
    jit_run_seconds: jnp.ndarray
    jit_eager_state_residual: jnp.ndarray
    final_all_finite: jnp.ndarray
    family_norm_drift: jnp.ndarray
    sm_link_unitarity_residual: jnp.ndarray
    higgs_link_unitarity_residual: jnp.ndarray


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _state_max_delta(
    left: PhysicalRightProductionRolloutState,
    right: PhysicalRightProductionRolloutState,
) -> jnp.ndarray:
    return jnp.max(
        jnp.asarray(
            [
                jnp.max(jnp.abs(left.family_state - right.family_state)),
                jnp.max(jnp.abs(left.higgs - right.higgs)),
                jnp.max(jnp.abs(left.higgs_momenta - right.higgs_momenta)),
                jnp.max(jnp.abs(left.sm_links - right.sm_links)),
                jnp.max(jnp.abs(left.sm_momenta - right.sm_momenta)),
                jnp.max(jnp.abs(left.higgs_links - right.higgs_links)),
            ],
        ),
    )


def _state_all_finite(state: PhysicalRightProductionRolloutState) -> jnp.ndarray:
    return jnp.all(
        jnp.asarray(
            [
                jnp.all(jnp.isfinite(state.family_state)),
                jnp.all(jnp.isfinite(state.higgs)),
                jnp.all(jnp.isfinite(state.higgs_momenta)),
                jnp.all(jnp.isfinite(state.sm_links)),
                jnp.all(jnp.isfinite(state.sm_momenta)),
                jnp.all(jnp.isfinite(state.higgs_links)),
            ],
        ),
    )


def _state_summary(state: PhysicalRightProductionRolloutState) -> dict[str, Any]:
    return {
        "type": type(state).__name__,
        "family_state_shape": tuple(int(size) for size in state.family_state.shape),
        "higgs_shape": tuple(int(size) for size in state.higgs.shape),
        "sm_links_shape": tuple(int(size) for size in state.sm_links.shape),
        "sm_momenta_shape": tuple(int(size) for size in state.sm_momenta.shape),
        "higgs_links_shape": tuple(int(size) for size in state.higgs_links.shape),
    }


def sm_physical_right_production_local_profile_diagnostics(
    lattice_shape: tuple[int, int, int] = (2, 1, 1),
) -> PhysicalRightProductionLocalProfileDiagnostics:
    """Return Stage 40 local-force production profiling diagnostics."""

    if len(lattice_shape) != 3 or any(size < 1 for size in lattice_shape):
        raise ValueError(f"lattice_shape must contain three positive sizes, got {lattice_shape}")

    initial = sm_physical_right_production_initial_state(lattice_shape)

    def step_once(state: PhysicalRightProductionRolloutState) -> PhysicalRightProductionRolloutState:
        return sm_physical_right_production_step(state, step_size=0.001)

    eager_profile = profile_callable(
        "qca_smv0_stage40_local_force_step",
        step_once,
        args=(initial,),
        metadata={"lattice_shape": lattice_shape, "step_size": 0.001},
        output_summary_fn=_state_summary,
    )
    eager_final = step_once(initial)
    _block_until_ready(eager_final)

    jitted_step = jax.jit(step_once)
    start = perf_counter()
    compiled_final = jitted_step(initial)
    _block_until_ready(compiled_final)
    compile_seconds = perf_counter() - start

    start = perf_counter()
    jit_final = jitted_step(initial)
    _block_until_ready(jit_final)
    run_seconds = perf_counter() - start

    site_count = lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
    payload = eager_profile.as_payload()
    return PhysicalRightProductionLocalProfileDiagnostics(
        site_count=jnp.asarray(site_count, dtype=jnp.int32),
        eager_profile_all_finite=jnp.asarray(eager_profile.all_finite),
        eager_profile_python_seconds=jnp.asarray(eager_profile.python_seconds, dtype=jnp.float32),
        eager_profile_payload_key_count=jnp.asarray(len(payload), dtype=jnp.int32),
        jit_compile_seconds=jnp.asarray(compile_seconds, dtype=jnp.float32),
        jit_run_seconds=jnp.asarray(run_seconds, dtype=jnp.float32),
        jit_eager_state_residual=_state_max_delta(eager_final, jit_final),
        final_all_finite=_state_all_finite(jit_final),
        family_norm_drift=jnp.abs(state_norm_squared(jit_final.family_state) - state_norm_squared(initial.family_state)),
        sm_link_unitarity_residual=sm_link_unitarity_residual(jit_final.sm_links),
        higgs_link_unitarity_residual=sm_higgs_link_unitarity_residual(jit_final.higgs_links),
    )

