"""Generic loop and ``jax.lax.scan`` runners for sidecar simulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp

from clifford_3plus2_d5.sim.observables import (
    ObservableMap,
    observations_all_finite,
    pytree_all_finite,
    stack_observations,
)


StateT = TypeVar("StateT")
StepFn = Callable[[StateT], StateT]
ObserveFn = Callable[[StateT], ObservableMap]


@dataclass(frozen=True)
class GenericRunConfig:
    """Physics-agnostic controls for recorded simulation runners."""

    steps: int = 0
    record_every: int = 1
    use_jit: bool = False


@dataclass(frozen=True)
class GenericRunResult:
    """Generic recorded simulation result."""

    config: GenericRunConfig
    initial_state: object
    final_state: object
    step_indices: jnp.ndarray
    observations: dict[str, jnp.ndarray]
    all_finite: jnp.ndarray


def validate_run_config(config: GenericRunConfig) -> None:
    """Validate generic recorded-run controls."""

    if config.steps < 0:
        raise ValueError(f"steps must be nonnegative, got {config.steps}")
    if config.record_every <= 0:
        raise ValueError(f"record_every must be positive, got {config.record_every}")


def recorded_step_indices(config: GenericRunConfig) -> tuple[int, ...]:
    """Return step indices recorded by loop/scan runners."""

    validate_run_config(config)
    indices = [0]
    for step_index in range(1, config.steps + 1):
        if step_index % config.record_every == 0 or step_index == config.steps:
            indices.append(step_index)
    return tuple(indices)


def _run_all_finite(final_state: object, observations: ObservableMap) -> jnp.ndarray:
    return pytree_all_finite(final_state) & observations_all_finite(observations)


def run_recorded_loop(
    initial_state: StateT,
    step_fn: StepFn[StateT],
    observe_fn: ObserveFn[StateT],
    config: GenericRunConfig,
) -> GenericRunResult:
    """Run a Python-loop simulation and record selected observable steps."""

    validate_run_config(config)
    state = initial_state
    step_indices: list[int] = [0]
    observations: list[ObservableMap] = [observe_fn(state)]

    for step_index in range(1, config.steps + 1):
        state = step_fn(state)
        if step_index % config.record_every == 0 or step_index == config.steps:
            step_indices.append(step_index)
            observations.append(observe_fn(state))

    stacked = stack_observations(tuple(observations))
    return GenericRunResult(
        config=config,
        initial_state=initial_state,
        final_state=state,
        step_indices=jnp.asarray(tuple(step_indices), dtype=jnp.int32),
        observations=stacked,
        all_finite=_run_all_finite(state, stacked),
    )


def _advance_state_with_scan(
    state: StateT,
    step_fn: StepFn[StateT],
    step_count: int,
    *,
    use_jit: bool,
) -> StateT:
    """Advance ``state`` by ``step_count`` steps without observing."""

    if step_count == 0:
        return state

    def body(carry: StateT, _: object) -> tuple[StateT, None]:
        return step_fn(carry), None

    def execute(current: StateT) -> StateT:
        final_state, _ = jax.lax.scan(body, current, xs=None, length=step_count)
        return final_state

    executor = jax.jit(execute) if use_jit else execute
    return executor(state)


def run_recorded_scan(
    initial_state: StateT,
    step_fn: StepFn[StateT],
    observe_fn: ObserveFn[StateT],
    config: GenericRunConfig,
) -> GenericRunResult:
    """Run a ``jax.lax.scan`` simulation and observe only recorded steps."""

    validate_run_config(config)
    record_indices = recorded_step_indices(config)
    state = initial_state
    observations: list[ObservableMap] = [observe_fn(state)]
    current_step = 0

    for record_step in record_indices[1:]:
        state = _advance_state_with_scan(
            state,
            step_fn,
            record_step - current_step,
            use_jit=config.use_jit,
        )
        current_step = record_step
        observations.append(observe_fn(state))

    selected = stack_observations(tuple(observations))
    return GenericRunResult(
        config=config,
        initial_state=initial_state,
        final_state=state,
        step_indices=jnp.asarray(record_indices, dtype=jnp.int32),
        observations=selected,
        all_finite=_run_all_finite(state, selected),
    )
