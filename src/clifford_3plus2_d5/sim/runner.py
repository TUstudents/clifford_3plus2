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
    select_observation_steps,
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


def run_recorded_scan(
    initial_state: StateT,
    step_fn: StepFn[StateT],
    observe_fn: ObserveFn[StateT],
    config: GenericRunConfig,
) -> GenericRunResult:
    """Run a ``jax.lax.scan`` simulation and record selected observables."""

    validate_run_config(config)
    initial_observation = observe_fn(initial_state)
    record_indices = recorded_step_indices(config)

    if config.steps == 0:
        observations = stack_observations((initial_observation,))
        return GenericRunResult(
            config=config,
            initial_state=initial_state,
            final_state=initial_state,
            step_indices=jnp.asarray(record_indices, dtype=jnp.int32),
            observations=observations,
            all_finite=_run_all_finite(initial_state, observations),
        )

    def body(carry: StateT, _: object) -> tuple[StateT, ObservableMap]:
        next_state = step_fn(carry)
        return next_state, observe_fn(next_state)

    def execute(state: StateT) -> tuple[StateT, ObservableMap]:
        return jax.lax.scan(body, state, xs=None, length=config.steps)

    executor = jax.jit(execute) if config.use_jit else execute
    final_state, step_observations = executor(initial_state)
    all_observations = {
        key: jnp.concatenate((jnp.expand_dims(jnp.asarray(initial_observation[key]), axis=0), value), axis=0)
        for key, value in step_observations.items()
    }
    selected = select_observation_steps(all_observations, record_indices)
    return GenericRunResult(
        config=config,
        initial_state=initial_state,
        final_state=final_state,
        step_indices=jnp.asarray(record_indices, dtype=jnp.int32),
        observations=selected,
        all_finite=_run_all_finite(final_state, selected),
    )
