"""Generic observable-history helpers for simulation runners."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp


ObservableMap: TypeAlias = Mapping[str, jnp.ndarray]


def stack_observations(observations: tuple[ObservableMap, ...]) -> dict[str, jnp.ndarray]:
    """Stack a nonempty tuple of homogeneous observable dictionaries."""

    if not observations:
        return {}
    keys = tuple(observations[0].keys())
    for observation in observations:
        if tuple(observation.keys()) != keys:
            raise ValueError("all observation dictionaries must have identical keys and order")
    return {
        key: jnp.stack(tuple(jnp.asarray(observation[key]) for observation in observations), axis=0)
        for key in keys
    }


def select_observation_steps(
    observations: ObservableMap,
    indices: tuple[int, ...],
) -> dict[str, jnp.ndarray]:
    """Select recorded step indices from stacked observable arrays."""

    return {key: jnp.asarray(value)[jnp.asarray(indices, dtype=jnp.int32)] for key, value in observations.items()}


def observations_all_finite(observations: ObservableMap) -> jnp.ndarray:
    """Return whether every array in an observable dictionary is finite."""

    if not observations:
        return jnp.asarray(True)
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(value)) for value in observations.values()]))


def pytree_all_finite(tree: object) -> jnp.ndarray:
    """Return whether every JAX-array leaf in a pytree is finite."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
